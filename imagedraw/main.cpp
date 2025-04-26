#include <cstdio>
#include <cstdint>
#include <climits>

using c64ptr_t = uint16_t;
#define MAKE_VTPTR(typename, address) reinterpret_cast<volatile typename*>(static_cast<c64ptr_t>(address))

constexpr uint8_t img[] = {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wc23-extensions"
    #embed "img.c64bmp"
    #pragma GCC diagnostic pop
};

enum class c64color_t : uint8_t {
    BLACK, WHITE, RED, CYAN,
    PURPLE, GREEN, BLUE, YELLOW,
    ORANGE, BROWN, PINK, DARK_GREY,
    GREY, LIGHT_GREEN, LIGHT_BLUE, LIGHT_GREY,
    END, START = BLACK
};

enum class c64graphics_t : uint8_t {
    SCM, MCM, SBM, MBM, EBCM, INVALID = 0b111,
    END, START = SCM
};

inline void set_border_color(c64color_t color) {
    auto bcr = MAKE_VTPTR(uint8_t, 0xd020);
    *bcr = static_cast<uint8_t>(color);
}

inline void set_background_color(c64color_t color) {
    auto bcr = MAKE_VTPTR(uint8_t, 0xd021);
    *bcr = static_cast<uint8_t>(color);
}

bool byte_select_bit(uint8_t source, uint8_t idx) {
    return source & 1 << idx;
}

constexpr uint8_t byte_replace_bits(uint8_t source, uint8_t offset, uint8_t target, uint8_t target_len = 1) {
    uint8_t mask = ~(0b11111111 >> (CHAR_BIT - target_len) << offset);
    return (source & mask) | target << offset;
}

template<c64ptr_t vbp_start_addr>
constexpr uint8_t get_vic_bank_partition_bits() {
    static_assert(vbp_start_addr % 0x4000 == 0, "VIC bank partition must start at multiple of 0x4000!");
    return static_cast<uint8_t>(~byte_replace_bits(vbp_start_addr / 0x4000, 2, 0b00111111, 6));
}

template<c64ptr_t screen_start_offset, c64ptr_t bitmap_start_offset, c64ptr_t character_start_offset = 0xffff>
constexpr uint8_t get_vic_memory_offset() {
    static_assert(screen_start_offset < 0x4000, "Screen memory offset must be less than 0x4000, since this is an offset inside the VIC bank partition!");
    static_assert(screen_start_offset % 0x0400 == 0, "Screen memory offset must start at multiple of 0x0400!");

    constexpr bool bitmap_provided = bitmap_start_offset != 0xffff;
    constexpr bool chara_provided = character_start_offset != 0xffff;
    static_assert(bitmap_provided ^ chara_provided, "You can only set one of bitmap start offset or character start offset, since they override each other!");

    uint8_t result = screen_start_offset / 0x0400 << 4;
    if constexpr (bitmap_provided) {
        static_assert(bitmap_start_offset < 0x4000, "Bitmap memory offset must be less than 0x4000, since this is an offset inside the VIC bank partition!");
        static_assert(bitmap_start_offset % 0x2000 == 0, "Bitmap memory offset must start at multiple of 0x2000!");
        result = byte_replace_bits(result, 3, bitmap_start_offset / 0x2000);
    } else if (chara_provided) {
        static_assert(character_start_offset < 0x4000, "Character memory offset must be less than 0x4000, since this is an offset inside the VIC bank partition!");
        static_assert(character_start_offset % 0x0800 == 0, "Character memory offset must start at multiple of 0x0800!");
        result = byte_replace_bits(result, 1, character_start_offset / 0x0800, 3);
    }
    return result;
}

// __attribute__((noinline))
void set_graphics_mode(c64graphics_t graphics_mode) {
    auto scr1 = MAKE_VTPTR(uint8_t, 0xd011);
    *scr1 = byte_replace_bits(*scr1, 6, byte_select_bit(static_cast<uint8_t>(graphics_mode), 2));

    bool bmm = byte_select_bit(static_cast<uint8_t>(graphics_mode), 1);

    // If using bitmap mode, move bitmap memory to 0x6000 and screen memory (color) to 0x5c00.
    auto vbp = MAKE_VTPTR(uint8_t, 0xdd00);
    *vbp = byte_replace_bits(*vbp, 0, bmm ? get_vic_bank_partition_bits<0x4000>() : get_vic_bank_partition_bits<0>(), 2);
    auto vmo = MAKE_VTPTR(uint8_t, 0xd018);
    *vmo = bmm ? get_vic_memory_offset<0x1c00, 0x2000>() : get_vic_memory_offset<0x0400, 0xffff, 0x1800>();

    *scr1 = byte_replace_bits(*scr1, 5, bmm);

    auto scr2 = MAKE_VTPTR(uint8_t, 0xd016);
    *scr2 = byte_replace_bits(*scr2, 4, byte_select_bit(static_cast<uint8_t>(graphics_mode), 0));
}

void memset_color_ram(const uint8_t* data) {
    data = &data[9001];
    for (c64ptr_t i = 0; i < 500; ++i) {
        auto target = MAKE_VTPTR(uint8_t, i * 2 + 0xd800);
        *target = data[i] >> 4;
        target = MAKE_VTPTR(uint8_t, i * 2 + 0xd801);
        *target = data[i]; // No need to do & 0b1111 since the upper half is unused.
    }
}

void memset_img(const uint8_t* data, c64ptr_t bitmap_mem_addr, c64ptr_t screen_mem_addr) {
    data = &data[1];
    for (c64ptr_t i = screen_mem_addr; i < screen_mem_addr + 1000; ++i) {
        auto target = MAKE_VTPTR(uint8_t, i);
        *target = data[i - screen_mem_addr + 8000];
    }
    for (c64ptr_t i = 0; i < 8000; ++i) {
        auto target = MAKE_VTPTR(uint8_t, bitmap_mem_addr + i / 320 * 320 + i % 320 / 40 + i % 40 * 8);
        *target = data[i];
    }
}

void display_img_sbm(const uint8_t* data) {
    set_graphics_mode(c64graphics_t::SBM);
    memset_img(data, 0x6000, 0x5c00);
}

void display_img_mbm(const uint8_t* data) {
    set_graphics_mode(c64graphics_t::MBM);
    set_background_color(static_cast<c64color_t>(data[9501]));
    memset_color_ram(data);
    memset_img(data, 0x6000, 0x5c00);
}

int main() {
    set_border_color(c64color_t::BLACK);
    uint8_t metadata = img[0];
    if (byte_select_bit(metadata, 7)) { // Check the highest bit of the first byte of the image to see if it uses RLE.
        puts("Run length encoded image is not yet supported!");
        return 1;
    }
    c64graphics_t graphics_mode = static_cast<c64graphics_t>(metadata & 0b111);
    switch (graphics_mode) {
        case c64graphics_t::SBM:
            display_img_sbm(img);
            break;
        case c64graphics_t::MBM:
            display_img_mbm(img);
            break;
        default:
            puts("Modes except SBM and MBM are not yet supported!");
            return 2;
    }
    return 0;
}
