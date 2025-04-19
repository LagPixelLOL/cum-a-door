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

uint8_t rotate_left(uint8_t x, uint8_t shift) {
    constexpr uint8_t bit_length = sizeof(x) * CHAR_BIT;
    // shift %= bit_length; // Assume shift <= bit_length.
    return x << shift | x >> (bit_length - shift);
}

bool byte_select_bit(uint8_t source, uint8_t idx) {
    return source & rotate_left(1, idx);
}

uint8_t byte_replace_bit(uint8_t source, uint8_t idx, bool target) {
    if (target)
        return source | rotate_left(1, idx);
    else
        return source & rotate_left(0b11111110, idx);
}

// __attribute__((noinline))
void set_graphics_mode(c64graphics_t graphics_mode) {
    auto scr1 = MAKE_VTPTR(uint8_t, 0xd011);
    *scr1 = byte_replace_bit(*scr1, 6, byte_select_bit(static_cast<uint8_t>(graphics_mode), 2));

    bool bmm = byte_select_bit(static_cast<uint8_t>(graphics_mode), 1);

    // If using bitmap mode, move bitmap memory to 0x6000 and screen memory (color) to 0x5c00.
    auto vbp = MAKE_VTPTR(uint8_t, 0xdd00);
    *vbp = byte_replace_bit(*vbp, 0, !bmm);
    auto vmo = MAKE_VTPTR(uint8_t, 0xd018);
    *vmo = byte_replace_bit(*vmo, 3, bmm);
    *vmo = byte_replace_bit(*vmo, 5, bmm);
    *vmo = byte_replace_bit(*vmo, 6, bmm);

    *scr1 = byte_replace_bit(*scr1, 5, bmm);

    auto scr2 = MAKE_VTPTR(uint8_t, 0xd016);
    *scr2 = byte_replace_bit(*scr2, 4, byte_select_bit(static_cast<uint8_t>(graphics_mode), 0));
}

void display_img_sbm(const uint8_t* data) {
    set_graphics_mode(c64graphics_t::INVALID);
    for (c64ptr_t i = 0x6000; i < 0x6000 + 8000; ++i) {
        auto target = MAKE_VTPTR(uint8_t, i);
        *target = data[i - 0x6000 + 1];
    }
    for (c64ptr_t i = 0x5c00; i < 0x5c00 + 1000; ++i) {
        auto target = MAKE_VTPTR(uint8_t, i);
        *target = data[i - 0x5c00 + 1 + 8000];
    }
    set_graphics_mode(c64graphics_t::SBM);
}

int main() {
    uint8_t metadata = img[0];
    if (byte_select_bit(metadata, 7)) { // Check the highest bit of the first byte of the image to see if it uses RLE.
        puts("Run length encoded image is not yet supported!");
        return 1;
    }
    uint8_t graphics_mode = metadata & 0b111;
    if (graphics_mode != static_cast<uint8_t>(c64graphics_t::SBM)) {
        puts("Non standard bitmap mode is not yet supported!");
        return 2;
    }

    set_border_color(c64color_t::BLACK);
    display_img_sbm(img);
    return 0;
}
