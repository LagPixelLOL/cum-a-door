#include <cstdio>
#include <cstdint>

using c64ptr_t = uint16_t;
#define MAKE_VTPTR(typename, address) reinterpret_cast<volatile typename*>(static_cast<c64ptr_t>(address))

constexpr uint8_t img[] = {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wc23-extensions"
    #embed "img.c64bmp"
    #pragma GCC diagnostic pop
};

void set_standard_bitmap_mode() {
    // Set to standard bitmap mode.
    // auto ecm = MAKE_VTPTR(uint8_t, 0xd011);
    // *ecm &= 0b10111111;
    auto bmm = MAKE_VTPTR(uint8_t, 0xd011);
    *bmm |= 0b00100000;
    // auto mcm = MAKE_VTPTR(uint8_t, 0xd016);
    // *mcm &= 0b11101111;

    // Fix a single bit to place bitmap address at 0x2000.
    auto lfg = MAKE_VTPTR(uint8_t, 0xd018);
    *lfg |= 0b00001000;
}

int main() {
    set_standard_bitmap_mode();
    uint8_t c = 0;
    while (true) {
        for (c64ptr_t i = 0x2000; i <= 0x3fff; ++i) {
            auto target = MAKE_VTPTR(uint8_t, i);
            *target = c;
        }
        for (c64ptr_t i = 0x0400; i <= 0x07ff; ++i) {
            auto target = MAKE_VTPTR(uint8_t, i);
            *target = c;
        }
        ++c;
    }
    return 0;
}
