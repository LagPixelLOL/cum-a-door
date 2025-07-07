#include <cstdint>
#include <climits>

using c64ptr_t = uint16_t;
#define MAKE_VTPTR(typename, address) reinterpret_cast<volatile typename*>(static_cast<c64ptr_t>(address))

enum class c64color_t : uint8_t {
    BLACK, WHITE, RED, CYAN,
    PURPLE, GREEN, BLUE, YELLOW,
    ORANGE, BROWN, PINK, DARK_GREY,
    GREY, LIGHT_GREEN, LIGHT_BLUE, LIGHT_GREY,
    END, START = BLACK
};

inline void set_border_color(c64color_t color) {
    auto bcr = MAKE_VTPTR(uint8_t, 0xd020);
    *bcr = static_cast<uint8_t>(color);
}

constexpr uint8_t byte_replace_bits(uint8_t source, uint8_t offset, uint8_t target, uint8_t target_len = 1) {
    uint8_t mask = ~(0b11111111 >> (CHAR_BIT - target_len) << offset);
    return (source & mask) | target << offset;
}

inline void wait_for_n_loops(uint16_t n_loops) {
    for (uint16_t i = 0; i < n_loops; ++i) {
        asm ("");
    }
}

inline void set_raster_interrupt_line(uint8_t line) {
    auto crl = MAKE_VTPTR(uint8_t, 0xd012);
    *crl = line;
}

__attribute__((interrupt)) __attribute__((no_isr)) void interrupt_handler() {
    set_border_color(c64color_t::DARK_GREY);
    wait_for_n_loops(14);
    set_border_color(c64color_t::RED);
    wait_for_n_loops(12);
    set_border_color(c64color_t::DARK_GREY);
    wait_for_n_loops(12);
    set_border_color(c64color_t::ORANGE);
    wait_for_n_loops(11);
    set_border_color(c64color_t::DARK_GREY);
    wait_for_n_loops(11);
    set_border_color(c64color_t::YELLOW);
    wait_for_n_loops(11);
    set_border_color(c64color_t::DARK_GREY);
    wait_for_n_loops(11);
    set_border_color(c64color_t::GREEN);
    wait_for_n_loops(11);
    set_border_color(c64color_t::DARK_GREY);
    wait_for_n_loops(12);
    set_border_color(c64color_t::CYAN);
    wait_for_n_loops(11);
    set_border_color(c64color_t::DARK_GREY);
    wait_for_n_loops(11);
    set_border_color(c64color_t::BLACK);
}

__attribute__((naked)) void bare_interrupt_caller() {
    asm (
        "jsr %0\n"
        "asl 0xd019\n"
        "jmp 0xea31\n"
        :
        : "i"(interrupt_handler)
    );
}

int main() {
    asm ("sei");
    uint8_t cia_flag = 0b01111111;
    auto cia1 = MAKE_VTPTR(uint8_t, 0xdc0d);
    *cia1 = cia_flag;
    // auto cia2 = MAKE_VTPTR(uint8_t, 0xdd0d);
    // *cia2 = cia_flag;
    auto scr = MAKE_VTPTR(uint8_t, 0xd011);
    *scr = byte_replace_bits(*scr, 7, 0);
    set_raster_interrupt_line(14);
    auto isr = MAKE_VTPTR(c64ptr_t, 0x0314);
    *isr = reinterpret_cast<c64ptr_t>(bare_interrupt_caller);
    auto icr = MAKE_VTPTR(uint8_t, 0xd01a);
    *icr = 1;
    asm ("cli");
    return 0;
}
