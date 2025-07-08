#include <cstdint>
#include <climits>
#include <cstdlib>

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
    asm (
        "ldx #%0\n"
        "stx 0xd020"
        :
        : "n"(color)
    );
}

constexpr uint8_t byte_replace_bits(uint8_t source, uint8_t offset, uint8_t target, uint8_t target_len = 1) {
    uint8_t mask = ~(0b11111111 >> (CHAR_BIT - target_len) << offset);
    return (source & mask) | target << offset;
}

template<size_t N>
__attribute__((always_inline)) inline void nops() {
    #pragma unroll
    for (size_t i = 0; i < N; ++i) {
        asm ("nop");
    }
}

template<size_t N>
__attribute__((always_inline)) inline void cycles_1() {
    static_assert(N >= 2, "N must be at least 2!");
    constexpr bool exact = N % 2 == 0;
    constexpr size_t n_nop = exact ? N / 2 : N / 2 - 1;
    nops<n_nop>();
    if constexpr (exact) {
        return;
    }
    asm (
        "jmp next%=\n"
        "next%=:"
        :
    );
}

template<size_t N>
__attribute__((always_inline)) inline void cycles_2() {
    static_assert(N >= 2, "N must be at least 2!");
    constexpr size_t n_cycle = N - 2;
    constexpr bool exact = n_cycle % 5 == 4;
    constexpr bool single_remained = n_cycle % 5 == 0;
    constexpr size_t n_loop = N < 4 ? 0 : single_remained ? n_cycle / 5 - 1 : exact ? n_cycle / 5 + 1 : n_cycle / 5;
    if constexpr (n_loop) {
        asm (
            "ldx #%0\n"
            "loop%=:\n"
            "dex\n"
            "bne loop%="
            :
            : "n"(n_loop)
        );
    }
    constexpr size_t remainder = n_loop ? n_cycle - n_loop * 5 + 1 : n_cycle + 2;
    if constexpr (!remainder) {
        return;
    }
    constexpr bool remainder_exact = remainder % 2 == 0;
    constexpr size_t n_left = remainder_exact ? remainder / 2 : remainder / 2 - 1;
    nops<n_left>();
    if constexpr (remainder_exact) {
        return;
    }
    asm (
        "jmp next%=\n"
        "next%=:"
        :
    );
}

#define cycles cycles_2

inline void set_raster_interrupt_line(uint8_t line) {
    auto crl = MAKE_VTPTR(uint8_t, 0xd012);
    *crl = line;
}

__attribute__((interrupt)) __attribute__((no_isr)) void interrupt_handler() {
    cycles<11>();
    set_border_color(c64color_t::DARK_GREY);
    cycles<57>();
    set_border_color(c64color_t::RED);
    cycles<57>();
    set_border_color(c64color_t::DARK_GREY);
    cycles<57>();
    set_border_color(c64color_t::ORANGE);
    cycles<57>();
    set_border_color(c64color_t::DARK_GREY);
    cycles<57>();
    set_border_color(c64color_t::YELLOW);
    cycles<57>();
    set_border_color(c64color_t::DARK_GREY);
    cycles<57>();
    set_border_color(c64color_t::GREEN);
    cycles<57>();
    set_border_color(c64color_t::DARK_GREY);
    cycles<57>();
    set_border_color(c64color_t::CYAN);
    cycles<57>();
    set_border_color(c64color_t::DARK_GREY);
    cycles<57>();
    set_border_color(c64color_t::BLACK);
}

__attribute__((naked)) void bare_interrupt_caller() {
    asm (
        "jsr %0\n"
        "asl 0xd019\n"
        "jmp 0xea31"
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
