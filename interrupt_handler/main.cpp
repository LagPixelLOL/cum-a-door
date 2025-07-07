#include <cstdio>

using c64ptr_t = uint16_t;
#define MAKE_VTPTR(typename, address) reinterpret_cast<volatile typename*>(static_cast<c64ptr_t>(address))

void asdf() {
    auto f = MAKE_VTPTR(uint8_t, 0x0400);
    *f = 1;
    asm volatile ("");
}

void ghjk() {
    auto f = MAKE_VTPTR(uint8_t, 0x0401);
    *f = 2;
    asm volatile ("");
}

__attribute__((interrupt)) __attribute__((no_isr)) void interrupt_handler() {
    asdf();
    ghjk();
}

__attribute__((naked)) void bare_interrupt_caller() {
    asm volatile (
        "jsr %0\n"
        "jmp 0xea31\n"
        :
        : "i"(interrupt_handler)
    );
}

int main() {
    asm volatile ("sei");
    auto isr = MAKE_VTPTR(c64ptr_t, 0x0314);
    *isr = reinterpret_cast<c64ptr_t>(bare_interrupt_caller);
    asm volatile ("cli");
    while (true) {
        printf("x");
    }
    return 0;
}
