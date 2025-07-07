#!/bin/bash

set -e
set -u
set -o pipefail

../llvm-mos/bin/mos-c64-clang++ -Os -Wall -o ../c64-emu/out.prg main.cpp
