#!/bin/bash
set -e

clean() {
    echo "Cleaning up..."
    rm -rf ./build ./final *.out *.err outfile
}

build() {
    echo "Building the project..."
    cmake -S . -B build
    cmake --build build -j8
    cp ./build/final ./final
}


case "$1" in
    clean) clean ;;
    build) build ;;
    *) echo "Usage: $0 {clean|build}" ;;
esac
