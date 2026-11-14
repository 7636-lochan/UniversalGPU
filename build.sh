#!/bin/bash

echo "======================================"
echo "   Building UniversalGPU Environment  "
echo "======================================"

# 1. Build the Compiler Frontend
echo "[1/2] Building Clang AST Parser..."
cd ~/ugpu/frontend/build
make
if [ $? -ne 0 ]; then
    echo "Error: Compiler build failed."
    exit 1
fi

# 2. Build the Runtime Shim
echo "[2/2] Building libcudart.so (Runtime Shim)..."
cd ~/ugpu/runtime
g++ -shared -fPIC cudart_shim.cpp -o libcudart.so -ldl
if [ $? -ne 0 ]; then
    echo "Error: Shim build failed."
    exit 1
fi

echo "======================================"
echo "   Build Complete!                    "
echo "======================================"
