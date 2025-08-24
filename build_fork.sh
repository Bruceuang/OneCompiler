#!/bin/bash

set -e

# 配置
JOBS=${JOBS:-$(nproc)}
BUILD_TYPE=${BUILD_TYPE:-"Release"}

echo "🔧 OneCompiler fork版LLVM构建"
echo "fork地址: https://github.com/Bruceuang/llvm-project.git"
echo "分支: onecompiler"
echo "并行作业: $JOBS"

# 检查fork版LLVM
if [ ! -d "third_party/llvm-project" ]; then
    echo "❌ fork版LLVM未找到，请先运行 ./init_fork_llvm.sh"
    exit 1
fi

# 创建构建目录
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# 配置CMake使用fork版LLVM
cmake \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DLLVM_DIR=$(pwd)/../third_party/llvm-project/llvm \
    -DMLIR_DIR=$(pwd)/../third_party/llvm-project/mlir \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    ..

# 构建项目
cmake --build . --target all -- -j$JOBS

# 运行测试
echo "🧪 运行测试..."
ctest --output-on-failure -j$JOBS

echo "✅ fork版LLVM构建完成！"
