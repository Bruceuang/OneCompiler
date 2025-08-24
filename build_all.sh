#!/bin/bash
set -e

# 配置
JOBS=${JOBS:-$(nproc)}
BUILD_TYPE=${BUILD_TYPE:-"Release"}

echo "🔧 OneCompiler LLVM专用构建系统"
echo "构建类型: $BUILD_TYPE"
echo "并行作业: $JOBS"

# 步骤1：初始化LLVM子模块（如需要）
if [[ ! -d "third_party/llvm-project" ]] || [[ ! -d "third_party/llvm-project/.git" ]]; then
    echo "➕ 初始化LLVM子模块..."
    ./scripts/init_submodules.sh
fi

# 步骤2：创建构建目录
mkdir -p build
cd build

# 步骤3：配置项目
echo "🔧 配置项目..."
cmake \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    ..

# 步骤4：构建所有内容
echo "🏗️  开始构建..."
cmake --build . --target all -- -j$JOBS

echo "✅ LLVM/MLIR构建完成！"