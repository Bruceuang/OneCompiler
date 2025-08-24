# OneCompiler - LLVM/MLIR专用版本

基于LLVM/MLIR的编译器框架，专注于提供简洁高效的编译器开发体验。

## 快速开始

### 1. 克隆和初始化
```bash
git clone <repository-url>
cd OneCompiler
```

### 2. 初始化LLVM子模块
```bash
# 初始化LLVM子模块
./scripts/init_submodules.sh

# 或者强制重新初始化
./scripts/init_submodules.sh --force
```

### 3. 构建项目
```bash
# 一键构建
./build_all.sh

# 或者自定义构建
BUILD_TYPE=Debug JOBS=8 ./build_all.sh
```

## 构建配置

### 环境变量
- `BUILD_TYPE`: 构建类型 (Release/Debug/RelWithDebInfo)
- `JOBS`: 并行编译线程数 (默认: CPU核心数)

### 示例
```bash
# Debug构建
BUILD_TYPE=Debug ./build_all.sh

# 使用8线程构建
JOBS=8 ./build_all.sh

# 完全清理后重新构建
rm -rf build && ./build_all.sh
```

## 项目结构