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

## 运行测试

### 运行编译器

构建完成后，编译器位于 `build/src/tools/onecompiler`：

```bash
# 查看版本
./build/src/tools/onecompiler --version

# 查看帮助
./build/src/tools/onecompiler --help
```

### 运行简单测试用例

项目提供了一个基本的MLIR测试用例，位于 `test/mlir/basic.mlir`：

```bash
# 使用编译器处理MLIR文件（输出到output.mlir）
./build/src/tools/onecompiler --emit-mlir test/mlir/basic.mlir

# 启用详细输出
./build/src/tools/onecompiler -v --emit-mlir test/mlir/basic.mlir

# 输出到标准输出
./build/src/tools/onecompiler --emit-mlir test/mlir/basic.mlir -o -

# 指定输出文件
./build/src/tools/onecompiler --emit-mlir test/mlir/basic.mlir -o result.mlir
```

### 测试用例说明

`test/mlir/basic.mlir` 包含一个简单的加法函数：

```mlir
module {
  func.func @test_add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    func.return %0 : i32
  }
}
```

### 预期输出

运行测试后，输出文件将包含解析后的MLIR：

```mlir
module {
  func.func @test_add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
}
```

### 使用lit运行测试套件（可选）

如果需要运行完整的测试套件，需要安装lit和FileCheck：

```bash
# 安装lit
pip install lit

# 运行测试
cd build
ninja check-one-mlir
```

## 项目结构
