#!/bin/bash
set -e

# 定义LLVM子模块路径
LLVM_SUBMODULE_PATH="third_party/llvm-project"

# 解析命令行参数
FORCE_INIT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            FORCE_INIT=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  -f, --force    强制重新初始化"
            echo "  -h, --help     显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 清理损坏的目录（如果需要强制重新初始化）
if [[ "$FORCE_INIT" == "true" ]] && [[ -d "$LLVM_SUBMODULE_PATH" ]]; then
    echo "🧹 清理LLVM子模块目录..."
    rm -rf "$LLVM_SUBMODULE_PATH"
fi

# 初始化LLVM子模块
echo "➕ 初始化LLVM子模块..."
git submodule update --init --recursive "$LLVM_SUBMODULE_PATH"

echo "✅ LLVM子模块初始化完成！"