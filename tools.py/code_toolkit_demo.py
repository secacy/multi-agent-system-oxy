"""
CodeAgent 工具包演示脚本（独立运行）

这个脚本演示了 python_interpreter 工具的核心功能，
不依赖 oxygent 框架，可以直接运行。
"""

import os
import sys
import traceback
from io import StringIO
from pathlib import Path


def python_interpreter_demo(code: str, task_id: str) -> str:
    """
    Python 解释器演示版本
    
    这是一个简化版本，用于演示核心功能。
    实际使用时应该使用 code_toolkit.py 中的完整版本。
    """
    # 创建 local_es_data 目录
    output_dir = Path("local_es_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备文件路径
    code_file = output_dir / f"{task_id}_code.py"
    output_file = output_dir / f"{task_id}_output.txt"
    
    # 保存代码到文件
    try:
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"✅ 代码已保存到: {code_file}")
    except Exception as e:
        error_msg = f"❌ 保存代码文件失败: {str(e)}"
        return error_msg
    
    # 捕获标准输出
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    execution_result = ""
    
    try:
        # 创建执行环境
        safe_globals = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
            "__file__": str(code_file),
        }
        safe_locals = {}
        
        # 执行代码
        exec(code, safe_globals, safe_locals)
        
        # 获取输出
        execution_result = captured_output.getvalue()
        
        if not execution_result:
            execution_result = "✅ 代码执行成功，但没有产生输出。"
        else:
            execution_result = f"✅ 代码执行成功:\n{execution_result}"
            
    except Exception as e:
        error_trace = traceback.format_exc()
        execution_result = f"❌ 代码执行出错:\n{error_trace}"
        
    finally:
        sys.stdout = old_stdout
    
    # 保存输出到文件
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"=== Task ID: {task_id} ===\n\n")
            f.write(f"=== 代码 ===\n{code}\n\n")
            f.write(f"=== 执行结果 ===\n{execution_result}\n")
        print(f"✅ 输出已保存到: {output_file}")
    except Exception as e:
        execution_result += f"\n\n⚠️ 警告: 保存输出文件失败: {str(e)}"
    
    return execution_result


def demo_1_basic_math():
    """演示 1: 基础数学计算"""
    print("\n" + "=" * 70)
    print("演示 1: 基础数学计算 (毒酒问题)")
    print("=" * 70)
    
    code = """
import math

# 毒酒问题：256 瓶酒需要多少只老鼠？
bottles = 256
mice_needed = int(math.log2(bottles))

print(f"问题: {bottles} 瓶酒中有一瓶有毒，需要多少只老鼠才能找出来？")
print(f"\\n原理: 每只老鼠可以表示一个二进制位（活/死）")
print(f"n 只老鼠可以表示 2^n 种不同的状态")
print(f"\\n答案: {mice_needed} 只老鼠")
print(f"验证: 2^{mice_needed} = {2**mice_needed}")
"""
    
    result = python_interpreter_demo(code=code, task_id="demo_math_001")
    print("\n输出:")
    print(result)


def demo_2_light_speed():
    """演示 2: 光速计算"""
    print("\n" + "=" * 70)
    print("演示 2: 光速计算")
    print("=" * 70)
    
    code = """
# 计算以恒定加速度达到光速需要多少天
speed_of_light = 299792458  # 米/秒
acceleration = 10  # 米/秒^2（约为地球重力加速度）

time_seconds = speed_of_light / acceleration
time_days = time_seconds / (24 * 3600)

print(f"光速: {speed_of_light:,} 米/秒")
print(f"加速度: {acceleration} 米/秒²")
print(f"\\n以 {acceleration} 米/秒² 的加速度达到光速:")
print(f"需要时间: {time_seconds:,.2f} 秒")
print(f"等于: {time_days:,.2f} 天")
print(f"约等于: {time_days/365.25:.2f} 年")
"""
    
    result = python_interpreter_demo(code=code, task_id="demo_physics_001")
    print("\n输出:")
    print(result)


def demo_3_file_operations():
    """演示 3: 文件系统操作"""
    print("\n" + "=" * 70)
    print("演示 3: 文件系统操作")
    print("=" * 70)
    
    code = """
import os
import glob

# 创建测试目录结构
base_dir = "local_es_data/demo_files"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(f"{base_dir}/logs", exist_ok=True)
os.makedirs(f"{base_dir}/data", exist_ok=True)

# 创建一些测试文件
test_files = [
    f"{base_dir}/logs/app.log",
    f"{base_dir}/logs/error.log",
    f"{base_dir}/logs/access.log",
    f"{base_dir}/data/config.txt",
    f"{base_dir}/readme.txt",
]

for file_path in test_files:
    with open(file_path, "w") as f:
        f.write(f"测试文件: {os.path.basename(file_path)}\\n")

# 统计 .log 文件
log_files = glob.glob(f"{base_dir}/**/*.log", recursive=True)

print(f"创建的目录: {base_dir}")
print(f"创建的文件总数: {len(test_files)}")
print(f"\\n找到 {len(log_files)} 个 .log 文件:")

for log_file in log_files:
    size = os.path.getsize(log_file)
    rel_path = os.path.relpath(log_file, base_dir)
    print(f"  - {rel_path}: {size} 字节")

print(f"\\n所有文件:")
all_files = glob.glob(f"{base_dir}/**/*", recursive=True)
for file in all_files:
    if os.path.isfile(file):
        rel_path = os.path.relpath(file, base_dir)
        print(f"  - {rel_path}")
"""
    
    result = python_interpreter_demo(code=code, task_id="demo_file_001")
    print("\n输出:")
    print(result)


def demo_4_data_structures():
    """演示 4: 数据结构和算法"""
    print("\n" + "=" * 70)
    print("演示 4: 数据结构和算法（天平称假币）")
    print("=" * 70)
    
    code = """
import math

def min_weighings(num_coins):
    \"\"\"
    计算需要多少次称重才能从 num_coins 个硬币中找出唯一的假币
    
    原理：每次称重可以将硬币分成三组（左、右、不称），
    所以 n 次称重最多可以区分 3^n 个硬币
    \"\"\"
    return math.ceil(math.log(num_coins, 3))

# 测试不同数量的硬币
test_cases = [3, 9, 12, 27, 100]

print("天平称假币问题")
print("-" * 50)
print("问题: 有 N 个硬币，其中一个是假币（重量不同）")
print("工具: 一个天平，可以比较两组硬币的重量")
print("目标: 找出最少需要称重几次")
print()

for coins in test_cases:
    weighings = min_weighings(coins)
    max_coins = 3 ** weighings
    print(f"{coins:3d} 个硬币 → 需要 {weighings} 次称重 "
          f"(最多可区分 {max_coins} 个)")

# 详细解释 12 个硬币的情况
print("\\n" + "=" * 50)
print("详细分析: 12 个硬币的情况")
print("=" * 50)
coins = 12
weighings = min_weighings(coins)
print(f"\\n需要 {weighings} 次称重")
print(f"\\n策略:")
print(f"  第1次: 分成 3 组，每组 4 个 (4-4-4)")
print(f"  第2次: 根据第1次结果，在可疑组中继续分成 3 组")
print(f"  第3次: 最终确定假币")
"""
    
    result = python_interpreter_demo(code=code, task_id="demo_algo_001")
    print("\n输出:")
    print(result)


def demo_5_complex_calculation():
    """演示 5: 复杂计算"""
    print("\n" + "=" * 70)
    print("演示 5: 复杂计算（斐波那契数列和黄金比例）")
    print("=" * 70)
    
    code = """
def fibonacci(n):
    \"\"\"计算斐波那契数列的第 n 项\"\"\"
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 计算前 20 项斐波那契数列
print("斐波那契数列前 20 项:")
print("-" * 50)
fib_numbers = []
for i in range(20):
    fib = fibonacci(i)
    fib_numbers.append(fib)
    if i < 10:
        print(f"F({i:2d}) = {fib:8d}", end="  ")
        if (i + 1) % 3 == 0:
            print()
    else:
        print(f"F({i:2d}) = {fib:8d}")

# 计算相邻项的比值（趋向于黄金比例）
print("\\n" + "=" * 50)
print("相邻项比值 (趋向黄金比例 φ ≈ 1.618)")
print("=" * 50)

golden_ratio = (1 + 5**0.5) / 2
print(f"黄金比例 φ = {golden_ratio:.10f}\\n")

for i in range(5, 20):
    if fib_numbers[i-1] > 0:
        ratio = fib_numbers[i] / fib_numbers[i-1]
        diff = abs(ratio - golden_ratio)
        print(f"F({i:2d})/F({i-1:2d}) = {ratio:.10f}  "
              f"(误差: {diff:.2e})")
"""
    
    result = python_interpreter_demo(code=code, task_id="demo_calc_001")
    print("\n输出:")
    print(result)


def demo_6_error_handling():
    """演示 6: 错误处理"""
    print("\n" + "=" * 70)
    print("演示 6: 错误处理")
    print("=" * 70)
    
    code = """
# 这段代码会故意产生错误，演示错误处理机制
print("开始执行...")
print("计算 10 / 2 =", 10 / 2)
print("计算 10 / 0 =", 10 / 0)  # 这里会出错
print("这行不会被执行")
"""
    
    result = python_interpreter_demo(code=code, task_id="demo_error_001")
    print("\n输出:")
    print(result)


def main():
    """主函数"""
    print("\n")
    print("=" * 70)
    print("CodeAgent 工具包演示")
    print("=" * 70)
    print("\n这个演示展示了 python_interpreter 工具的各种能力:")
    print("  1. 数学计算和逻辑推理")
    print("  2. 文件系统操作")
    print("  3. 复杂算法实现")
    print("  4. 错误处理")
    print("\n所有代码和输出都会保存到 local_es_data/ 目录")
    
    # 运行所有演示
    demo_1_basic_math()
    demo_2_light_speed()
    demo_3_file_operations()
    demo_4_data_structures()
    demo_5_complex_calculation()
    demo_6_error_handling()
    
    # 总结
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\n📁 查看 local_es_data/ 目录可以看到所有保存的文件:")
    
    output_dir = Path("local_es_data")
    if output_dir.exists():
        files = sorted(output_dir.glob("demo_*"))
        if files:
            print(f"\n生成的文件 ({len(files)} 个):")
            for file in files:
                size = file.stat().st_size
                print(f"  - {file.name} ({size} 字节)")
        else:
            print("\n  (没有找到演示文件)")
    
    print("\n" + "=" * 70)
    print("💡 提示:")
    print("  - 查看 README_code_toolkit.md 了解完整使用指南")
    print("  - 在实际使用中，应通过 OxyGent 框架调用这个工具")
    print("  - CodeAgent 会根据任务自动生成并执行 Python 代码")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

