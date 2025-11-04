#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
京东多智能体挑战赛 - 评估脚本
用于对比验证集的标准答案和模型输出结果
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_jsonl(file_path: str) -> Dict[str, dict]:
    """
    加载 JSONL 文件并以 task_id 为 key 构建字典
    
    Args:
        file_path: JSONL 文件路径
        
    Returns:
        {task_id: {完整的任务数据}} 字典
    """
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                task_id = item.get('task_id')
                if not task_id:
                    print(f"⚠️  警告: 第 {line_num} 行缺少 task_id，跳过")
                    continue
                if task_id in data_dict:
                    print(f"⚠️  警告: task_id '{task_id}' 重复出现")
                data_dict[task_id] = item
            except json.JSONDecodeError as e:
                print(f"❌ 错误: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
    
    return data_dict


def normalize_answer(answer: str) -> str:
    """
    规范化答案文本（去除首尾空白）
    
    Args:
        answer: 原始答案
        
    Returns:
        规范化后的答案
    """
    if answer is None:
        return ""
    return str(answer).strip()


def compare_answers(predicted: str, ground_truth: str) -> bool:
    """
    对比预测答案和标准答案
    当前实现：严格字符串匹配（去除首尾空白后）
    
    Args:
        predicted: 模型预测的答案
        ground_truth: 标准答案
        
    Returns:
        True 表示答案正确，False 表示错误
    """
    # 严格对比模式
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def evaluate(ground_truth_file: str, 
             prediction_file: str, 
             verbose: bool = True,
             show_errors_in_terminal: bool = False,
             only_predicted: bool = False) -> Tuple[float, List[dict], Dict]:
    """
    评估模型输出结果
    
    Args:
        ground_truth_file: 标准答案文件路径
        prediction_file: 模型预测文件路径
        verbose: 是否输出详细信息
        show_errors_in_terminal: 是否在终端显示错误详情
        only_predicted: 是否只评测预测文件中包含的 task_id
        
    Returns:
        (准确率, 错误列表, 统计信息字典)
    """
    print("=" * 80)
    print("🚀 京东多智能体挑战赛 - 评估系统")
    print("=" * 80)
    
    # 1. 加载数据
    print(f"\n📂 加载标准答案: {ground_truth_file}")
    ground_truth_dict = load_jsonl(ground_truth_file)
    print(f"   ✅ 加载了 {len(ground_truth_dict)} 条标准答案")
    
    print(f"\n📂 加载模型预测: {prediction_file}")
    prediction_dict = load_jsonl(prediction_file)
    print(f"   ✅ 加载了 {len(prediction_dict)} 条预测结果")
    
    # 2. 检查 task_id 覆盖情况
    ground_truth_ids = set(ground_truth_dict.keys())
    prediction_ids = set(prediction_dict.keys())
    
    missing_ids = ground_truth_ids - prediction_ids
    extra_ids = prediction_ids - ground_truth_ids
    
    # 决定评测范围
    if only_predicted:
        eval_ids = prediction_ids & ground_truth_ids  # 只评测两者交集
        print(f"\n📌 模式: 只评测预测文件中包含的 task_id")
        print(f"   评测范围: {len(eval_ids)} 个任务")
        if extra_ids:
            print(f"   ⚠️  发现 {len(extra_ids)} 个预测文件中多余的 task_id（不在标准答案中）")
    else:
        eval_ids = ground_truth_ids  # 评测所有标准答案
        if missing_ids:
            print(f"\n⚠️  警告: 缺失 {len(missing_ids)} 个 task_id 的预测结果")
            if verbose and len(missing_ids) <= 10:
                print(f"   缺失的 task_id: {list(missing_ids)[:10]}")
        
        if extra_ids:
            print(f"\n⚠️  警告: 发现 {len(extra_ids)} 个多余的 task_id")
            if verbose and len(extra_ids) <= 10:
                print(f"   多余的 task_id: {list(extra_ids)[:10]}")
    
    # 3. 逐个对比答案
    print("\n" + "=" * 80)
    print("🔍 开始评估...")
    print("=" * 80)
    
    correct_count = 0
    total_count = len(eval_ids)
    errors = []
    correct_items = []
    level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for task_id in eval_ids:
        ground_truth_item = ground_truth_dict[task_id]
        ground_truth_answer = ground_truth_item.get('answer', '')
        level = ground_truth_item.get('level', 'unknown')
        query = ground_truth_item.get('query', '')
        
        # 统计各级别任务
        level_stats[level]["total"] += 1
        
        if task_id not in prediction_dict:
            # 缺失预测（在 only_predicted 模式下不会发生）
            errors.append({
                'task_id': task_id,
                'level': level,
                'query': query,
                'query_short': query[:100] + '...' if len(query) > 100 else query,
                'ground_truth': ground_truth_answer,
                'predicted': '[缺失预测]',
                'status': '❌ 缺失'
            })
        else:
            predicted_answer = prediction_dict[task_id].get('answer', '')
            
            if compare_answers(predicted_answer, ground_truth_answer):
                correct_count += 1
                level_stats[level]["correct"] += 1
                correct_items.append({
                    'task_id': task_id,
                    'level': level,
                    'query': query,
                    'answer': ground_truth_answer
                })
            else:
                errors.append({
                    'task_id': task_id,
                    'level': level,
                    'query': query,
                    'query_short': query[:100] + '...' if len(query) > 100 else query,
                    'ground_truth': ground_truth_answer,
                    'predicted': predicted_answer,
                    'status': '❌ 错误'
                })
    
    # 4. 输出结果
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\n{'=' * 80}")
    print(f"📊 评估结果")
    print(f"{'=' * 80}")
    print(f"\n✅ 总体准确率: {correct_count}/{total_count} ({accuracy:.2f}%)")
    
    # 按级别统计
    print(f"\n📈 分级别统计:")
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        level_acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"   Level {level}: {stats['correct']}/{stats['total']} ({level_acc:.2f}%)")
    
    # 错误摘要
    if errors:
        print(f"\n❌ 错误数量: {len(errors)} 个")
        
        # 只在启用时显示终端详情
        if show_errors_in_terminal:
            print(f"\n{'=' * 80}")
            print(f"❌ 错误详情")
            print(f"{'=' * 80}\n")
            
            for idx, error in enumerate(errors, 1):
                print(f"[{idx}] Task ID: {error['task_id']}")
                print(f"    级别: Level {error['level']}")
                print(f"    问题: {error['query_short']}")
                print(f"    标准答案: {error['ground_truth']}")
                print(f"    模型答案: {error['predicted']}")
                print(f"    状态: {error['status']}")
                print()
        else:
            print(f"   💡 提示: 使用 --show_errors 参数可在终端查看错误详情")
            print(f"   💡 提示: 使用 --output 参数可将完整报告保存到文件")
    else:
        print(f"\n🎉 恭喜！所有答案都正确！")
    
    print("=" * 80)
    
    # 构建统计信息
    stats_info = {
        'total': total_count,
        'correct': correct_count,
        'errors': len(errors),
        'accuracy': accuracy,
        'level_stats': dict(level_stats),
        'missing_count': len(missing_ids) if not only_predicted else 0,
        'extra_count': len(extra_ids),
        'evaluation_mode': 'only_predicted' if only_predicted else 'full'
    }
    
    return accuracy, errors, stats_info


def save_error_report(errors: List[dict], output_file: str):
    """
    保存错误报告到 JSON 文件
    
    Args:
        errors: 错误列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    print(f"\n💾 错误报告 (JSON) 已保存至: {output_file}")


def save_full_report(stats_info: Dict, errors: List[dict], output_file: str):
    """
    保存完整的评估报告到文本文件
    
    Args:
        stats_info: 统计信息字典
        errors: 错误列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("京东多智能体挑战赛 - 完整评估报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. 基本信息
        f.write("📊 评估摘要\n")
        f.write("-" * 80 + "\n")
        f.write(f"评估模式: {'只评测预测任务' if stats_info['evaluation_mode'] == 'only_predicted' else '完整评测'}\n")
        f.write(f"总任务数: {stats_info['total']}\n")
        f.write(f"正确数量: {stats_info['correct']}\n")
        f.write(f"错误数量: {stats_info['errors']}\n")
        f.write(f"总体准确率: {stats_info['accuracy']:.2f}%\n\n")
        
        # 2. 分级别统计
        f.write("📈 分级别统计\n")
        f.write("-" * 80 + "\n")
        for level in sorted(stats_info['level_stats'].keys()):
            level_stat = stats_info['level_stats'][level]
            level_acc = (level_stat['correct'] / level_stat['total'] * 100) if level_stat['total'] > 0 else 0
            f.write(f"Level {level}: {level_stat['correct']}/{level_stat['total']} ({level_acc:.2f}%)\n")
        f.write("\n")
        
        # 3. 错误详情
        if errors:
            f.write("=" * 80 + "\n")
            f.write(f"❌ 错误详情 (共 {len(errors)} 个)\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, error in enumerate(errors, 1):
                f.write(f"[{idx}] Task ID: {error['task_id']}\n")
                f.write(f"    级别: Level {error['level']}\n")
                f.write(f"    状态: {error['status']}\n")
                f.write(f"    问题: {error['query']}\n")
                f.write(f"    标准答案: {error['ground_truth']}\n")
                f.write(f"    模型答案: {error['predicted']}\n")
                f.write("\n" + "-" * 80 + "\n\n")
        else:
            f.write("🎉 恭喜！所有答案都正确！\n")
        
        f.write("=" * 80 + "\n")
        f.write("报告结束\n")
        f.write("=" * 80 + "\n")
    
    print(f"💾 完整评估报告 (文本) 已保存至: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='京东多智能体挑战赛评估脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础用法（默认：只评测预测任务，保存报告到 outputs/report.txt）
  python test/evaluate.py
  
  # 指定文件路径
  python test/evaluate.py --ground_truth data/validation_set.jsonl --prediction outputs/my_validation_run.jsonl
  
  # 评测所有任务（包括缺失的预测）
  python test/evaluate.py --no_only_predicted
  
  # 在终端显示错误详情
  python test/evaluate.py --show_errors
  
  # 自定义报告输出路径
  python test/evaluate.py --output outputs/custom_report.txt
  
  # 保存错误列表为 JSON 格式
  python test/evaluate.py --save_errors
        """
    )
    
    parser.add_argument(
        '--ground_truth',
        type=str,
        default='data/validation_set.jsonl',
        help='标准答案文件路径 (默认: data/validation_set.jsonl)'
    )
    
    parser.add_argument(
        '--prediction',
        type=str,
        default='outputs/my_validation_run.jsonl',
        help='模型预测文件路径 (默认: outputs/my_validation_run.jsonl)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/report.txt',
        help='完整评估报告输出文件路径（文本格式，包含所有错误详情）(默认: outputs/report.txt)'
    )
    
    parser.add_argument(
        '--save_errors',
        action='store_true',
        help='保存错误列表到 JSON 文件'
    )
    
    parser.add_argument(
        '--error_output',
        type=str,
        default='outputs/error_report.json',
        help='错误报告 JSON 文件输出路径 (默认: outputs/error_report.json)'
    )
    
    parser.add_argument(
        '--show_errors',
        action='store_true',
        help='在终端显示所有错误详情'
    )
    
    parser.add_argument(
        '--only_predicted',
        action='store_true',
        default=True,
        help='只评测预测文件中包含的 task_id（默认启用，使用 --no_only_predicted 禁用）'
    )
    
    parser.add_argument(
        '--no_only_predicted',
        action='store_true',
        help='评测所有标准答案中的任务（包括缺失的预测）'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='精简输出模式（不显示详细警告）'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.ground_truth).exists():
        print(f"❌ 错误: 标准答案文件不存在: {args.ground_truth}")
        return
    
    if not Path(args.prediction).exists():
        print(f"❌ 错误: 预测文件不存在: {args.prediction}")
        return
    
    # 处理 only_predicted 参数
    # 如果指定了 --no_only_predicted，则覆盖默认值
    only_predicted = args.only_predicted and not args.no_only_predicted
    
    # 执行评估
    accuracy, errors, stats_info = evaluate(
        args.ground_truth,
        args.prediction,
        verbose=not args.quiet,
        show_errors_in_terminal=args.show_errors,
        only_predicted=only_predicted
    )
    
    # 保存完整报告到文本文件
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save_full_report(stats_info, errors, args.output)
    
    # 保存错误列表到 JSON 文件
    if args.save_errors and errors:
        Path(args.error_output).parent.mkdir(parents=True, exist_ok=True)
        save_error_report(errors, args.error_output)


if __name__ == '__main__':
    main()
