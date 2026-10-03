#!/usr/bin/env python
"""
3 随机种子消融实验 — 7 模型 × 3 种子
独立脚本，不依赖庞然大物 exp_chronological.py

用法:
    PYTHONIOENCODING=utf-8 python run_3seeds.py
    PYTHONIOENCODING=utf-8 python run_3seeds.py --seeds 368,1037,5780
    PYTHONIOENCODING=utf-8 python run_3seeds.py --epochs 50   # 快速测试
"""
import sys
import os
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_ablation import main as run_single_seed


DEFAULT_SEEDS = [368, 1037, 5780]


def main(seeds=None, epochs=None):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    print("=" * 80)
    print(f"启动 3 种子消融实验: {len(seeds)} 个种子 × 7 个模型 = {len(seeds) * 7} 次训练")
    print(f"种子列表: {seeds}")
    print("=" * 80)

    all_summaries = []
    success_count = 0

    for i, seed in enumerate(seeds):
        print(f"\n{'#' * 80}")
        print(f"### 第 {i + 1}/{len(seeds)} 次独立实验，随机种子 = {seed}")
        print(f"{'#' * 80}")

        try:
            results = run_single_seed(seed=seed, epochs=epochs)
            success_count += 1
            print(f"\n✅ 种子 {seed} 实验完成 ({i + 1}/{len(seeds)})")
        except Exception as e:
            print(f"\n❌ 种子 {seed} 实验出错: {e}")
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print(f"所有种子实验完毕! 成功: {success_count}/{len(seeds)}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3种子消融实验 — 7模型×3种子')
    parser.add_argument('--seeds', type=str, default=None,
                        help='逗号分隔的种子列表 (默认: 368,1037,5780)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (默认: 各模型预设值)')
    args = parser.parse_args()

    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]

    main(seeds=seeds, epochs=args.epochs)
