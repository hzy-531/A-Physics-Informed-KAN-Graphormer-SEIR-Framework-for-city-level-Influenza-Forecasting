import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 从主实验脚本导入模型类
import sys
sys.path.insert(0, r"E:\Claude code\KAN+\code\code\experiments")
from exp_chronological import EnhancedFull_Graphormer_V3, RobustKANLinear


def visualize_kan_activation_spline(model, target_module_path, input_node_idx=0, output_node_idx=0,
                                    grid_range=(-3.0, 3.0), num_points=300, save_dir="visualizations/architecture"):
    """
    绘制指定KAN层的特定输入→输出B样条激活曲线。

    Args:
        model: 已加载的PyTorch模型
        target_module_path: 目标模块的路径，如 'case_head.0' 或 'beta_head.0'
        input_node_idx: 输入特征索引
        output_node_idx: 输出节点索引
        grid_range: 评估区间
        num_points: 采样点数
        save_dir: 保存目录
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    os.makedirs(save_dir, exist_ok=True)

    # 根据路径获取目标模块
    if isinstance(target_module_path, str):
        target_layer = model
        for part in target_module_path.split('.'):
            if part.isdigit():
                target_layer = target_layer[int(part)]
            else:
                target_layer = getattr(target_layer, part)
    else:
        target_layer = target_module_path

    # 确保是KAN层
    if not isinstance(target_layer, RobustKANLinear):
        raise ValueError(f"目标模块不是 RobustKANLinear，而是 {type(target_layer)}")

    # 提取权重
    spline_weight = target_layer.spline_weight.detach().cpu()  # [out, in, coeff]
    base_weight = target_layer.base_weight.detach().cpu()      # [out, in]

    out_dim, in_dim = spline_weight.shape[:2]
    if output_node_idx >= out_dim or input_node_idx >= in_dim:
        raise ValueError(f"索引超出范围: out_dim={out_dim}, in_dim={in_dim}")

    coeff = spline_weight[output_node_idx, input_node_idx]      # [coeff_dim]
    base_w = base_weight[output_node_idx, input_node_idx].item()

    # 生成评估点
    x_eval = torch.linspace(grid_range[0], grid_range[1], num_points)

    # 利用模块自身的 b_splines 方法计算基函数
    basis = target_layer.b_splines(x_eval)                      # [num_points, coeff_dim]
    spline_out = torch.matmul(basis, coeff).numpy()             # 样条部分

    # 基部分：SiLU 乘以基础权重
    base_out = base_w * F.silu(x_eval).numpy()                  # [num_points]

    total_out = spline_out + base_out                            # 总激活

    x_np = x_eval.numpy()

    # 绘图
    plt.figure(figsize=(9, 6), dpi=350)
    plt.plot(x_np, spline_out, color="#2C3E50", linewidth=2.5,
             label=f'Spline Part (Input {input_node_idx} → Out {output_node_idx})')
    plt.plot(x_np, base_out, color="#E74C3C", linestyle='--', linewidth=1.5, alpha=0.85, label='Base SiLU Part')
    plt.plot(x_np, total_out, color="#27AE60", linewidth=2.0, alpha=0.95, label='Total Activation')

    # 标记网格节点
    grid = target_layer.grid.detach().cpu().numpy()
    valid_grid = grid[(grid >= grid_range[0]) & (grid <= grid_range[1])]
    for knot in valid_grid:
        plt.axvline(x=knot, color='gray', linestyle=':', alpha=0.45)

    plt.title(f"KAN B-Spline Activation: {target_module_path}", fontsize=14, fontweight='bold')
    plt.xlabel("Input Feature Amplitude", fontsize=12)
    plt.ylabel("Activation Output", fontsize=12)
    plt.legend(loc='best', frameon=True, edgecolor='black')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"kan_spline_{target_module_path.replace('.', '_')}_in{input_node_idx}_out{output_node_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ B样条曲线已保存至: {save_path}")
    plt.close()


if __name__ == "__main__":
    # 创建与训练时完全一致的模型结构
    model = EnhancedFull_Graphormer_V3(
        num_nodes=24,
        input_dim=17,
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        lookback=7,
        predict_window=3,
        use_tcn=True,
        enable_physics=True,
        use_kan=True
    )

    # 自动查找最新训练权重
    import glob as _glob
    result_dirs = sorted(_glob.glob(r"E:\Claude code\KAN+\result\ablation_study_v9_*"))
    result_dirs = [d for d in result_dirs if os.path.isdir(d)]
    checkpoint_path = None
    for d in reversed(result_dirs):
        candidate = os.path.join(d, "models", "Full_KAN_Graphormer_best.pth")
        if os.path.exists(candidate):
            checkpoint_path = candidate
            break

    if checkpoint_path is None:
        print("❌ 找不到权重文件，请先运行训练。")
    else:
        print(f"📂 使用检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ 模型权重加载成功，架构尺寸完美匹配！")

        output_dir = r"E:\Claude code\KAN+\result\kan_interpretability"
        os.makedirs(output_dir, exist_ok=True)

        # 遍历 case_head 和 beta_head 的全部输入→输出通道
        for head_name in ['case_head', 'beta_head', 'gamma_head', 'contact_head']:
            layer = getattr(model, head_name, None)
            if layer is None or not isinstance(layer[0], RobustKANLinear):
                continue
            kan_layer = layer[0]
            out_dim, in_dim = kan_layer.spline_weight.shape[:2]
            print(f"\n{'='*60}")
            print(f"  {head_name}[0]: KAN({in_dim}→{out_dim})")
            print(f"{'='*60}")
            for in_idx in range(min(in_dim, 16)):   # 最多16个输入通道
                for out_idx in range(min(out_dim, 8)):  # 最多8个输出通道
                    visualize_kan_activation_spline(
                        model, target_module_path=f'{head_name}.0',
                        input_node_idx=in_idx, output_node_idx=out_idx,
                        save_dir=output_dir
                    )
        print(f"\n✅ 全部样条曲线已保存至: {output_dir}")