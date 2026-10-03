"""
损失函数模块 — 从 exp_chronological.py 提取
CurriculumScientificLoss: 自适应相对损失平衡 + 精准局部正则化
"""
import torch
import torch.nn as nn
from .physics import calculate_R0_NGM


class CurriculumScientificLoss(nn.Module):
    """
    【学术终极重构】：物理信息 KAN (PIKAN) 的自适应相对损失平衡与精准局部正则化。
    彻底解决混合模型中"正则化溢出 (Regularization Spillover)"导致的表征坍塌问题，终结消融悖论。
    """

    def __init__(self, total_epochs=150, lambda_physics_base=1e-6, lambda_r0=1e-7,
                 lambda_tv=0.0, lambda_spline_smooth=0.0001, r0_prior_mean=1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=0.1)
        self.total_epochs = total_epochs

        # 极低物理约束 — 预测优先，物理仅作弱正则化
        self.lambda_physics_base = lambda_physics_base
        self.lambda_r0 = lambda_r0
        self.lambda_tv = lambda_tv
        self.lambda_spline_smooth = lambda_spline_smooth
        self.r0_prior_mean = r0_prior_mean

    def forward(self, pred_cases_nn, true_cases, pred_cases_ode, beta_seq, gamma,
                base_contact_seq, contact_seq, adj_matrix, epoch, model=None):
        loss_data = self.huber(pred_cases_nn, true_cases)

        if beta_seq is None or pred_cases_ode is None:
            return loss_data, {'total': loss_data.item(),
                               'unweighted_total': loss_data.item(),
                               'data': loss_data.item()}

        # 直接使用年龄组级别的 true_cases 与 pred_cases_ode 比较
        loss_physics = self.huber(pred_cases_ode, true_cases)

        beta_mean = beta_seq.mean(dim=-1) if beta_seq.ndim == 3 else beta_seq
        base_contact_mean = base_contact_seq.mean(dim=-1) if base_contact_seq.ndim == 3 else base_contact_seq
        r0_ngm = calculate_R0_NGM(beta_mean, gamma, base_contact_mean, adj_matrix)

        r0_prior_mean = self.r0_prior_mean
        loss_r0 = torch.nn.functional.huber_loss(
            r0_ngm, torch.full_like(r0_ngm, r0_prior_mean), delta=0.5)

        if beta_seq.ndim == 3 and beta_seq.shape[-1] > 1:
            tv_loss = torch.mean(torch.square(beta_seq[:, :, 1:] - beta_seq[:, :, :-1])) + \
                      torch.mean(torch.square(contact_seq[:, :, 1:] - contact_seq[:, :, :-1]))
        else:
            tv_loss = torch.tensor(0.0, device=loss_data.device)

        loss_spline_smooth = torch.tensor(0.0, device=loss_data.device)
        if model is not None:
            for name, param in model.named_parameters():
                if 'spline_weight' in name:
                    if any(head in name for head in
                           ['beta_head', 'gamma_head', 'contact_head', 'i0_head']):
                        loss_spline_smooth = loss_spline_smooth + torch.mean(param ** 2) * 1e-5
                        if param.shape[-1] > 1:
                            diff = param[..., 1:] - param[..., :-1]
                            loss_spline_smooth = loss_spline_smooth + torch.mean(diff ** 2) * 1e-4

        with torch.no_grad():
            scale_physics = (loss_data / (loss_physics + 1e-8)).clamp(max=1.0)
            scale_r0 = (loss_data / (loss_r0 + 1e-8)).clamp(max=1.0)

        warmup_epochs = 30.0
        curriculum_factor = min(1.0, float(epoch) / warmup_epochs)

        weight_physics = self.lambda_physics_base * curriculum_factor * scale_physics
        weight_r0 = self.lambda_r0 * curriculum_factor * scale_r0
        weight_tv = self.lambda_tv * curriculum_factor
        weight_spline = self.lambda_spline_smooth * curriculum_factor

        total_loss = (loss_data + weight_physics * loss_physics + weight_r0 * loss_r0 +
                      weight_tv * tv_loss + weight_spline * loss_spline_smooth)

        loss_dict = {
            'total': total_loss,
            'unweighted_total': loss_data.item() + loss_physics.item() + loss_r0.item(),
            'data': loss_data.item(),
            'ode': loss_physics.item(),
            'r0_loss': loss_r0.item(),
            'tv_loss': tv_loss.item(),
            'w_prior': weight_physics.item()
        }
        return total_loss, loss_dict
