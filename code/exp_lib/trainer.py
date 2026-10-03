"""
训练管理器模块 — 从 exp_chronological.py 提取
包含 ImprovedAblationStudyManager: 模型初始化、训练、评估、全时段预测
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import random
import hashlib
import os
import time
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .models import MODEL_DEFINITIONS
from .data import DataCollector
from .loss import CurriculumScientificLoss
from .physics import age_structured_seir_step, calculate_R0_NGM
from . import config as _config


class ImprovedAblationStudyManager:
    """改进版消融实验管理器"""

    def __init__(self, config):
        self.config = config
        self.results = {}
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_collectors = {}
        self.model_definitions = self._get_model_definitions()
        self.dates = None
        self.case_scaler = None
        self.case_scaler_min = None
        self.case_scaler_range = None

    def _get_model_definitions(self):
        input_dim = self.config.get('input_dim', 16)
        num_nodes = self.config.get('num_nodes', 24)
        lookback = self.config.get('lookback', 7)
        common_dim = self.config.get('hidden_dim', 32)
        predict = self.config.get('predict', 3)

        return {
            'MLP_Baseline': {
                'class': MODEL_DEFINITIONS['MLP_Baseline']['class'],
                'args': {'input_dim': input_dim, 'hidden_dim': common_dim, 'output_dim': 3,
                         'lookback': lookback, 'predict_window': predict},
                'requires_adj': False, 'requires_dist': False, 'has_physics': False,
                'requires_full_seq': False,
                'learning_rate': 0.001, 'epochs': 150, 'grad_clip': 1.0, 'optimizer': 'AdamW',
                'loss_type': 'standard', 'weight_decay': 1e-3, 'patience': 30,
            },
            'LSTM_Baseline': {
                'class': MODEL_DEFINITIONS['LSTM_Baseline']['class'],
                'args': {'input_dim': input_dim, 'hidden_dim': common_dim, 'num_layers': 1,
                         'output_dim': 3, 'lookback': lookback, 'predict_window': predict},
                'requires_adj': False, 'requires_dist': False, 'has_physics': False,
                'requires_full_seq': True,
                'learning_rate': 0.001, 'epochs': 150, 'grad_clip': 1.0, 'optimizer': 'AdamW',
                'loss_type': 'standard', 'weight_decay': 1e-3, 'patience': 40,
            },
            'GAT_Baseline': {
                'class': MODEL_DEFINITIONS['GAT_Baseline']['class'],
                'args': {'in_features': lookback * input_dim, 'hidden_dim': common_dim, 'heads': 2,
                         'output_dim': 3, 'num_layers': 1, 'lookback': lookback, 'predict_window': predict},
                'requires_adj': True, 'requires_dist': False, 'has_physics': False,
                'requires_full_seq': False,
                'learning_rate': 0.0005, 'epochs': 150, 'grad_clip': 1.0, 'optimizer': 'AdamW',
                'loss_type': 'standard', 'weight_decay': 1e-3, 'patience': 30,
            },
            'M_Graphormer_Baseline': {
                'class': MODEL_DEFINITIONS['M_Graphormer_Baseline']['class'],
                'args': {'num_nodes': num_nodes, 'input_dim': input_dim, 'hidden_dim': common_dim,
                         'num_heads': 2, 'num_layers': 1, 'lookback': lookback, 'use_tcn': True,
                         'enable_physics': True, 'use_kan': False, 'predict_window': predict},
                'requires_adj': True, 'requires_dist': True, 'has_physics': True,
                'requires_full_seq': True,
                'learning_rate': 0.0005, 'epochs': 150, 'grad_clip': 1.0, 'optimizer': 'AdamW',
                'loss_type': 'dynamic', 'weight_decay': 1e-3, 'patience': 40,
            },
            'KAN_Only': {
                'class': MODEL_DEFINITIONS['KAN_Only']['class'],
                'args': {'input_dim': input_dim, 'hidden_dim': common_dim, 'use_tcn': False,
                         'predict_window': predict},
                'requires_adj': False, 'requires_dist': False, 'has_physics': True,
                'requires_full_seq': True,
                'learning_rate': 0.0005, 'epochs': 150, 'grad_clip': 1.0, 'optimizer': 'AdamW',
                'loss_type': 'dynamic', 'weight_decay': 1e-3, 'patience': 40,
            },
            'NoPhysics_KAN_Graphormer': {
                'class': MODEL_DEFINITIONS['NoPhysics_KAN_Graphormer']['class'],
                'args': {'num_nodes': num_nodes, 'input_dim': input_dim, 'hidden_dim': common_dim,
                         'num_heads': 2, 'num_layers': 1, 'lookback': lookback, 'use_tcn': True,
                         'enable_physics': False, 'use_kan': True, 'predict_window': predict},
                'requires_adj': True, 'requires_dist': True, 'has_physics': False,
                'requires_full_seq': True,
                'learning_rate': 0.0005, 'epochs': 150, 'grad_clip': 1.0, 'optimizer': 'AdamW',
                'loss_type': 'standard', 'patience': 50,
            },
            'Full_KAN_Graphormer': {
                'class': MODEL_DEFINITIONS['Full_KAN_Graphormer']['class'],
                'args': {'num_nodes': num_nodes, 'input_dim': input_dim, 'hidden_dim': 64,
                         'num_heads': 2, 'num_layers': 1, 'lookback': lookback, 'use_tcn': True,
                         'enable_physics': True, 'use_kan': True, 'use_kan_ffn': True,
                         'predict_window': predict},
                'requires_adj': True, 'requires_dist': True, 'has_physics': True,
                'requires_full_seq': True,
                'learning_rate': 0.0005, 'epochs': 150, 'grad_clip': 1.0, 'optimizer': 'AdamW',
                'loss_type': 'dynamic', 'weight_decay': 1e-3, 'patience': 20,
            },
        }

    def _reseed_for(self, model_name):
        """用 (base_seed, model_name) 派生的确定性种子重置 RNG。

        目的：让每个模型的参数初始化与训练过程独立于其他模型的存在/顺序，
        从而真正实现"6 个模型共享同一组固定种子且可复现"。
        """
        base = int(getattr(_config, 'SEED', 0))
        h = int(hashlib.md5(model_name.encode('utf-8')).hexdigest()[:8], 16)
        s = (base + h) % (2 ** 31)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    def setup_models(self):
        print("\n>>> 初始化改进版消融实验模型...")
        for model_name, model_def in self.model_definitions.items():
            try:
                self._reseed_for(model_name)
                model_class = model_def['class']
                model_args = model_def['args'].copy()
                if 'input_dim' in model_args:
                    model_args['input_dim'] = self.config.get('input_dim', 16)
                if model_name == 'GAT_Baseline' and 'in_features' in model_args:
                    lookback = self.config.get('lookback', 7)
                    input_dim = self.config.get('input_dim', 16)
                    model_args['in_features'] = lookback * input_dim
                model = model_class(**model_args).to(self.device)
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.models[model_name] = {
                    'model': model,
                    'requires_adj': model_def['requires_adj'],
                    'requires_dist': model_def['requires_dist'],
                    'has_physics': model_def['has_physics'],
                    'requires_full_seq': model_def['requires_full_seq'],
                    'learning_rate': model_def.get('learning_rate', 0.0005),
                    'epochs': model_def.get('epochs', 50),
                    'grad_clip': model_def.get('grad_clip', 1.0),
                    'loss_type': model_def.get('loss_type', 'standard'),
                    'loss_params': model_def.get('loss_params', {}),
                    'patience': model_def.get('patience', 15),
                    'optimizer': model_def.get('optimizer', 'Adam'),
                    'weight_decay': model_def.get('weight_decay', 1e-4),
                    'trainable_params': trainable_params
                }
                self.data_collectors[model_name] = DataCollector()
                print(f"    {model_name}: {trainable_params:,} 可训练参数, "
                      f"LR={self.models[model_name]['learning_rate']:.6f}")
            except Exception as e:
                print(f"    ❌ 模型 {model_name} 初始化失败: {e}")
                import traceback
                traceback.print_exc()
        print(f"✅ 模型初始化完成")

    def compute_diffusion_distance(self, transition_matrix, t_steps=3):
        if not isinstance(transition_matrix, torch.Tensor):
            transition_matrix = torch.tensor(transition_matrix, dtype=torch.float32)
        was_2d = False
        if transition_matrix.dim() == 2:
            transition_matrix = transition_matrix.unsqueeze(0)
            was_2d = True
        elif transition_matrix.dim() != 3:
            raise ValueError(f"transition_matrix must be 2D or 3D, got {transition_matrix.dim()}D")
        row_sum = transition_matrix.sum(dim=-1, keepdim=True)
        row_sum_zero = (row_sum.squeeze(-1) == 0)
        P = transition_matrix / (row_sum + 1e-8)
        if row_sum_zero.any():
            P = P.clone()
            if P.dim() == 3:
                B_size, N_size, _ = P.shape
                eye_mat = torch.eye(N_size, device=P.device).unsqueeze(0).expand(B_size, -1, -1)
                P[row_sum_zero] = eye_mat[row_sum_zero]
        P_t = P.clone()
        for _ in range(t_steps - 1):
            P_t = torch.bmm(P_t, P)
        pi = P_t.mean(dim=1, keepdim=True) + 1e-8
        P_t_i = P_t.unsqueeze(2)
        P_t_j = P_t.unsqueeze(1)
        diff_sq = torch.pow(P_t_i - P_t_j, 2) / pi.unsqueeze(1)
        d_diff = torch.sqrt(torch.sum(diff_sq, dim=-1) + 1e-8)
        if was_2d:
            d_diff = d_diff.squeeze(0)
        return d_diff

    def train_model(self, model_name, train_loader, val_loader, base_contact_matrix, pops,
                    epochs=None):
        print(f"\n>>> 训练模型: {model_name}")
        huber_loss = nn.HuberLoss(delta=0.1)
        model_info = self.models[model_name]
        model = model_info['model']
        collector = self.data_collectors[model_name]
        collector.start_training_timer()

        learning_rate = model_info['learning_rate']
        total_epochs = epochs if epochs else model_info['epochs']
        patience = model_info['patience']
        grad_clip = model_info['grad_clip']

        if model_info.get('loss_type') in ['dynamic', 'dynamic_v2']:
            criterion = CurriculumScientificLoss(
                total_epochs=total_epochs,
                lambda_physics_base=self.config.get('lambda_physics_base', 1e-6),
                lambda_r0=self.config.get('lambda_r0', 1e-7),
                r0_prior_mean=self.config.get('r0_prior_mean', 1.0),
            ).to(self.device)
            use_dynamic_loss = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                          weight_decay=model_info.get('weight_decay', 1e-4))
        else:
            criterion = None
            use_dynamic_loss = False
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                          weight_decay=model_info.get('weight_decay', 1e-4))

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8,
                                      min_lr=1e-6)

        history = {'train_loss': [], 'val_loss': [], 'val_r2': [],
                   'learning_rates': [], 'epoch_times': []}
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None
        base_contact = base_contact_matrix.to(self.device).float()
        pop_tensor = pops.to(self.device).float()

        for epoch in range(total_epochs):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0
            train_batches = 0

            for batch_idx, (x, adj_seq, y_norm, y_raw) in enumerate(train_loader):
                x = x.to(self.device).float()
                adj_seq = adj_seq.to(self.device).float()
                y_norm = y_norm.to(self.device).float()
                y_raw = y_raw.to(self.device).float()

                if x.dim() == 3:
                    B, L, D_feat = x.shape
                    N_c = self.config['num_nodes']
                    x = x.view(B, L, N_c, D_feat // N_c)

                B, L, N_c, feat_dim = x.shape
                adj_cur = adj_seq[:, 0, :, :] if model_info['requires_adj'] else None
                dist_matrix = self.compute_diffusion_distance(adj_cur).to(
                    self.device).float() if (model_info['requires_dist'] and
                                             adj_cur is not None) else None

                optimizer.zero_grad()
                time_step_tensor = torch.tensor([batch_idx % 100], dtype=torch.long,
                                                device=self.device)

                if model_info['has_physics'] and use_dynamic_loss:
                    cases_pred_nn_norm, beta, gamma, base_contact_mod, contact_mod, \
                        i0_mult, _ = model(x, adj_seq, dist_matrix, time_step_tensor)
                    beta, gamma = beta.float(), gamma.float()
                    base_contact_mod = base_contact_mod.float()
                    contact_mod = contact_mod.float()

                    I0_norm_last = x[:, -1, :, 0:3].float()
                    I0_log_last = I0_norm_last * self.case_scaler_range + self.case_scaler_min
                    I0_raw = torch.expm1(I0_log_last) * i0_mult.unsqueeze(-1)
                    pop_batch = pop_tensor.unsqueeze(0).expand(B, -1, -1).float()
                    I0_raw = torch.min(I0_raw, pop_batch * 0.5)
                    E0_raw = I0_raw * 1.5
                    S0 = torch.clamp(pop_batch - I0_raw - E0_raw, min=0)
                    R0 = torch.zeros_like(I0_raw).float()

                    pred_list_norm = []
                    for t in range(y_norm.shape[1]):
                        mig_mat = adj_seq[:, t, :, :].float()
                        beta_t = beta[:, :, t].float() if beta.ndim == 3 else beta.float()
                        contact_t = contact_mod[:, :, t].float() if contact_mod.ndim == 3 else contact_mod.float()
                        S0, E0_raw, I0_raw, R0, incidence_raw = age_structured_seir_step(
                            S0, E0_raw, I0_raw, R0, beta_t, gamma, contact_t, mig_mat,
                            pop_batch, base_contact, mobility_rate=0.01)
                        inc_log_new = torch.log1p(torch.clamp(incidence_raw, min=0))
                        inc_norm_new = (inc_log_new - self.case_scaler_min) / (self.case_scaler_range + 1e-8)
                        inc_norm_new = torch.clamp(inc_norm_new, 0.0, 1.2)
                        pred_list_norm.append(inc_norm_new)

                    cases_ode_norm = torch.stack(pred_list_norm, dim=1)
                    loss, loss_dict = criterion(
                        pred_cases_nn=cases_pred_nn_norm, true_cases=y_norm,
                        pred_cases_ode=cases_ode_norm, beta_seq=beta, gamma=gamma,
                        base_contact_seq=base_contact_mod, contact_seq=contact_mod,
                        adj_matrix=adj_seq.mean(dim=1), epoch=epoch, model=model)
                    display_loss = loss_dict['unweighted_total']
                elif model_info['has_physics'] and not use_dynamic_loss:
                    # Physics heads exist but use standard loss (no ODE, no physics penalty)
                    if model_info['requires_full_seq']:
                        cases_pred_nn_norm, _, _, _, _, _, _ = model(
                            x, adj_seq, dist_matrix, time_step_tensor)
                    else:
                        cases_pred_nn_norm, _, _, _, _, _, _ = model(
                            x, adj_cur, dist_matrix, time_step_tensor)
                    loss = huber_loss(cases_pred_nn_norm, y_norm)
                    display_loss = loss.item()
                else:
                    if model_info['requires_full_seq']:
                        cases_pred_nn_norm, _, _, _, _, _, _ = model(
                            x, adj_seq, dist_matrix, time_step_tensor)
                    else:
                        cases_pred_nn_norm, _, _, _, _, _, _ = model(
                            x, adj_cur, dist_matrix, time_step_tensor)
                    loss = huber_loss(cases_pred_nn_norm, y_norm)
                    display_loss = loss.item()

                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                optimizer.step()
                train_loss += display_loss
                train_batches += 1

            avg_train_loss = train_loss / max(train_batches, 1)

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for x, adj_seq, y_norm, y_raw in val_loader:
                    x = x.to(self.device).float()
                    adj_seq = adj_seq.to(self.device).float()
                    y_norm = y_norm.to(self.device).float()
                    y_raw = y_raw.to(self.device).float()
                    B = x.shape[0]
                    if x.dim() == 3:
                        L, D_feat = x.shape[1], x.shape[2]
                        x = x.view(B, L, self.config['num_nodes'],
                                   D_feat // self.config['num_nodes'])
                    adj_cur = adj_seq[:, 0, :, :] if model_info['requires_adj'] else None
                    dist_matrix = self.compute_diffusion_distance(adj_cur).to(
                        self.device).float() if (model_info['requires_dist'] and
                                                 adj_cur is not None) else None
                    time_step_tensor = torch.tensor([0], dtype=torch.long, device=self.device)
                    if model_info['has_physics']:
                        cases_pred_nn_norm, beta, gamma, base_contact_mod, contact_mod, \
                            i0_mult, _ = model(x, adj_seq, dist_matrix, time_step_tensor)
                        pred_norm_val = cases_pred_nn_norm
                        pred_log = pred_norm_val * self.case_scaler_range + self.case_scaler_min
                        pred_raw = torch.expm1(pred_log)
                    else:
                        if model_info['requires_full_seq']:
                            pred_norm_val, *_ = model(x, adj_seq, dist_matrix, time_step_tensor)
                        else:
                            pred_norm_val, *_ = model(x, adj_cur, dist_matrix, time_step_tensor)
                        pred_log = pred_norm_val * self.case_scaler_range + self.case_scaler_min
                        pred_raw = torch.expm1(pred_log)
                    val_loss += huber_loss(pred_norm_val, y_norm).item()
                    val_batches += 1
                    if pred_raw.ndim == 4:
                        val_preds.append(pred_raw.sum(dim=-1).flatten().cpu())
                        val_targets.append(y_raw.sum(dim=-1).flatten().cpu())
                    else:
                        val_preds.append(pred_raw.flatten().cpu())
                        val_targets.append(y_raw.flatten().cpu())

            avg_val_loss = val_loss / max(val_batches, 1)
            if val_preds and val_targets:
                all_pred = torch.cat(val_preds, dim=0).numpy().flatten()
                all_target = torch.cat(val_targets, dim=0).numpy().flatten()
                val_r2 = r2_score(all_target, all_pred) if len(all_pred) > 1 else 0.0
            else:
                val_r2 = 0.0

            epoch_time = time.time() - epoch_start
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_r2'].append(val_r2)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            history['epoch_times'].append(epoch_time)
            collector.record_training_epoch(epoch, avg_train_loss, avg_val_loss,
                                            epoch_time, optimizer.param_groups[0]['lr'])
            print(f"Epoch {epoch + 1:3d}/{total_epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                  f"Val R²: {val_r2:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss - 1e-4:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                best_model_state = model.state_dict().copy()
                torch.save({
                    'epoch': epoch, 'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss, 'val_r2': val_r2
                }, os.path.join(_config.OUTPUT_DIR or '.', 'models', f"{model_name}_best.pth"))
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"早停触发，训练结束")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"加载最佳模型，验证损失: {best_val_loss:.6f}")
        collector.end_training_timer()
        collector.record_model_parameters(model)
        return history

    def evaluate_model(self, model_name, test_loader, base_contact_matrix, pops, cities):
        pop_tensor = pops.to(self.device)
        base_contact = base_contact_matrix.to(self.device)
        print(f"\n>>> 评估模型: {model_name}")
        model_info = self.models[model_name]
        model = model_info['model']
        model.eval()
        collector = self.data_collectors[model_name]
        collector.start_evaluation_timer()

        test_full_preds = []
        test_full_targets = []

        with torch.no_grad():
            for batch_idx, (x, adj_seq, y_norm, y_raw) in enumerate(test_loader):
                x = x.to(self.device).float()
                adj_seq = adj_seq.to(self.device).float()
                y_norm = y_norm.to(self.device).float()
                y_raw = y_raw.to(self.device).float()

                B = x.shape[0]
                if x.dim() == 3:
                    L, D_feat = x.shape[1], x.shape[2]
                    x = x.view(B, L, self.config['num_nodes'],
                               D_feat // self.config['num_nodes'])

                adj_cur = adj_seq[:, 0, :, :] if model_info['requires_adj'] else None
                dist_matrix = self.compute_diffusion_distance(adj_cur).to(
                    self.device).float() if (model_info['requires_dist'] and
                                             adj_cur is not None) else None
                time_step_tensor = torch.tensor([batch_idx % 100], dtype=torch.long,
                                                device=self.device)

                if model_info['has_physics']:
                    cases_pred_nn_norm, beta, gamma, base_contact_mod, contact_mod, \
                        i0_mult, attn_map = model(x, adj_seq, dist_matrix, time_step_tensor)
                    beta, gamma = beta.float(), gamma.float()
                    base_contact_mod = base_contact_mod.float()
                    contact_mod = contact_mod.float()
                    i0_mult = i0_mult.float()

                    I0_norm_last = x[:, -1, :, 0:3].float()
                    I0_log_last = I0_norm_last * self.case_scaler_range + self.case_scaler_min
                    I0_raw = torch.expm1(I0_log_last) * i0_mult.unsqueeze(-1)
                    pop_batch = pop_tensor.unsqueeze(0).expand(B, -1, -1).float()
                    I0_raw = torch.min(I0_raw, pop_batch * 0.5)
                    E0_raw = I0_raw * 1.5
                    S0 = torch.clamp(pop_batch - I0_raw - E0_raw, min=0)
                    R0 = torch.zeros_like(I0_raw).float()

                    pred_list_norm = []
                    for t in range(y_raw.shape[1]):
                        mig_mat = adj_seq[:, t, :, :].float()
                        beta_t = beta[:, :, t].float() if beta.ndim == 3 else beta.float()
                        contact_t = contact_mod[:, :, t].float() if contact_mod.ndim == 3 else contact_mod.float()
                        S0, E0_raw, I0_raw, R0, incidence_raw = age_structured_seir_step(
                            S0, E0_raw, I0_raw, R0, beta_t, gamma, contact_t, mig_mat,
                            pop_batch, base_contact, mobility_rate=0.01)
                        inc_log_new = torch.log1p(torch.clamp(incidence_raw, min=0))
                        inc_norm_new = (inc_log_new - self.case_scaler_min) / (self.case_scaler_range + 1e-8)
                        inc_norm_new = torch.clamp(inc_norm_new, 0.0, 1.2)
                        pred_list_norm.append(inc_norm_new)

                    pred_norm_val = cases_pred_nn_norm
                    pred_log = pred_norm_val * self.case_scaler_range + self.case_scaler_min
                    cases_pred_raw = torch.expm1(pred_log)

                    beta_mean = beta.mean(dim=-1) if beta.ndim == 3 else beta
                    contact_mean = contact_mod.mean(dim=-1) if contact_mod.ndim == 3 else contact_mod

                    # Local Rt
                    mig_mat_eval = adj_seq.mean(dim=1).float() if adj_seq.ndim == 4 else adj_seq.float()
                    mu = 1.0 / (70 * 365)
                    lambda_dis = 0.00008
                    sigma_incubation = 1.0 / 3.0
                    effective_mig = 0.01 * mig_mat_eval
                    mask = torch.eye(self.config['num_nodes'], device=self.device).unsqueeze(0).bool()
                    effective_mig = effective_mig.masked_fill(mask, 0.0)
                    outflow_rates = torch.sum(effective_mig, dim=-1)
                    prob_survive_E = sigma_incubation / (sigma_incubation + mu + outflow_rates)
                    duration_I = 1.0 / (gamma + mu + lambda_dis + outflow_rates)
                    rt_local = (beta_mean * contact_mean) * prob_survive_E * duration_I
                    r0 = rt_local
                    contact_mean = contact_mod.mean(dim=-1) if contact_mod.ndim == 3 else contact_mod
                else:
                    if model_info['requires_full_seq']:
                        cases_pred_nn_norm, _, _, _, _, _, attn_map = model(
                            x, adj_seq, dist_matrix, time_step_tensor)
                    else:
                        cases_pred_nn_norm, _, _, _, _, _, attn_map = model(
                            x, adj_cur, dist_matrix, time_step_tensor)
                    pred_log = cases_pred_nn_norm * self.case_scaler_range + self.case_scaler_min
                    cases_pred_raw = torch.expm1(pred_log)
                    beta_mean = gamma = contact_mean = r0 = None

                collector.record_evaluation_batch(
                    cases_pred_raw, y_raw, beta_mean, gamma, contact_mean, r0, attn_map)

                num_nodes = self.config['num_nodes']
                if cases_pred_raw.ndim == 4:
                    test_full_preds.append(cases_pred_raw.sum(dim=-1).reshape(-1, num_nodes).cpu())
                    test_full_targets.append(y_raw.sum(dim=-1).reshape(-1, num_nodes).cpu())
                else:
                    test_full_preds.append(cases_pred_raw.reshape(-1, num_nodes).cpu())
                    test_full_targets.append(y_raw.reshape(-1, num_nodes).cpu())

        collector.end_evaluation_timer()
        eval_data = collector.get_consolidated_data()

        if len(test_full_preds) > 0:
            pred_np_raw = torch.cat(test_full_preds, dim=0).numpy()
            target_np_raw = torch.cat(test_full_targets, dim=0).numpy()
            print(f"    [3天完整窗口] 预测值范围: [{pred_np_raw.min():.2f}, {pred_np_raw.max():.2f}], "
                  f"真实值范围: [{target_np_raw.min():.2f}, {target_np_raw.max():.2f}]")

            if np.var(target_np_raw.flatten()) < 1e-5:
                r2 = 0.0
            else:
                r2 = float(r2_score(target_np_raw.flatten(), pred_np_raw.flatten()))
            rmse = float(np.sqrt(mean_squared_error(target_np_raw.flatten(), pred_np_raw.flatten())))
            mae = float(mean_absolute_error(target_np_raw.flatten(), pred_np_raw.flatten()))
            mse = float(mean_squared_error(target_np_raw.flatten(), pred_np_raw.flatten()))

            collector.record_performance_metrics(r2, rmse, mae, mse)
            print(f"    性能指标: 全局R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            eval_data['performance_metrics'] = collector.performance_metrics.copy()

            # 城市等级分层
            try:
                tier1_idx = [i for i, city in enumerate(cities) if city in _config.TIER_1_CITIES]
                tier2_idx = [i for i, city in enumerate(cities) if city in _config.TIER_2_CITIES]
                if tier1_idx and tier2_idx:
                    t1_target = target_np_raw[:, tier1_idx].flatten()
                    t1_pred = pred_np_raw[:, tier1_idx].flatten()
                    t2_target = target_np_raw[:, tier2_idx].flatten()
                    t2_pred = pred_np_raw[:, tier2_idx].flatten()
                    r2_t1 = float(r2_score(t1_target, t1_pred)) if np.var(t1_target) > 1e-5 else 0.0
                    r2_t2 = float(r2_score(t2_target, t2_pred)) if np.var(t2_target) > 1e-5 else 0.0
                    print(f"    [空间泛化分析] 一线核心城市 R²={r2_t1:.4f} | 其他节点城市 R²={r2_t2:.4f}")
                    eval_data['performance_metrics']['r2_tier1'] = r2_t1
                    eval_data['performance_metrics']['r2_tier2'] = r2_t2
            except Exception:
                pass

            # ==========================================
            # 误差分析 (Error Analysis)
            # ==========================================
            residuals_flat = (pred_np_raw.flatten() - target_np_raw.flatten())
            n_res = len(residuals_flat)

            # 残差统计
            residual_mean = float(np.mean(residuals_flat))
            residual_std = float(np.std(residuals_flat))
            residual_skew = float(
                np.mean((residuals_flat - residual_mean) ** 3) / (residual_std ** 3 + 1e-8)
            ) if residual_std > 1e-8 else 0.0

            # Durbin-Watson 统计量 (检验残差自相关)
            if n_res > 2:
                dw_numer = np.sum((residuals_flat[1:] - residuals_flat[:-1]) ** 2)
                dw_denom = np.sum(residuals_flat ** 2)
                dw_stat = float(dw_numer / (dw_denom + 1e-8))
            else:
                dw_stat = 2.0

            # 每城市误差指标
            per_city_errors = {}
            for i, city in enumerate(cities):
                city_res = pred_np_raw[:, i].flatten() - target_np_raw[:, i].flatten()
                city_target = target_np_raw[:, i].flatten()
                city_r2 = float(r2_score(city_target, pred_np_raw[:, i].flatten())) \
                    if np.var(city_target) > 1e-5 else 0.0
                city_rmse = float(np.sqrt(np.mean(city_res ** 2)))
                city_mae = float(np.mean(np.abs(city_res)))
                per_city_errors[city] = {
                    'r2': city_r2, 'rmse': city_rmse, 'mae': city_mae,
                    'residual_mean': float(np.mean(city_res)),
                    'residual_std': float(np.std(city_res))
                }

            # 保存误差分析结果
            error_analysis = {
                'residual_mean': residual_mean,
                'residual_std': residual_std,
                'residual_skewness': residual_skew,
                'durbin_watson': dw_stat,
                'per_city_errors': per_city_errors,
            }
            eval_data['error_analysis'] = error_analysis

            # 打印误差分析摘要
            print(f"\n    [误差分析]")
            print(f"    残差均值: {residual_mean:.4f} | 残差标准差: {residual_std:.4f} | "
                  f"偏度: {residual_skew:.3f} | Durbin-Watson: {dw_stat:.3f}")
            # 打印最差3个城市
            city_rmse_sorted = sorted(per_city_errors.items(),
                                      key=lambda x: x[1]['rmse'], reverse=True)
            top3_worst = ', '.join(
                f"{c}({e['rmse']:.2f})" for c, e in city_rmse_sorted[:3])
            top3_best = ', '.join(
                f"{c}({e['rmse']:.2f})" for c, e in city_rmse_sorted[-3:])
            print(f"    RMSE最高3城: {top3_worst}")
            print(f"    RMSE最低3城: {top3_best}")

        # 参数有效性
        param_validity = {}
        if model_info['has_physics'] and eval_data['evaluation_results']['betas'] is not None:
            betas_np = eval_data['evaluation_results']['betas'].numpy().flatten()
            gammas_np = eval_data['evaluation_results']['gammas'].numpy().flatten()
            contacts_np = eval_data['evaluation_results']['contacts'].numpy().flatten()
            r0s_np = eval_data['evaluation_results']['r0s'].numpy().flatten()
            valid_r0_ratio = float(np.mean((r0s_np >= 1.0) & (r0s_np <= 2.0)))
            param_validity = {
                'beta_mean': float(np.mean(betas_np)), 'beta_std': float(np.std(betas_np)),
                'gamma_mean': float(np.mean(gammas_np)), 'gamma_std': float(np.std(gammas_np)),
                'contact_mean': float(np.mean(contacts_np)), 'contact_std': float(np.std(contacts_np)),
                'r0_mean': float(np.mean(r0s_np)), 'r0_std': float(np.std(r0s_np)),
                'r0_valid_ratio': valid_r0_ratio
            }
            eval_data['param_validity'] = param_validity
            print(f"\n    [流行病学参数动态估计结果]")
            print(f"    传播率(β): {param_validity['beta_mean']:.4f}±{param_validity['beta_std']:.4f}")
            print(f"    恢复率(γ): {param_validity['gamma_mean']:.4f}±{param_validity['gamma_std']:.4f}")
            print(f"    接触系数(C): {param_validity['contact_mean']:.4f}±{param_validity['contact_std']:.4f}")
            print(f"    有效再生数(Rt): {param_validity['r0_mean']:.3f}±{param_validity['r0_std']:.3f}")
        else:
            eval_data['param_validity'] = param_validity

        return eval_data

    def predict_full_timeseries(self, model, full_dataset, case_scaler, device,
                                lookback=7, predict=3):
        model.eval()
        features_raw = full_dataset.features_raw
        features_norm = full_dataset.features
        adjs = full_dataset.adjs

        if not torch.is_tensor(features_raw):
            features_raw = torch.from_numpy(features_raw).float().to(device)
        else:
            features_raw = features_raw.float().to(device)
        if not torch.is_tensor(features_norm):
            features_norm = torch.from_numpy(features_norm).float().to(device)
        else:
            features_norm = features_norm.float().to(device)
        if not torch.is_tensor(adjs):
            adjs = torch.from_numpy(adjs).float().to(device)
        else:
            adjs = adjs.float().to(device)

        n_days, n_cities, _ = features_raw.shape
        case_min = case_scaler.data_min_[:, :, :3].to(device)
        case_range = case_scaler.data_range_[:, :, :3].to(device)

        pred_sum = torch.zeros((n_days, n_cities, 3), device=device)
        pred_count = torch.zeros((n_days, n_cities, 3), device=device)
        dist_matrix = torch.eye(n_cities, device=device).unsqueeze(0)

        with torch.no_grad():
            for t in range(0, n_days - lookback - predict + 1):
                x_window = features_norm[t:t + lookback]
                adj_window = adjs[t:t + lookback]
                x_tensor = x_window.unsqueeze(0)
                adj_tensor = adj_window.unsqueeze(0)
                outputs = model(x_tensor, adj_tensor, dist_matrix)
                pred_norm = outputs[0]
                pred_log = pred_norm * case_range + case_min
                pred_raw = torch.expm1(pred_log)
                for p_idx in range(predict):
                    day_idx = t + lookback + p_idx
                    pred_sum[day_idx] += pred_raw[0, p_idx]
                    pred_count[day_idx] += 1

        pred_count = torch.clamp(pred_count, min=1)
        full_pred = pred_sum / pred_count
        full_pred_total = full_pred.sum(dim=-1).cpu().numpy()
        true_total = features_raw[:, :, 0:3].sum(dim=-1).cpu().numpy()
        return true_total, full_pred_total

    def predict_full_timeseries_with_params(self, model, full_dataset, case_scaler, device,
                                             base_contact_matrix, pops, lookback=7, predict=3):
        model.eval()
        features_raw = full_dataset.features_raw
        features_norm = full_dataset.features
        adjs = full_dataset.adjs

        if not torch.is_tensor(features_raw):
            features_raw = torch.from_numpy(features_raw).float().to(device)
        else:
            features_raw = features_raw.float().to(device)
        if not torch.is_tensor(features_norm):
            features_norm = torch.from_numpy(features_norm).float().to(device)
        else:
            features_norm = features_norm.float().to(device)
        if not torch.is_tensor(adjs):
            adjs = torch.from_numpy(adjs).float().to(device)
        else:
            adjs = adjs.float().to(device)

        n_days, n_cities, _ = features_raw.shape
        case_min = case_scaler.data_min_[:, :, :3].to(device)
        case_range = case_scaler.data_range_[:, :, :3].to(device)

        pred_sum = torch.zeros((n_days, n_cities, 3), device=device)
        pred_count = torch.zeros((n_days, n_cities, 3), device=device)
        beta_full = torch.zeros((n_days, n_cities), device=device)
        beta_count = torch.zeros((n_days, n_cities), device=device)
        contact_full = torch.zeros((n_days, n_cities), device=device)
        contact_count = torch.zeros((n_days, n_cities), device=device)
        rt_full = torch.zeros((n_days, n_cities), device=device)
        rt_count = torch.zeros((n_days, n_cities), device=device)

        dist_matrix = torch.eye(n_cities, device=device).unsqueeze(0)
        base_contact = base_contact_matrix.to(device).float()
        pop_tensor = pops.to(device).float()

        with torch.no_grad():
            for t in range(0, n_days - lookback - predict + 1):
                x_window = features_norm[t:t + lookback]
                adj_window = adjs[t:t + lookback]
                x_tensor = x_window.unsqueeze(0)
                adj_tensor = adj_window.unsqueeze(0)
                outputs = model(x_tensor, adj_tensor, dist_matrix)
                pred_norm = outputs[0]
                beta_out = outputs[1]
                gamma_out = outputs[2]
                contact_out = outputs[4]

                # 统一 beta
                if beta_out.ndim == 3:
                    if beta_out.shape[1] == predict and beta_out.shape[2] == n_cities:
                        beta_t = beta_out[0, 0, :]
                    elif beta_out.shape[1] == n_cities and beta_out.shape[2] == predict:
                        beta_t = beta_out[0, :, 0]
                    else:
                        beta_t = beta_out[0, 0, :]
                elif beta_out.ndim == 2:
                    beta_t = beta_out[0, :]
                else:
                    beta_t = beta_out.squeeze()[:n_cities]

                # 统一 contact
                if contact_out.ndim == 3:
                    if contact_out.shape[1] == predict and contact_out.shape[2] == n_cities:
                        contact_t = contact_out[0, 0, :]
                    elif contact_out.shape[1] == n_cities and contact_out.shape[2] == predict:
                        contact_t = contact_out[0, :, 0]
                    else:
                        contact_t = contact_out[0, 0, :]
                elif contact_out.ndim == 2:
                    contact_t = contact_out[0, :]
                else:
                    contact_t = contact_out.squeeze()[:n_cities]

                # 统一 gamma
                if gamma_out.ndim == 2:
                    gamma_t = gamma_out[0, :]
                elif gamma_out.ndim == 1:
                    gamma_t = gamma_out
                else:
                    gamma_t = gamma_out.squeeze()
                if gamma_t.numel() == 1:
                    gamma_t = gamma_t.expand(n_cities)

                # Rt
                mig_mat_eval = adj_window.mean(dim=0)
                mu = 1.0 / (70 * 365)
                lambda_dis = 0.00008
                sigma_incubation = 1.0 / 3.0
                effective_mig = 0.01 * mig_mat_eval
                mask = torch.eye(n_cities, device=device).bool()
                effective_mig = effective_mig.masked_fill(mask, 0.0)
                outflow_rates = torch.sum(effective_mig, dim=-1)
                prob_survive_E = sigma_incubation / (sigma_incubation + mu + outflow_rates)
                duration_I = 1.0 / (gamma_t + mu + lambda_dis + outflow_rates)
                rt_t = (beta_t * contact_t) * prob_survive_E * duration_I

                pred_log = pred_norm * case_range + case_min
                pred_raw = torch.expm1(pred_log)

                for p_idx in range(predict):
                    day_idx = t + lookback + p_idx
                    pred_sum[day_idx] += pred_raw[0, p_idx]
                    pred_count[day_idx] += 1

                day_idx = t + lookback
                beta_full[day_idx] += beta_t
                beta_count[day_idx] += 1
                contact_full[day_idx] += contact_t
                contact_count[day_idx] += 1
                rt_full[day_idx] += rt_t
                rt_count[day_idx] += 1

        pred_count = torch.clamp(pred_count, min=1)
        full_pred = pred_sum / pred_count
        full_pred_total = full_pred.sum(dim=-1).cpu().numpy()

        beta_count = torch.clamp(beta_count, min=1)
        beta_full_np = (beta_full / beta_count).cpu().numpy()
        contact_count = torch.clamp(contact_count, min=1)
        contact_full_np = (contact_full / contact_count).cpu().numpy()
        rt_count = torch.clamp(rt_count, min=1)
        rt_full_np = (rt_full / rt_count).cpu().numpy()

        true_total = features_raw[:, :, 0:3].sum(dim=-1).cpu().numpy()
        return true_total, full_pred_total, beta_full_np, rt_full_np, contact_full_np

    def run_all_experiments(self, train_loader, val_loader, test_loader, base_contact_matrix,
                            pops, cities, case_scaler, dates, epochs=None):
        self.dates = dates
        self.case_scaler = case_scaler
        self.case_scaler_min = case_scaler.data_min_[:, :, :3].to(self.device)
        self.case_scaler_range = case_scaler.data_range_[:, :, :3].to(self.device)

        print("\n" + "=" * 80)
        print("开始改进版消融实验")
        print("=" * 80)

        self.setup_models()

        for model_name in list(self.models.keys()):
            print(f"\n{'=' * 60}")
            print(f"实验: {model_name}")
            print(f"{'=' * 60}")
            self._reseed_for(model_name)
            train_history = self.train_model(model_name, train_loader, val_loader,
                                             base_contact_matrix, pops, epochs)
            eval_data = self.evaluate_model(model_name, test_loader, base_contact_matrix,
                                            pops, cities)
            self.results[model_name] = {
                'train_history': train_history,
                'eval_data': eval_data,
                'model_info': self.models[model_name],
                'collector_data': self.data_collectors[model_name].get_consolidated_data()
            }
            result_path = os.path.join(_config.OUTPUT_DIR or '.', 'results', f"{model_name}_result.pkl")
            with open(result_path, 'wb') as f:
                pickle.dump(self.results[model_name], f)

        # Full_KAN_Graphormer 全时段预测
        if 'Full_KAN_Graphormer' in self.results:
            print("\n>>> 正在为 Full_KAN_Graphormer 生成全时段预测及参数...")
            model_info = self.models['Full_KAN_Graphormer']
            model = model_info['model']
            train_ds = train_loader.dataset

            class FullDataset:
                def __init__(self, features_raw, features_norm, adjs):
                    self.features_raw = features_raw
                    self.features = features_norm
                    self.adjs = adjs

            full_ds = FullDataset(train_ds.features_raw, train_ds.features, train_ds.adjs)
            true_total, pred_total, beta_full, rt_full, contact_full = \
                self.predict_full_timeseries_with_params(
                    model, full_ds, self.case_scaler, self.device,
                    base_contact_matrix, pops,
                    lookback=self.config['lookback'],
                    predict=self.config.get('predict', 3))
            self.results['Full_KAN_Graphormer']['full_pred'] = {
                'true_total': true_total, 'pred_total': pred_total,
                'beta_full': beta_full, 'rt_full': rt_full,
                'contact_full': contact_full, 'dates': self.dates
            }
            # 保存 full_pred 到独立 pkl (同时更新模型 pkl)
            full_pred_path = os.path.join(_config.OUTPUT_DIR or '.', 'results',
                                          'Full_KAN_Graphormer_result.pkl')
            with open(full_pred_path, 'wb') as f:
                pickle.dump(self.results['Full_KAN_Graphormer'], f)
            fp_pkl = os.path.join(_config.OUTPUT_DIR or '.', 'results', 'full_pred.pkl')
            with open(fp_pkl, 'wb') as f:
                pickle.dump(self.results['Full_KAN_Graphormer']['full_pred'], f)
            print("全时段预测及参数已保存。")

        print(f"\n{'=' * 80}")
        print("所有消融实验完成")
        print(f"{'=' * 80}")
        return self.results
