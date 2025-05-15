import torch
from torch.optim import Optimizer
import math


class MomentumFilteringAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, alpha=0.1, logger=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, alpha=alpha)
        super(MomentumFilteringAdam, self).__init__(params, defaults)
        self.logger = logger


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        # print(self.param_groups)
        # print()

        for group in self.param_groups:
            alpha = group['alpha']
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']

            # `params` is already partitioned (e.g., Q, K, V, FFN)
            params = group['params']

            for p in params:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # 动量项
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差校正
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # print_rank_0(f"m_hat shape: {m_hat.shape}, v_hat shape: {v_hat.shape}")

                # 计算过滤器
                abs_m_hat = torch.abs(m_hat.view(-1))
                top_k_threshold = torch.kthvalue(abs_m_hat, int((1 - alpha) * len(abs_m_hat))).values.item()
                filter_mask = (torch.abs(m_hat) >= top_k_threshold).float()

                # self.logger.info(f"filter_mask shape: {filter_mask.shape}")

                # 使用过滤器更新参数
                # denom = (v_hat.sqrt() + eps)
                # step_size = lr
                p.data.addcdiv_(m_hat * filter_mask, v_hat.sqrt() + eps, value=-lr)

        return loss
    

def get_partitioned_params(model):
    partitioned_params = []

    # 定义分组规则
    groups = [
        ("query_params", "to_q"),
        ("key_params", "to_k"),
        ("value_params", "to_v"),
        ("output_params", "to_out"),
        ("mlp_params", "ff"),
    ]

    # 遍历每个规则并分组
    for group_name, keyword in groups:
        params = [
            param for name, param in model.named_parameters()
            if keyword in name and param.requires_grad
        ]
        if params:
            partitioned_params.append({
                "name": group_name,  # 分组名称（便于调试或追踪）
                "params": params
            })

    # 其他参数（不属于上述任一组）
    regular_params = [
        param for name, param in model.named_parameters()
        if not any(keyword in name for _, keyword in groups) and param.requires_grad
    ]
    if regular_params:
        partitioned_params.append({
            "name": "regular_params",
            "params": regular_params
        })

    return partitioned_params
