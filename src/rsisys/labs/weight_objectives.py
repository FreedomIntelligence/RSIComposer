"""Small tensor objective; torch is optional until the objective is executed."""


def grpo_objective(logps, old_logps, reference_logps, advantage, response_mask, beta=0.001):
    import torch
    ratio = (logps - old_logps.detach()).exp()
    clipped = ratio.clamp(0.8, 1.2)
    policy = -torch.minimum(ratio * advantage, clipped * advantage)
    delta = reference_logps.detach() - logps
    kl = delta.exp() - delta - 1
    per_sequence = ((policy + beta * kl) * response_mask).sum(-1) / response_mask.sum(-1)
    return per_sequence.mean(), ratio.detach(), kl.detach()
