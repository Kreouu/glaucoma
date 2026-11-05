"""L2 SAM perturbation; caller computes the second gradient before restoration."""
from contextlib import contextmanager
import torch


@contextmanager
def perturb_weights(parameters, rho):
    parameters = [p for p in parameters if p.grad is not None]
    norm = torch.nn.utils.get_total_norm([p.grad for p in parameters], error_if_nonfinite=True)
    scale = rho / max(float(norm), 1e-12)
    original = [(p, p.detach().clone()) for p in parameters]
    try:
        with torch.no_grad():
            for p in parameters:
                p.add_(p.grad, alpha=scale)
        yield
    finally:
        with torch.no_grad():
            for p, value in original:
                p.copy_(value)
