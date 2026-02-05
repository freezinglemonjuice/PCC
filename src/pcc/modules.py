import torch


class RMSN(torch.nn.Module):


    def __init__(self, mean_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.mean_dim = mean_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)
