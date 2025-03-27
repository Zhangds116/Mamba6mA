import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from .ps_cnn import PSConvNet1, PSConvNet2, PSConvNet3
class MambaBlockBase(nn.Module):
    def __init__(self, args, conv_class, conv_attr_name):
        super().__init__()
        self.args = args
        self.conv_attr_name = conv_attr_name
        setattr(self, conv_attr_name, conv_class())
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x = rearrange(x, 'b l d_in -> b d_in l')
        conv_layer = getattr(self, self.conv_attr_name)
        x = conv_layer(x)
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y

class MambaBlock1(MambaBlockBase):
    def __init__(self, args):
        super().__init__(args, PSConvNet1, "conv1d1")

class MambaBlock2(MambaBlockBase):
    def __init__(self, args):
        super().__init__(args, PSConvNet2, "conv1d2")

class MambaBlock3(MambaBlockBase):
    def __init__(self, args):
        super().__init__(args, PSConvNet3, "conv1d3")