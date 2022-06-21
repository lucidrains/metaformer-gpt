import torch
from torch import nn, einsum
from einops import rearrange, repeat

from scipy.fftpack import next_fast_len

# helper functions

def cummean(x, *, dim):
    numer = x.cumsum(dim = dim)
    denom = torch.arange(x.shape[1], device = x.device) + 1
    return numer / rearrange(denom, '... -> ... 1')

def conv1d_fft(x, weights, dim = -2, weight_dim = -1):
    # O(N log(N)) 1d convolution using some fourier trick

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)

    f_x = torch.fft.rfft(x, n = fast_len, dim = dim)
    f_weight = torch.fft.rfft(weights, n = fast_len, dim = weight_dim)

    f_v_weight = f_x * rearrange(f_weight.conj(), '... -> ... 1')
    out = torch.fft.irfft(f_v_weight, fast_len, dim = dim)
    out = out.roll(-1, dims = (dim,))

    indices = torch.arange(start = fast_len - N, end = fast_len, dtype = torch.long, device = x.device)
    out = out.index_select(dim, indices)
    return out

# classes

class MeanCenteringPool(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.norm(x)
        x = cummean(x, dim = 1) - x
        return self.proj(x)

class MHESA(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.initial_state = nn.Parameter(torch.randn(heads, dim // heads))

        self.norm = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.randn(heads))

        self.project_in = nn.Linear(dim, dim)
        self.project_out = nn.Linear(dim, dim)

    def forward(self, x, naive = False):
        b, n, d, h, device = *x.shape, self.heads, x.device

        x = self.norm(x)

        # linear project in

        x = self.project_in(x)

        # split out heads

        x = rearrange(x, 'b n (h d) -> b h n d', h = h)

        # temporal difference

        x = torch.cat((
            repeat(self.initial_state, 'h d -> b h 1 d', b = b),
            x
        ), dim = -2)

        x = x[:, :, 1:] - x[:, :, :-1]

        # prepare exponential alpha

        alpha = self.alpha.sigmoid()
        alpha = rearrange(alpha, 'h -> h 1')

        # arange == powers

        arange = torch.arange(n, device = device)
        weights = alpha * (1 - alpha) ** torch.flip(arange, dims = (0,))
        output = conv1d_fft(x, weights)

        # get initial state contribution

        init_weight = (1 - alpha) ** (arange + 1)
        init_output = rearrange(init_weight, 'h n -> h n 1') * rearrange(self.initial_state, 'h d -> h 1 d')

        output = output + init_output

        # merge heads

        output = rearrange(output, 'b h n d -> b n (h d)')
        return self.project_out(output)

def FeedForward(dim, mult = 4):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim)
    )

class MetaformerGPT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        heads = 8,
        max_seq_len = 2048,
        ff_mult = 4
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MHESA(dim, heads = heads),
                MeanCenteringPool(dim),
                FeedForward(dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device = device))

        for mh_esa, pool, ff in self.layers:
            x = mh_esa(x) + x
            x = pool(x) + x
            x = ff(x) + x

        return self.to_logits(x)
