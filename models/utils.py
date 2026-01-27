import torch
import torch.nn as nn
import torch.nn.functional as F

def clammped_one_hot(x, num_classes):
    mask = (x >= 0) & (x < num_classes)
    x = x.clamp(min=0, max=num_classes - 1)
    y = F.one_hot(x, num_classes=num_classes) * mask[..., None]
    return y


def tuple_sum(*args):
    '''
    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.

    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.

    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]



def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    # x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
    # print("x shape:", x.shape)
    # print("x:", x)
    # print("axis:", axis)
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    # print("out shape:", out.shape)
    # print("out:", out)
    return torch.sqrt(out) if sqrt else out

def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple.
    Should be used only with `_merge(s, V)` and only if the tuple
    representation cannot be used.

    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    s = x[..., :-3 * nv]
    v = x[..., -3 * nv:].contiguous().view(x.shape[0], nv, 3)
    return s, v


def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = v.contiguous().view(v.shape[0], v.shape[1] * 3)
    return torch.cat([s, v], -1)

class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''

    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn

class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''

    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = x.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x

class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''

    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)