"""Find which op in ManoharanNet backward pulls in triton on CUDA (bm_manoharan distill crashed 2026-10-29 09:19:
triton needs Python.h to build its driver module, which this image lacks). Tiny tensors, < 1 GB GPU."""
import sys
import traceback

import torch
import torch.nn as nn

dev = "cuda"


def run(name, fn):
    try:
        fn()
        torch.cuda.synchronize()
        print("ok  ", name)
    except Exception as e:
        print("FAIL", name, type(e).__name__, str(e).splitlines()[0][:120])


def mha():
    m = nn.MultiheadAttention(128, 8, batch_first=True).to(dev)
    t = torch.randn(16, 4, 128, device=dev, requires_grad=True)
    y, _ = m(t, t, t, need_weights=False)
    y.sum().backward()


def mha_amp():
    m = nn.MultiheadAttention(128, 8, batch_first=True).to(dev)
    t = torch.randn(16, 4, 128, device=dev, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y, _ = m(t, t, t, need_weights=False)
    y.float().sum().backward()


def index_put():
    s = torch.randn(16, 4, device=dev, requires_grad=True)
    lg = torch.zeros(16, 100, 4, device=dev)
    lg[torch.arange(16, device=dev), torch.randint(0, 100, (16,), device=dev)] = s
    lg.sum().backward()


def full_net():
    sys.path.insert(0, ".")
    from baselines_ml.manoharan import ManoharanNet
    net = ManoharanNet(3, 8).to(dev)
    T, G = 3, 19
    D = 4 + T + G + 2 + 3 * T
    x = torch.randn(16, 100, D, device=dev)
    lg, st = net(x, None, 10, 10)
    (lg.sum() + st.sum()).backward()


def full_net_amp():
    sys.path.insert(0, ".")
    from baselines_ml.manoharan import ManoharanNet
    net = ManoharanNet(3, 8).to(dev)
    T, G = 3, 19
    D = 4 + T + G + 2 + 3 * T
    x = torch.randn(16, 100, D, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg, st = net(x, None, 10, 10)
    (lg.float().sum() + st.float().sum()).backward()


def lin1():
    m = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 128)).to(dev)
    m(torch.randn(16, 1, device=dev)).sum().backward()


def einsum_qk():
    q = torch.randn(16, 128, device=dev, requires_grad=True)
    k = torch.randn(16, 4, 128, device=dev, requires_grad=True)
    torch.einsum("ed,etd->et", q, k).sum().backward()


def empty_expand_cat():
    p = nn.Parameter(torch.randn(128, device=dev))
    a = torch.randn(16, 3, 128, device=dev, requires_grad=True)
    torch.cat([a, p.expand(16, 1, 128)], 1).sum().backward()


def layernorm():
    m = nn.LayerNorm(128).to(dev)
    m(torch.randn(16, 4, 128, device=dev, requires_grad=True)).sum().backward()


def mha_ln():
    m = nn.MultiheadAttention(128, 8, batch_first=True).to(dev)
    ln = nn.LayerNorm(128).to(dev)
    t = torch.randn(16, 4, 128, device=dev, requires_grad=True)
    y, _ = m(t, t, t, need_weights=False)
    ln(t + y).sum().backward()


def mha_param_tok():
    m = nn.MultiheadAttention(128, 8, batch_first=True).to(dev)
    e = nn.Linear(3, 128).to(dev)
    p = nn.Parameter(torch.randn(128, device=dev))
    tok = torch.cat([e(torch.randn(16, 3, 3, device=dev)), p.expand(16, 1, 128)], 1)
    y, _ = m(tok, tok, tok, need_weights=False)
    y.sum().backward()


def bmm_qk():
    q = torch.randn(16, 128, device=dev, requires_grad=True)
    k = torch.randn(16, 4, 128, device=dev, requires_grad=True)
    torch.bmm(k, q[:, :, None])[..., 0].sum().backward()


def mulsum_qk():
    q = torch.randn(16, 128, device=dev, requires_grad=True)
    k = torch.randn(16, 4, 128, device=dev, requires_grad=True)
    (q[:, None] * k).sum(-1).sum().backward()


def equal_qk():
    q = torch.randn(16, 128, device=dev)
    k = torch.randn(16, 4, 128, device=dev)
    a = torch.einsum("ed,etd->et", q, k)
    b = torch.bmm(k, q[:, :, None])[..., 0]
    print("     max |einsum - bmm| fwd", float((a - b).abs().max()))


if __name__ == "__main__":
    print("torch", torch.__version__)
    for n, f in [("bmm_qk", bmm_qk), ("mulsum_qk", mulsum_qk), ("equal_qk", equal_qk), ("full_net", full_net)]:
        run(n, f)
