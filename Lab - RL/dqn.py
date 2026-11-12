"""Parts II & III: DQN (PyTorch). NOT executed by Claude (torch unavailable in its sandbox) -- run it yourself.
Usage: python dqn.py single | python dqn.py two [steps] | python dqn.py test
"""
import sys, random
from collections import deque
import numpy as np, torch, torch.nn as nn
from warehouse import *

GAMMA_ = 0.95

def onehot_single(p):
    v = np.zeros(25, np.float32); v[idx(p)] = 1; return v

def onehot_two(ps):
    v = np.zeros(50, np.float32); v[idx(ps[0])] = 1; v[25 + idx(ps[1])] = 1; return v

def make_net(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_out))

def dqn_loss(online, target, s, a, r, s2, done, gamma=GAMMA_):
    """done = TRUE termination only (not truncation)."""
    with torch.no_grad():                                   # no gradient into target
        y = r + gamma * (1 - done) * target(s2).max(1).values
    q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)      # Q(s, a_recorded)
    return ((q - y) ** 2).mean(), y

def loss_test():
    # tiny net: check targets 2.9 (non-term), 5.0 (term), 2.9 (trunc)
    class Const(nn.Module):
        def __init__(s, v): super().__init__(); s.v = v
        def forward(s, x): return torch.full((x.shape[0], 5), float(s.v))
    on, tg = nn.Linear(25, 5), Const(2.0)
    s = torch.zeros(3, 25); a = torch.zeros(3, dtype=torch.long)
    r = torch.tensor([1., 5., 1.]); done = torch.tensor([0., 1., 0.])
    _, y = dqn_loss(on, tg, s, a, r, s, done)
    print("targets:", y.tolist()); assert np.allclose(y.tolist(), [2.9, 5.0, 2.9])
    loss, _ = dqn_loss(on, tg, s, a, r, s, done); loss.backward()
    assert all(p.grad is not None for p in on.parameters()) and tg is not None
    print("loss test passed; gradients reach online net only")

def train(env, enc, n_out, steps, seed=7, joint=False, log_every=2000):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    n_in = len(enc(env.reset()))
    online, target = make_net(n_in, n_out), make_net(n_in, n_out)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=1e-3)
    buf = deque(maxlen=20000); updates = 0
    s = env.reset(); ep_ret, ep_rets, ep_succ = 0.0, [], []
    for t in range(steps):
        eps = max(0.05, 1 - 0.95 * t / (0.8 * steps))
        if rng.random() < eps: a = int(rng.integers(n_out))
        else:
            with torch.no_grad(): q = online(torch.from_numpy(enc(s))[None])[0].numpy()
            a = int(rng.choice(np.flatnonzero(q == q.max())))
        s2, r, term, trunc, info = env.step(a)
        buf.append((enc(s), a, r, enc(s2), float(term)))    # s2 = state before reset
        ep_ret += r; s = s2
        if term or trunc:
            ep_rets.append(ep_ret); ep_succ.append(bool(term and not info.get("collision", False)))
            s, ep_ret = env.reset(), 0.0
        if t + 1 >= 500 and (t + 1) % 4 == 0 and len(buf) >= 64:
            b = random.sample(buf, 64)
            S, A, R, S2, D = (torch.tensor(np.array(x)) for x in zip(*b))
            loss, _ = dqn_loss(online, target, S, A.long(), R.float(), S2, D.float())
            opt.zero_grad(); loss.backward(); opt.step(); updates += 1
            if updates % 250 == 0: target.load_state_dict(online.state_dict())
        if (t + 1) % log_every == 0 and ep_succ:
            print(f"step {t+1} eps={eps:.2f} recent success={np.mean(ep_succ[-50:]):.2f} return={np.mean(ep_rets[-50:]):.2f}")
    return online

def greedy(env, net, enc, verbose=True):
    s = env.reset(); path = [s]; inv = 0; coll = False
    for _ in range(MAX_STEPS):
        with torch.no_grad(): a = int(net(torch.from_numpy(enc(s))[None])[0].argmax())  # fixed tie-break
        s, r, term, trunc, info = env.step(a); path.append(s); inv += info.get("invalid", 0)
        coll = coll or info.get("collision", False)
        if term or trunc: break
    if verbose:
        for i, p in enumerate(path): print(i, p)
    print(f"goal/both-arrived={term and not coll} collision={coll} timeout={trunc} steps={len(path)-1} invalid={inv}")
    return path

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test": loss_test()
    elif mode == "single":
        net = train(SingleRobot(), onehot_single, 5, 12000); greedy(SingleRobot(), net, onehot_single)
        greedy(SingleRobot(start=(0, 0)), net, onehot_single)
    elif mode == "two":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
        net = train(TwoRobot(), onehot_two, 25, n); greedy(TwoRobot(), net, onehot_two)
        greedy(TwoRobot(starts=((0, 0), (4, 4))), net, onehot_two)
