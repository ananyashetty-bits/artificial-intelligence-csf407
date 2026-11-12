"""Part I: tabular Q-learning (NumPy). Run: python qlearn.py"""
import numpy as np
from warehouse import *

def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(5))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))   # random tie-break

def td_target(r, max_next, terminated, gamma=GAMMA):
    return r if terminated else r + gamma * max_next        # truncation bootstraps

def q_update(Q, s, a, r, s2, terminated, alpha=0.2, gamma=GAMMA):
    y = td_target(r, Q[s2].max(), terminated, gamma)
    Q[s, a] += alpha * (y - Q[s, a])
    return y

def train(episodes=1500, alpha=0.2, decay_eps=1200, seed=7, eps_fixed=None, zero_tie_greedy=False):
    rng = np.random.default_rng(seed)
    env, Q = SingleRobot(), np.zeros((25, 5))
    rets, succ, epss = [], [], []
    for ep in range(episodes):
        eps = eps_fixed if eps_fixed is not None else max(0.05, 1 - 0.95 * ep / decay_eps)
        s, G, ok = idx(env.reset()), 0.0, False
        while True:
            a = eps_greedy(Q, s, eps, rng)
            p2, r, term, trunc, _ = env.step(a)
            s2 = idx(p2)                      # next state BEFORE reset
            q_update(Q, s, a, r, s2, term, alpha)
            G += r; s = s2
            if term: ok = True
            if term or trunc: break
        rets.append(G); succ.append(ok); epss.append(eps)
    return Q, np.array(rets), np.array(succ), np.array(epss)

def greedy_rollout(Q, start=(2, 0)):
    env = SingleRobot(start=start); p = env.reset(); path = [p]; inv = 0
    for _ in range(MAX_STEPS):
        a = int(np.argmax(Q[idx(p)]))         # fixed tie-break: lowest action code
        p, r, term, trunc, info = env.step(a); path.append(p); inv += info["invalid"]
        if term or trunc: break
    return path, term, trunc, inv

def random_policy(n=100, seed=7):
    rng = np.random.default_rng(seed); env = SingleRobot()
    succ, steps, to, inv = 0, [], 0, 0
    for _ in range(n):
        env.reset()
        while True:
            _, r, term, trunc, info = env.step(int(rng.integers(5))); inv += info["invalid"]
            if term: succ += 1; steps.append(env.t)
            if term or trunc:
                to += trunc; break
    return succ/n, (np.mean(steps) if steps else None), to/n, inv/n

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Q, rets, succ, epss = train()
    ma = lambda x, w=50: np.convolve(x, np.ones(w)/w, mode="valid")
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    ax[0].plot(ma(rets)); ax[0].set_title("Training return (MA 50)")
    ax[1].plot(ma(succ.astype(float))); ax[1].set_title("Training success (MA 50)")
    ax[2].plot(epss); ax[2].set_title("epsilon")
    for a in ax: a.set_xlabel("episode")
    plt.tight_layout(); plt.savefig("q_training.png", dpi=120)
    path, term, trunc, inv = greedy_rollout(Q)
    print("Greedy path:", path, "\nlength:", len(path)-1, "goal:", term, "timeout:", trunc, "invalid:", inv)
    print("Random policy (success, mean steps|success, timeout frac, mean invalid/run):", random_policy())
    print("Extra start (0,0):", greedy_rollout(Q, (0, 0)))
    np.save("Q_table.npy", Q)
    # Optional: no exploration
    for sd in (7, 8):
        Q0, _, s0, _ = train(eps_fixed=0.0, seed=sd)
        p0, t0, tr0, i0 = greedy_rollout(Q0)
        print(f"eps=0 seed {sd}: train success {s0.mean():.3f}; greedy goal={t0} len={len(p0)-1} invalid={i0}")
    # plot trajectory
    g = np.zeros((5, 5));
    for o in OBST: g[o] = 1
    plt.figure(figsize=(3.5, 3.5)); plt.imshow(g, cmap="Greys", vmin=0, vmax=2)
    ys, xs = zip(*path); plt.plot(xs, ys, "o-", color="tab:blue")
    plt.text(0, 2, "S", color="r", ha="center", va="center"); plt.text(4, 2, "G", color="g", ha="center", va="center")
    plt.title("Greedy trajectory"); plt.savefig("q_trajectory.png", dpi=120)
