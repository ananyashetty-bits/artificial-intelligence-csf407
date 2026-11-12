"""Task 4 + Task 7 tests (environment and TD update). Run: python tests.py"""
import numpy as np
from warehouse import *
from qlearn import td_target, q_update

env = SingleRobot()
print("== Single-robot environment tests ==")
for p, a, exp in [((0,0),0,((0,0),-1,False)), ((1,1),3,((1,1),-1,False)),
                  ((2,1),3,((2,2),-0.1,False)), ((2,1),4,((2,1),-0.1,False)),
                  ((2,3),3,((2,4),10,True))]:
    env.set_pos(p); p2, r, term, trunc, _ = env.step(a)
    ok = (p2, r, term) == exp
    print(p, ACTIONS[a], "->", p2, r, term, "PASS" if ok else "FAIL"); assert ok
env.reset()
for i in range(40): _, r, term, trunc, _ = env.step(4)
print("40 waits -> truncated:", trunc, "terminated:", term); assert trunc and not term

print("\n== TD update tests (Q=0.4, r=1, maxQ'=2, alpha=0.2, gamma=0.95) ==")
for name, r, term in [("non-terminal", 1, False), ("terminal r=5", 5, True), ("truncated (bootstrap)", 1, False)]:
    Q = np.zeros((2, 5)); Q[0, 0] = 0.4; Q[1, 1] = 2.0
    y = q_update(Q, 0, 0, r, 1, term)
    print(f"{name}: target={y:.3f}, new Q={Q[0,0]:.4f}")
Q = np.zeros((2,5)); Q[0,0]=0.4; Q[1,1]=2.0
assert abs(td_target(1,2,False)-2.9)<1e-9
Q2 = Q.copy(); q_update(Q2,0,0,1,1,False); assert abs(Q2[0,0]-0.9)<1e-9
Q3 = Q.copy(); q_update(Q3,0,0,5,1,True); assert abs(Q3[0,0]-1.32)<1e-9

print("\n== Two-robot tests ==")
e = TwoRobot(); J = lambda a, b: 5*a + b
R, L, U, W, D = 3, 2, 0, 4, 1
for A, B, a, b in [((2,1),(2,3),R,L), ((2,1),(2,2),R,L), ((2,1),(2,2),R,R), ((2,1),(2,2),U,W)]:
    e.set_pos(A, B); pos, r, term, trunc, info = e.step(J(a, b))
    print(f"A{A} B{B} {ACTIONS[a]}/{ACTIONS[b]} -> {pos} r={r:.1f} term={term} collision={info['collision']}")
# goal-sitting robot: A at goal, B moves; A requests Up, stays, no repeated reward
e.set_pos((2,4),(2,2)); pos, r, term, *_ = e.step(J(U, L)); print("A at goal asks Up, B Left ->", pos, round(r,2))
e.set_pos((2,4),(2,3)); pos, r, term, *_ = e.step(J(W, W)); print("A at goal, B waits ->", pos, round(r,2), "(no repeat arrival reward)")
e.set_pos((2,3),(2,1)); pos, r, term, *_ = e.step(J(R, W)); print("A arrives first time ->", pos, round(r,2))
e.set_pos((2,3),(1,0)); e.goals_ = None
