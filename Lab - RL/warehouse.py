"""Warehouse environments (single robot and two robots). NumPy only."""
import numpy as np

N = 5
OBST = {(0, 2), (1, 2), (3, 2), (4, 2)}
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # up, down, left, right, wait
ACTIONS = ["Up", "Down", "Left", "Right", "Wait"]
GAMMA = 0.95
MAX_STEPS = 40


def idx(p):
    return 5 * p[0] + p[1]


def _move(p, a):
    """Return (new_pos, invalid) for one robot."""
    dr, dc = MOVES[a]
    r, c = p[0] + dr, p[1] + dc
    if not (0 <= r < N and 0 <= c < N) or (r, c) in OBST:
        return p, True
    return (r, c), False


class SingleRobot:
    def __init__(self, start=(2, 0), goal=(2, 4)):
        self.start, self.goal = start, goal
        self.reset()

    def reset(self):
        self.pos, self.t = self.start, 0
        return self.pos

    def set_pos(self, p):  # test mode
        self.pos, self.t = p, 0

    def step(self, a):
        new, invalid = _move(self.pos, a)
        self.pos = new
        self.t += 1
        if new == self.goal:
            r, term = 10.0, True
        elif invalid:
            r, term = -1.0, False
        else:
            r, term = -0.1, False
        trunc = (not term) and self.t >= MAX_STEPS
        return new, r, term, trunc, {"invalid": invalid}


class TwoRobot:
    goals = ((2, 4), (2, 0))
    starts = ((2, 0), (2, 4))

    def __init__(self, starts=None):
        self.starts_ = starts or self.starts
        self.reset()

    def reset(self):
        self.pos, self.t = tuple(self.starts_), 0
        return self.pos

    def set_pos(self, pA, pB):
        self.pos, self.t = (pA, pB), 0

    def step(self, joint):
        aA, aB = divmod(joint, 5)
        acts = (aA, aB)
        old = self.pos
        new, inv, at_goal_before = [], 0, []
        for i in range(2):
            ag = old[i] == self.goals[i]
            at_goal_before.append(ag)
            if ag:
                new.append(old[i])  # stays regardless of action
            else:
                p, bad = _move(old[i], acts[i])
                new.append(p)
                inv += bad
        same = new[0] == new[1]
        swap = new[0] == old[1] and new[1] == old[0] and old[0] != old[1]
        self.t += 1
        if same or swap:  # collision overrides everything; keep old positions
            return old, -10.0, True, False, {"collision": True, "invalid": 0}
        self.pos = tuple(new)
        k = sum((new[i] == self.goals[i]) and not at_goal_before[i] for i in range(2))
        both = all(new[i] == self.goals[i] for i in range(2))
        r = -0.1 - inv + 5 * k + 5 * int(both)
        trunc = (not both) and self.t >= MAX_STEPS
        return self.pos, r, both, trunc, {"collision": False, "invalid": inv}
