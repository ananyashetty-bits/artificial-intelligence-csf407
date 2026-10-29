"""
Logical Planning Lab - Locked-Door Navigation Domain
----------------------------------------------------
State      : a frozenset of propositions (strings)
Action     : name + positive/negative preconditions + positive/negative effects
Applicable : all positive preconditions in state, no negative precondition in state
Apply      : remove negative effects, then add positive effects
Search     : breadth-first search (returns a shortest plan)

Generated with the assistance of an LLM (Claude); tested/verified separately by
validate_plan(), which re-executes the plan independently of the BFS.
"""
from collections import deque
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

State = FrozenSet[str]


@dataclass(frozen=True)
class Action:
    name: str
    pos_pre: FrozenSet[str] = frozenset()
    neg_pre: FrozenSet[str] = frozenset()
    pos_eff: FrozenSet[str] = frozenset()
    neg_eff: FrozenSet[str] = frozenset()


def act(name, pos_pre=(), neg_pre=(), add=(), delete=()):
    return Action(name, frozenset(pos_pre), frozenset(neg_pre),
                  frozenset(add), frozenset(delete))


# ---------------------------------------------------------------- logic part
def applicable(state: State, a: Action) -> bool:
    """S |= Preconditions(a)"""
    return a.pos_pre <= state and not (a.neg_pre & state)


def apply_action(state: State, a: Action) -> State:
    """S' = (S - neg_eff) + pos_eff"""
    return frozenset((state - a.neg_eff) | a.pos_eff)


def goal_satisfied(state: State, goal: FrozenSet[str]) -> bool:
    """S |= G"""
    return goal <= state


# --------------------------------------------------------------- search part
def bfs_plan(initial: State, goal: FrozenSet[str],
             actions: List[Action]) -> Optional[List[Tuple[Action, State]]]:
    """Return list of (action, resulting_state) or None if no plan exists."""
    if goal_satisfied(initial, goal):
        return []
    frontier = deque([initial])
    parent = {initial: None}          # state -> (previous_state, action)
    while frontier:
        s = frontier.popleft()
        for a in actions:
            if not applicable(s, a):  # logic decides what is possible
                continue
            s2 = apply_action(s, a)
            if s2 in parent:          # already explored
                continue
            parent[s2] = (s, a)
            if goal_satisfied(s2, goal):
                path = []
                cur = s2
                while parent[cur] is not None:
                    prev, a_used = parent[cur]
                    path.append((a_used, cur))
                    cur = prev
                return list(reversed(path))
            frontier.append(s2)       # search decides what to try
    return None


# ------------------------------------------------- independent verification
def validate_plan(initial: State, goal: FrozenSet[str],
                  plan: List[Tuple[Action, State]]) -> Tuple[bool, str]:
    """Re-execute the plan step by step; do NOT trust the BFS bookkeeping."""
    s = initial
    for i, (a, claimed) in enumerate(plan, 1):
        missing = a.pos_pre - s
        blocked = a.neg_pre & s
        if missing or blocked:
            return False, f"step {i} {a.name}: preconditions fail (missing={set(missing)}, blocked={set(blocked)})"
        s = apply_action(s, a)
        if s != claimed:
            return False, f"step {i} {a.name}: reported state differs from recomputed state"
        at_count = sum(1 for p in s if p.startswith("At("))
        if at_count != 1:
            return False, f"step {i}: invariant 'exactly one At(.)' violated: {sorted(s)}"
    if not goal_satisfied(s, goal):
        return False, "final state does not satisfy goal"
    return True, "valid"


# --------------------------------------------------------------- the domain
def locked_door_actions() -> List[Action]:
    return [
        act("Move(A,B)", pos_pre=["At(A)"], add=["At(B)"], delete=["At(A)"]),
        act("Move(B,A)", pos_pre=["At(B)"], add=["At(A)"], delete=["At(B)"]),
        act("PickKey", pos_pre=["At(A)", "KeyAt(A)"], add=["HasKey"], delete=["KeyAt(A)"]),
        act("Unlock", pos_pre=["At(B)", "HasKey"], add=["Unlocked"]),
        act("Move(B,C)", pos_pre=["At(B)", "Unlocked"], add=["At(C)"], delete=["At(B)"]),
        act("Move(C,B)", pos_pre=["At(C)", "Unlocked"], add=["At(B)"], delete=["At(C)"]),
    ]


INITIAL = frozenset({"At(A)", "KeyAt(A)"})
GOAL = frozenset({"At(C)"})


def fmt(state):
    return "{" + ", ".join(sorted(state)) + "}"


def run_test(title, initial, goal, actions):
    print("=" * 70)
    print(title)
    print("Initial :", fmt(initial))
    print("Goal    :", fmt(goal))
    print("Actions :", ", ".join(a.name for a in actions))
    plan = bfs_plan(initial, goal, actions)
    if plan is None:
        print("Result  : No plan found")
        return None
    print("Plan found (length %d): %s" % (len(plan), " -> ".join(a.name for a, _ in plan) or "(empty)"))
    s = initial
    print("  S0 =", fmt(s))
    for i, (a, s2) in enumerate(plan, 1):
        print(f"  S{i} = {fmt(s2)}   after {a.name}")
    ok, msg = validate_plan(initial, goal, plan)
    print("Independent validation:", ok, "-", msg)
    return plan


if __name__ == "__main__":
    base = locked_door_actions()

    # Applicability checks (Task 0 style)
    print("Applicable actions in the initial state:",
          [a.name for a in base if applicable(INITIAL, a)])

    # Test A: solvable
    run_test("TEST A: Solvable problem", INITIAL, GOAL, base)

    # Test B: impossible - robot cannot pick up the key (PickKey removed)
    run_test("TEST B1: Impossible (PickKey removed)", INITIAL, GOAL,
             [a for a in base if a.name != "PickKey"])
    # Test B2: impossible - key is not in room A, robot never has it
    run_test("TEST B2: Impossible (key not in A at start)", frozenset({"At(A)"}), GOAL, base)

    # Test C: irrelevant / distractor actions
    distractors = base + [
        act("Wave", add=["Waved"]),                                   # no effect on position
        act("Spin(A)", pos_pre=["At(A)"], add=["Dizzy"]),             # irrelevant
        act("Move(A,A)", pos_pre=["At(A)"], add=["At(A)"], delete=["At(A)"]),  # no-op loop
    ]
    run_test("TEST C1: Irrelevant actions added (goal At(C))", INITIAL, GOAL, distractors)

    # Test C2: a goal that is only partly related: reaching B must NOT be treated as reaching C
    run_test("TEST C2: Goal At(B) (reaching B is not reaching C)", INITIAL,
             frozenset({"At(B)"}), base)

    # Test C3: the door must really be opened - Unlock removed, so C is unreachable
    run_test("TEST C3: Unlock removed (door stays locked)", INITIAL, GOAL,
             [a for a in base if a.name != "Unlock"])
