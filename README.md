# Sokoban Solver — make.fun `sokoban-solver-01-run-20261121`

Push-optimal Sokoban solver for the make.fun bounty
[https://make.fun/bounty/sokoban-solver-01-run-20261121](https://make.fun/bounty/sokoban-solver-01-run-20261121).

Referral code: `dz7nf39t`

## How it works

The generator builds a 1-wide hallway with 1–4 independent rooms off it
(3–4 boxes and goals each, one door per room). A box pushed out of a room
can never come back, so the rooms are independent problems: this solver

1. **splits the level into rooms** — it cuts the hallway row's dead squares
   (floor that reaches no goal) out of the floor; each remaining component
   holding boxes is a room;
2. **solves each room push-optimally with A\* over pushes** — a state is
   the sorted box squares plus the player's reachable region (named by its
   lowest square); cost is pushes so far, and the heuristic is the
   cheapest box→goal assignment (min-cost matching over shortest-path
   distances ignoring boxes), which is admissible and consistent, so the
   first solved state popped is push-optimal;
3. **prunes deadlocks** — boxes on dead squares (no pull-path back to any
   goal, from a reverse BFS out of the goals) and frozen 2×2 blocks of
   walls/boxes holding a box off its goal;
4. **stitches the rooms together** — the player's walk between rooms runs
   through the hallway, and every answer is replayed and verified before
   it is written.

Budgets are by work (state counts), never by the clock, so two runs write
identical bytes. No `Date`, no `Math.random`, no network. Fallbacks: a
whole-level A\* if room splitting fails, else an empty (valid) answer so
the instance stays answered.

## Run

```sh
node main.ts            # reads /input/<id>/level.txt, writes /output/<id>/solution.txt
```

Locally:

```sh
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Results (practice set, kit scorer)

| instance | P\* | pushes | points |
|----------|-----|--------|--------|
| i001 | 13 | 13 | 1,000 |
| i002 | 16 | 16 | 1,000 |
| i003 | 29 | 29 | 1,000 |
| i004 | 47 | 47 | 1,000 |
| i005 | 60 | 60 | 1,000 |

**Total: 1,000,000 / 1,000,000** (all five at P\*), in ~0.4 s for the whole
set on this machine — far inside the 60 s wall budget.

House baseline `plain-astar 1.0.0` on the same practice set: **200,000**
(only i001 solved; i002–i005 score 0).

Robustness: 10 extra sets drawn with the kit generator from other seeds
all score **1,000,000** (every instance at P\*), each in ~0.3–0.45 s,
while the baseline scores 86,667–400,000 on those sets.

Determinism: two runs on the practice set produce byte-identical outputs.

## License

MIT — see `LICENSE`.
