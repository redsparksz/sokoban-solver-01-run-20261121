// Sokoban solver: push-optimal per room, then stitched into one walk.
//
// Design (written from scratch for this entry):
//  - The generator builds a 1-wide hallway with 1-4 independent rooms off it
//    (each room 3-4 boxes/goals, one door). A box pushed out of a room can
//    never come back, so rooms are independent: solve each room push-optimal
//    from the player's current square and stitch the walks together.
//  - Per room: A* over PUSHES (not steps). A state is the sorted box
//    squares plus the player's reachable region (named by its lowest
//    square). Cost g = pushes so far; heuristic h = the cheapest
//    box->goal assignment (min-cost matching over shortest-path distances
//    ignoring boxes), which is admissible and consistent, so the first
//    solved state popped is push-optimal.
//  - Deadlock pruning: a box on a "dead" square (no pull-path back to any
//    goal, computed by reverse BFS from the goals) can never be solved, and
//    a frozen 2x2 block of walls/boxes holding a box off its goal is pruned.
//  - Everything is budgeted by WORK (state counts), never by the clock, so
//    two runs write identical bytes. No Date, no Math.random, no env-driven
//    variation. Fallbacks: whole-level A* if room splitting fails, else an
//    empty (valid, 0-point) answer so the instance stays answered.
//  - Every answer is replayed and verified before it is written.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (state counts, deterministic). The generator keeps a room
// only when its own push search solves it within 250k states; A* with the
// assignment heuristic expands far fewer than that, so these caps are headroom.
const MAX_ROOM_STATES = 900_000;
const MAX_WHOLE_STATES = 900_000;

const DIRS: readonly { dx: number; dy: number; step: string; push: string }[] = [
  { dx: 0, dy: -1, step: 'u', push: 'U' },
  { dx: 0, dy: 1, step: 'd', push: 'D' },
  { dx: -1, dy: 0, step: 'l', push: 'L' },
  { dx: 1, dy: 0, step: 'r', push: 'R' },
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares from which a box can still be pushed onto some goal: reverse
// pull BFS from the goals. A box on any other square is dead.
function liveSquares(level: Level, wall: Uint8Array): Uint8Array {
  const { w, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const d of DIRS) {
      const delta = d.dy * w + d.dx;
      const y = x - delta;
      if (!wall[y] && !wall[y - delta] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Shortest walking distances from each goal over floor (walls only),
// used by the assignment heuristic.
function goalDistances(level: Level, wall: Uint8Array): Int32Array[] {
  const { w, goals } = level;
  const out: Int32Array[] = [];
  for (const g of goals) {
    const dist = new Int32Array(wall.length).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (const d of DIRS) {
        const q = p + d.dy * w + d.dx;
        if (!wall[q] && dist[q] < 0) {
          dist[q] = dist[p]! + 1;
          queue.push(q);
        }
      }
    }
    out.push(dist);
  }
  return out;
}

// Cheapest assignment of boxes to goals (brute-force permutations, n <= 4
// here, so at most 24). Infinity when some box cannot reach any goal.
function assignmentHeuristic(boxes: number[], dists: Int32Array[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  const used = new Array<boolean>(dists.length).fill(false);
  let best = Infinity;
  const rec = (i: number, acc: number): void => {
    if (acc >= best) return;
    if (i === n) {
      best = acc;
      return;
    }
    for (let j = 0; j < dists.length; j++) {
      if (used[j]) continue;
      const d = dists[j]![boxes[i]!]!;
      if (d < 0) continue;
      used[j] = true;
      rec(i + 1, acc + d);
      used[j] = false;
    }
  };
  rec(0, 0);
  return best;
}

// Flood fill from `from` around walls and boxes; returns the lowest
// reachable square and marks the region in `seen` with `stamp`.
function floodRegion(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (const d of DIRS) {
      const q = p + d.dy * w + d.dx;
      if (!wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// True when the box just pushed to `t` sits in a frozen 2x2 block: four
// squares of walls/boxes, at least one box off its goal.
function frozen2x2(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  goalSet: Uint8Array,
  t: number,
): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a])) continue;
    if (!(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c])) continue;
    if (!(wall[d] || occupied[d])) continue;
    if (
      (occupied[a] && !goalSet[a]) ||
      (occupied[b] && !goalSet[b]) ||
      (occupied[c] && !goalSet[c]) ||
      (occupied[d] && !goalSet[d])
    ) {
      return true;
    }
  }
  return false;
}

// Shortest walk from `from` to `to` around walls and boxes (BFS, direction
// order u,d,l,r for deterministic ties). Throws when unreachable.
function walkTo(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  to: number,
): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const d of DIRS) {
      const q = p + d.dy * w + d.dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = d.step;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walk target unreachable');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

interface PushSolution {
  pushes: number;
  moves: string;
}

// A* over pushes for one (sub-)level. `wall` already has other rooms'
// boxes/goals folded in as walls. Returns null when the state cap is hit
// or the level is unsolvable.
function solvePushAStar(
  level: Level,
  wall: Uint8Array,
  boxes0: number[],
  goals: number[],
  player0: number,
  maxStates: number,
): PushSolution | null {
  const { w } = level;
  const size = wall.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const live = liveSquares({ ...level, goals }, wall);
  if (boxes0.some((b) => !live[b])) return null;
  const dists = goalDistances({ ...level, goals }, wall);
  const solved = (bs: number[]): boolean => bs.every((b) => goalSet[b]);

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  // State key: region-low, then box squares, as digits in base `size`.
  const keyOf = (boxes: number[], low: number): number => {
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  const boxesArr: number[][] = [boxes0.slice().sort((a, b) => a - b)];
  const playerArr: number[] = [player0];
  const gArr: number[] = [0];
  const parent: number[] = [-1];
  // push that created the state: [boxFrom, boxTo]
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const best = new Map<number, number>();
  for (const b of boxesArr[0]!) occupied[b] = 1;
  const low0 = floodRegion(w, wall, occupied, player0, seen, ++stamp, stack);
  for (const b of boxesArr[0]!) occupied[b] = 0;
  best.set(keyOf(boxesArr[0]!, low0), 0);

  // Binary heap ordered by (f asc, g desc, insertion seq asc).
  interface HeapEntry {
    f: number;
    g: number;
    seq: number;
    state: number;
  }
  const heap: HeapEntry[] = [];
  let seq = 0;
  const heapLess = (a: HeapEntry, b: HeapEntry): boolean =>
    a.f < b.f || (a.f === b.f && (a.g > b.g || (a.g === b.g && a.seq < b.seq)));
  const heapPush = (e: HeapEntry): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): HeapEntry => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  if (solved(boxesArr[0]!)) return { pushes: 0, moves: '' };
  heapPush({ f: assignmentHeuristic(boxesArr[0]!, dists), g: 0, seq: seq++, state: 0 });

  let found = -1;
  let expanded = 0;
  while (heap.length > 0) {
    const { g, state: s } = heapPop();
    // Recompute the state's key (its region depends on its boxes only).
    const boxes = boxesArr[s]!;
    for (const b of boxes) occupied[b] = 1;
    const low = floodRegion(w, wall, occupied, playerArr[s]!, seen, ++stamp, stack);
    const k = keyOf(boxes, low);
    const owner = best.get(k);
    if (owner !== s || gArr[s] !== g) {
      for (const b of boxes) occupied[b] = 0;
      continue;
    }
    if (solved(boxes)) {
      found = s;
      for (const b of boxes) occupied[b] = 0;
      break;
    }
    if (++expanded > maxStates) {
      for (const b of boxes) occupied[b] = 0;
      return null;
    }
    // `seen` now holds this state's region under stamp `stamp`.
    const here = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (const d of DIRS) {
        const delta = d.dy * w + d.dx;
        const from = b - delta;
        const to = b + delta;
        if (seen[from] !== here) continue;
        if (wall[to] || occupied[to] || !live[to]) continue;
        occupied[b] = 0;
        occupied[to] = 1;
        const frozen = frozen2x2(w, wall, occupied, goalSet, to);
        let next: number[] | null = null;
        let nkey = -1;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = to;
          next.sort((a, b2) => a - b2);
          const nlow = floodRegion(w, wall, occupied, b, mark, ++stamp, stack);
          nkey = keyOf(next, nlow);
        }
        occupied[to] = 0;
        occupied[b] = 1;
        if (frozen || next === null) continue;
        const prevOwner = best.get(nkey);
        if (prevOwner !== undefined && gArr[prevOwner]! <= g + 1) continue;
        if (boxesArr.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        const t = boxesArr.length;
        boxesArr.push(next);
        playerArr.push(b);
        gArr.push(g + 1);
        parent.push(s);
        pushFrom.push(b);
        pushTo.push(to);
        best.set(nkey, t);
        const h = assignmentHeuristic(next, dists);
        if (h === Infinity) {
          // Dead end by distance (should be caught by live squares, but be safe).
          continue;
        }
        heapPush({ f: g + 1 + h, g: g + 1, seq: seq++, state: t });
        if (solved(next)) {
          found = t;
          for (const x of boxes) occupied[x] = 0;
          break;
        }
      }
      if (found >= 0) break;
    }
    for (const b of boxes) occupied[b] = 0;
    if (found >= 0) break;
  }
  if (found < 0) return null;

  // Rebuild the push sequence, then the walks between pushes.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = player0;
  let boxes = boxesArr[0]!;
  for (const s of path) {
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const delta = to - from;
    const dir = DIRS.find((d) => d.dy * w + d.dx === delta)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(w, wall, occupied, player, from - delta);
    for (const b of boxes) occupied[b] = 0;
    moves += dir.push;
    player = from;
    const nb = boxes.slice();
    nb[nb.indexOf(from)] = to;
    nb.sort((a, b2) => a - b2);
    boxes = nb;
  }
  return { pushes: path.length, moves };
}

interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}

// Split the level into rooms: cut the hallway row's dead squares (floor
// that reaches no goal) out of the floor; each remaining component holding
// boxes is a room. Null when the split is not clean.
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level, wall);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const d of DIRS) {
        const q = squares[i]! + d.dy * w + d.dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Solve one level room by room; null when any room fails.
function solveByRooms(level: Level): PushSolution | null {
  const rooms = roomsOf(level);
  if (rooms === null) return null;
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const solved = solvePushAStar(
      level,
      wall,
      room.boxes.slice().sort((a, b) => a - b),
      room.goals.slice().sort((a, b) => a - b),
      player,
      MAX_ROOM_STATES,
    );
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    // Advance the player to where the room's moves left it.
    for (const ch of solved.moves.toLowerCase()) {
      player += ch === 'u' ? -level.w : ch === 'd' ? level.w : ch === 'l' ? -1 : 1;
    }
    room.done = true;
  }
  return { pushes, moves };
}

// Replay a move string on the level; returns pushes when it solves the
// level legally, -1 otherwise.
function replay(level: Level, moves: string): number {
  const { w, wall, goals } = level;
  const goalSet = new Uint8Array(wall.length);
  for (const g of goals) goalSet[g] = 1;
  const box = new Uint8Array(wall.length);
  for (const b of level.boxes) box[b] = 1;
  let player = level.player;
  let pushes = 0;
  for (const ch of moves) {
    const d = DIRS.find((x) => x.step === ch.toLowerCase());
    if (!d) return -1;
    const delta = d.dy * w + d.dx;
    const next = player + delta;
    if (wall[next]) return -1;
    if (box[next]) {
      const beyond = next + delta;
      if (wall[beyond] || box[beyond]) return -1;
      box[next] = 0;
      box[beyond] = 1;
      pushes++;
    }
    player = next;
  }
  for (let i = 0; i < box.length; i++) if (box[i] && !goalSet[i]) return -1;
  return pushes;
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  let sol = solveByRooms(level);
  if (sol === null) {
    sol = solvePushAStar(level, level.wall.slice(), level.boxes, level.goals, level.player, MAX_WHOLE_STATES);
  }
  if (sol === null) return '';
  const moves = sol.moves.replace(/[^udlrUDLR]/g, '');
  return replay(level, moves) >= 0 ? moves : '';
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
