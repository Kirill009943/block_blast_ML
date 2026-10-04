# Block Blast AI

A complete Block Blast game in Python with a self-learning AI based on
Reinforcement Learning (PPO). The project grew out of a small console
implementation and is split into three independent layers:

```
game/          pure game engine (no I/O, no ML)
environment/   Gymnasium RL environment (no UI)
agents/        Random / Heuristic / Solver / Imitation / RL agents
training/      PPO training, evaluation, plotting
ui/            Pygame interface (human play + AI watch mode)
```

The engine knows nothing about ML or the UI; the RL code never touches
board internals directly; the UI only calls public engine/agent APIs.

## How Block Blast works

* The board is 8x8.
* You have three pieces at a time, drawn randomly from 34 shapes in random
  rotations (the generator only deals pieces that fit the current board).
* A piece can be placed on any set of empty cells it fits.
* Complete rows **and** columns are cleared simultaneously and score points.
* Once all three pieces are used, three new ones are drawn.
* The game ends when none of the remaining pieces can be placed.

Scoring (defaults, configurable in `game/game.py: GameConfig`):
+1 per placed cell, +10 per cleared line, +10 extra per additional
simultaneous line. Set `points_per_cell=0, combo_bonus=0` to reproduce the
original console game's pure `10 * lines` scoring.

## Installation

Requires **Python 3.10+**. Tested on Windows with an NVIDIA GPU.

```bash
pip install -r requirements.txt
# CUDA build of PyTorch (recommended for training; pick your CUDA version):
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

`sb3-contrib` is required because the action space is highly constrained:
of the 192 possible `(piece, row, col)` actions most are illegal at any
moment. `MaskablePPO` applies the environment's action mask so the agent
only ever samples legal placements instead of wasting training on invalid
actions.

## Playing

```bash
python main.py                    # Pygame UI, human mode
python main.py --mode console     # classic text-mode game
python main.py --mode watch       # watch the best available AI play
python main.py --mode watch --agent heuristic
python main.py --mode watch --agent solver    # near-perfect search player
python main.py --mode watch --agent rl:models/main_final.zip
```

UI controls: click a piece, then click a board cell (ghost preview shows
legality). `H` toggles Human/AI-watch, `R` restarts, `D` toggles the AI
debug panel (chosen piece, position, value estimate, valid-action count,
last reward), `+`/`-` change the AI move delay (default 0.1 s).

## Training the AI

```bash
python -m training.train                          # full run (10M steps)
python -m training.train --timesteps 50000        # quick smoke run
python -m training.train --timesteps 10000000 --n-envs 16 --seed 42
python -m training.train --resume models/checkpoints/ppo_block_blast_250000_steps.zip
```

All PPO hyperparameters are CLI-configurable (defaults = the proven
baseline): `--lr --gamma --gae-lambda --ent-coef --clip-range --n-steps
--batch-size --n-epochs --vf-coef --max-grad-norm`, plus
`--reward-profile baseline|survival|lines|strategic` and
`--observation-profile basic|enhanced`. The full resolved configuration is
saved to `results/config_<run>.json` together with reproducibility
metadata (Python/torch/CUDA/SB3 versions, GPU, git commit).

The script prints the device it uses:

```
Device: cuda
GPU: NVIDIA GeForce RTX 4070 SUPER
```

Outputs:

| What | Where |
|---|---|
| Final model | `models/<run>_final.zip` |
| Model saved on Ctrl+C | `models/<run>_interrupted.zip` |
| Best model (by masked eval) | `models/best/best_model.zip` |
| Checkpoints | `models/checkpoints/` |
| Per-episode training log | `results/training_log_<run>.csv` |
| Eval history | `results/eval/evaluations.npz` |
| TensorBoard | `results/tensorboard/<run>/` |

Progress and stopping: every `--progress-interval` seconds (default 30,
`0` disables) a `[timer]` line prints steps done, elapsed time, current
steps/sec and the estimated time remaining. Pressing **Ctrl+C** stops
training gracefully and saves the current model to
`models/<run>_interrupted.zip` — resume it with
`python -m training.train --resume models/<run>_interrupted.zip --timesteps ...`.

Training runs fully headless in 8 parallel environments
(`SubprocVecEnv`) at roughly 2,500–3,000 steps/sec on a desktop CPU+GPU;
rendering is never done during training.

### Observing training live

```bash
# terminal 1
python -m training.train --timesteps 5000000 --run-name myrun
# terminal 2 — live dashboard (updates every 5 s, also saves a PNG)
python -m training.dashboard --run myrun --total-steps 5000000
# TensorBoard
python -m tensorboard.main --logdir results/tensorboard
```

The dashboard (`training/dashboard.py`) polls the training CSVs in a
separate process — it never slows training down — and plots reward, score,
episode length, lines cleared, policy/value loss, entropy, explained
variance and learning rate with rolling-mean overlays, plus current
steps/episodes/FPS/best score/elapsed/ETA. It does not use Pygame.

TensorBoard namespaces: `game/*` (score, reward, episode length, lines
cleared, valid actions, game-over rate, invalid-action rate, and every
reward component as `game/reward_<name>`), `training/*` (aliases),
`performance/*` (fps, episodes), plus SB3's own `train/*` (entropy,
approx KL, losses, explained variance, learning rate), `rollout/*` and
`eval/*`. Per-update optimizer stats also land in
`results/metrics_<run>.csv`; per-episode stats with per-component reward
sums in `results/training_log_<run>.csv`.

### Plotting learning progress

```bash
python -m training.plot_results --log results/training_log_<run>.csv
tensorboard --logdir results/tensorboard
```

This produces the two graphs — *training steps vs average episode reward*
and *training steps vs average game score* — saved next to the log file.

## Evaluating and comparing agents

```bash
python -m training.evaluate --agent random --games 200
python -m training.evaluate --agent heuristic --games 100
python -m training.evaluate --agent rl:models/main_final.zip --games 200
python -m training.evaluate --agent solver --games 5 --max-moves 2000
```

Reports average/median/best/worst score, standard deviation, average game
length and average lines cleared; appends to
`results/evaluation_results.csv` and writes a full JSON dump per run.
`--max-moves` caps game length — required for the **solver**, which
effectively never reaches game over (see below).

### The solver agent

`agents/solver_agent.py` is a pure search player (no learning): a beam
search plans the ENTIRE current piece set — every ordering and placement
of the remaining pieces — and scores complete sequences by lines cleared
plus board quality (holes, fragmentation, bumpiness, height). Because the
board-aware generator only deals sequentially viable sets, always playing
a completing sequence means the solver does not die in practice: it
survived 3 x 1000-move and 1 x 2500-move test runs (~10.5k score per 1000
moves, ~550 lines). Two safety nets back the beam search: an exhaustive
node-budgeted survival DFS (used once per ~3000 moves) and the heuristic
agent as a last resort. "Never dies" is empirical, not proven — a
non-viable fallback set is possible in principle. At ~20 ms/move it is
fine for watching and benchmarking, but do not use it for training
rollouts.

Baseline results on this machine (200 games each, identical seeds 1000+):

| Agent | Avg score | Median | Best | Avg moves | Avg lines |
|---|---|---|---|---|---|
| Random | 68.5 | 56 | 385 | 16.0 | 1.5 |
| Heuristic | 705.3 | 526 | 3056 | 98.6 | 36.2 |
| PPO (500k steps) | 76.0 | 64 | 223 | 17.1 | 1.8 |
| PPO (5M steps) | 141.9 | 135 | 407 | 25.5 | 5.3 |

PPO clearly learns — the 5M-step model doubles the random baseline and
its learning curve is still rising — but the hand-crafted heuristic is a
strong benchmark that needs much longer training to beat. Measure, don't
assume. The 5M learning curves are in `results/training_log_main.png`.

## How the RL works

### State representation

`Box(0, 1, shape=(4, 8, 8), float32)` — four binary channels:

* channel 0: board occupancy (0 = empty, 1 = occupied),
* channels 1–3: one channel per piece slot, the piece rendered in an 8x8
  grid at its natural origin; all zeros when the slot is used.

(board, piece slots) fully determines the future, so the state is
Markovian. The channel-first layout feeds a CNN directly.

### Action representation

`Discrete(192)` = 3 pieces x 8 rows x 8 cols, decoded as
`piece = a // 64`, `row = (a % 64) // 8`, `col = a % 8`.
`env.action_masks()` exposes the legal subset; `MaskablePPO` enforces it
during both rollouts and inference. `get_valid_actions()` on the engine is
the single source of truth for legality.

### Reward function

All weights live in `environment/block_blast_env.py: RewardConfig`:

```
reward = place_per_cell * cells_placed        # +1.0 per cell (survival)
       + line  * lines_cleared                # +10 per line
       + combo * lines_cleared ** 2           # +5, superlinear multi-line bonus
       - holes * holes_created                # -0.1 per new hole (shaping)
       - game_over                            # -50 on termination
```

Placing pieces is the only way to make progress, so the placement reward
cannot be farmed without actually playing; the game-over penalty and the
superlinear combo term push the policy toward survival and line clears.

### Network and algorithm

PPO (`sb3_contrib.MaskablePPO`) with a custom CNN feature extractor
(`training/cnn_extractor.py`): Conv 4->64 -> Conv 64->64 -> Conv 64->128
(all 3x3, padding 1, keeping the 8x8 resolution), flatten, FC -> 256,
then separate policy/value heads (`pi=[256,256]`, `vf=[256,256]`).
Hyperparameters: lr 3e-4, n_steps 512 x 8 envs, batch 1024, 4 epochs,
gamma 0.995, GAE 0.95, clip 0.2, entropy 0.01.

The agent starts from random weights and improves purely through
environment interaction — there is no hard-coded strategy in the RL agent.

## Reproducibility

* `env.reset(seed=42)` reseeds the engine RNG — the same seed yields the
  same initial board and the same piece sequence (covered by tests).
* `python -m training.train --seed 42 ...` seeds torch, numpy and all
  vectorized environments.
* `python -m training.evaluate --agent X --seed 1000` plays game `i` with
  seed `1000 + i`, so agent comparisons use identical game sequences.

## Seeing what the network does

```bash
# architecture graph + conv filters + feature maps + policy heatmaps
python -m training.visualize_network --model models/main_final.zip --mode all
# AI Brain mode: live board + action-probability heatmaps + value + top-5 + activations
python main.py --mode brain --agent rl:models/main_final.zip --ai-delay 0.3
```

`training/visualize_network.py` produces:

* `results/network_architecture.png` — layer graph with tensor shapes,
  parameter counts, activations and trainable status.
* `results/activations/conv_weights.png` — conv filter grids + FC weight
  distribution.
* `results/activations/feature_maps_seed<N>.png` — CNN feature maps for a
  real game state.
* `results/policy_maps/policy_seed<N>.png` — the 192 masked action
  probabilities as three 8x8 heatmaps (one per piece slot), illegal
  actions in gray, selected action outlined.

These are diagnostic views of what the network computes — they do not
prove the network "understands" the game.

## Reward and observation profiles

`--reward-profile` (`environment/rewards.py`, every component is logged
individually under `game/reward_*`):

* `baseline` — the permanent reference: +1/cell, +10/line, +5×lines²,
  −0.1/new hole, −50 on game over.
* `survival` — heavier game-over penalty (−100) + small per-move bonus.
* `lines` — line clears dominate (+15/line, +10×lines²), cells worth less.
* `strategic` — baseline + absolute shaping against holes (−0.2),
  fragmentation (−0.05/region), occupancy (−0.02/cell) and +0.02 per
  remaining valid move.
* `balanced` — long-term-survival profile: +0.5/cell, +10/line,
  +4×lines², −0.25/new hole, −0.2 per net new empty region (delta),
  +0.05 per net change in legal options (delta, clipped ±3, skipped on
  set refresh), +0.1/move survived, −50 on game over, per-step reward
  clipped to ±25. Delta components telescope over an episode, so they
  cannot be farmed by oscillation. Board-quality metrics (mean holes,
  regions, occupancy, future legal moves, future-moves delta) are logged
  per episode under `game/holes`, `game/empty_regions`,
  `game/occupancy_fraction`, `game/future_legal_moves`,
  `game/future_moves_delta`.

`--observation-profile` (`environment/observations.py`):

* `basic` (default) — 4 channels: board + 3 piece masks.
* `enhanced` — 8 channels: basic + normalized column heights, holes mask,
  line-completion potential, legal-placement density. All cheap, spatial
  and Markovian.

## Experiments, behavior cloning and HPO

```bash
# controlled experiment matrix (A baseline, B entropy, C lr, D reward, E obs, F BC-init)
python -m training.experiments --timesteps 2000000
# expert dataset from the heuristic agent + behavior cloning + PPO init
python -m training.generate_expert_data --games 1000
python -m training.behavior_cloning --epochs 10 --eval-games 50
python -m training.train --init-from models/bc_init.zip --timesteps 5000000
# optional Optuna hyperparameter search (pip install optuna)
python -m training.optimize --trials 20 --timesteps 1000000
```

Each experiment gets `results/experiments/<name>/` with `config.json`
(full config + versions + GPU + git commit), the model, logs, TensorBoard
events, eval results and plots; `summary.csv`/`summary.png` compare all of
them against Random/Heuristic on identical seeds. An experiment only
counts as an improvement if its **evaluation score** beats the baseline
experiment — training reward alone proves nothing, and different reward
profiles are never comparable by reward.

The expert dataset (`results/expert_dataset.npz`) stores observations,
action masks and expert actions; `training/behavior_cloning.py` trains the
policy with masked cross-entropy and saves a PPO-initialization model.
Whether BC init actually helps evaluation score is tested as experiment F.

## Imitation learning (demonstrations + behavioral cloning)

A complete system for learning from recorded gameplay — human or
autonomous. The pipeline:

```
human play / autonomous agents  ->  demonstration dataset  ->  behavioral
cloning (masked cross-entropy)  ->  ImitationAgent  ->  optional PPO init
```

### Collecting demonstrations

```bash
# human: play normally; every move is recorded before it changes the state
python main.py --mode record
# human feedback: the AI suggests a move — A accepts (source human_accepted_ai),
# placing your own move overrides it (source human, suggestion kept as metadata)
python main.py --mode feedback --agent heuristic
# autonomous, headless, thousands of games (any agent spec)
python -m training.collect_demos --agent heuristic --games 1000
python -m training.collect_demos --agent rl:models/main_final.zip --games 200
python -m training.collect_demos --agent solver --games 50 --max-moves 5000
```

Record mode shows `Game / Steps / Dataset samples / Score`; `R` starts a
new game without restarting. Only valid actions are recorded. The
collector flushes completed games to disk every 25 games, so memory stays
flat; game ids continue across runs.

### Dataset format

Shards `results/demos/shard_*.npz` + `meta.json`. Per sample: bit-packed
observation `(C,8,8)` (state BEFORE the action), action id (0-191),
bit-packed action mask, reward (game points), score before, lines cleared,
game-over flag, source code (`human`, `heuristic`, `ppo`, `solver`,
`random`, `human_accepted_ai`, `imitation`), game id, step, suggested
action. Inspect it (sizes, per-source stats, action distribution — the
bias check):

```bash
python -m training.dataset_stats
```

### Training the imitation policy

```bash
python -m training.train_imitation --source human
python -m training.train_imitation --source heuristic --epochs 20
python -m training.train_imitation --source all --weight-human 5.0 --augment
```

Masked cross-entropy: illegal actions get `-inf` logits before the loss,
so they never receive probability mass. Per-source sample weights keep a
small human dataset from being overwhelmed (default: human 5x, others 1x).
`--augment` adds 4x symmetry copies (horizontal/vertical flip, 180°
rotation; basic observation profile only — enhanced channels are not
mirror-symmetric). Metrics per epoch: train/val loss, accuracy, masked
accuracy, top-3, top-5; the split is BY GAME to prevent leakage. Best
checkpoint goes to `models/imitation/bc_<source>.pt`.

### Using and evaluating the imitation agent

```bash
python main.py --mode watch --agent imitation:models/imitation/bc_all.pt
python -m training.evaluate_imitation --model models/imitation/bc_all.pt --games 200
```

`evaluate_imitation` compares Random / Heuristic / Imitation (deterministic
argmax and, with `--stochastic`, the sampling variant) on identical seeds.
High training accuracy does not imply strong play — the evaluation score
is the metric that matters.

### Initializing PPO from imitation

The BC network mirrors the PPO policy exactly (same CNN, trunk and action
head, identical state-dict keys), so a `.pt` checkpoint initializes PPO
directly — the value head starts random and is learned by PPO:

```bash
python -m training.train --init-from models/imitation/bc_all.pt --timesteps 10000000
# or export a .zip first (equivalent):
python -m training.train_imitation --source all --export-ppo models/imitation/bc_ppo.zip
python -m training.train --init-from models/imitation/bc_ppo.zip --timesteps 10000000
```

This is an optional initialization path; it does not replace the existing
PPO pipeline or models.

## Performance

Environment and training-loop profiling:

```bash
python -m training.profile_performance
```

Measured on the RTX 4070 SUPER machine: valid-action caching in the
engine (the mask was computed 3-4x per step before), a vectorized enhanced
observation (508 -> 105 µs/call), and 16 vectorized envs raised training
throughput from ~2,870 to ~4,260 steps/sec. The remaining bottleneck is
PPO's GPU update + mask-fetch IPC, not the game simulation
(env.step alone runs >100k/s per process).

## Project structure

```
├── game/
│   ├── game.py           # BlockBlastGame engine, GameConfig, MoveResult (cached valid actions)
│   ├── pieces.py         # the 34 piece shapes, rotations, Piece dataclass
│   └── utils.py          # board analysis (holes, regions, heights, ...)
├── environment/
│   ├── block_blast_env.py# Gymnasium env, per-component reward tracking
│   ├── rewards.py        # RewardConfig + baseline/survival/lines/strategic profiles
│   └── observations.py   # basic (4ch) / enhanced (8ch) observation profiles
├── agents/
│   ├── base.py           # Agent interface + DecisionInfo (debug panel)
│   ├── random_agent.py
│   ├── heuristic_agent.py
│   ├── solver_agent.py   # beam-search planner (near-perfect, no learning)
│   ├── imitation_agent.py# behavioral-cloning policy (det./stochastic)
│   └── rl_agent.py       # MaskablePPO wrapper (act / value / probabilities)
├── training/
│   ├── cnn_extractor.py  # board CNN for the Cx8x8 observation
│   ├── config.py         # TrainConfig: every hyperparameter in one place
│   ├── train.py          # PPO training (CLI, checkpoints, resume, init-from)
│   ├── metrics.py        # episode + optimizer metrics callbacks (CSV + TB)
│   ├── dashboard.py      # live training dashboard (separate process)
│   ├── visualize_network.py # architecture / weights / activations / policy maps
│   ├── evaluate.py       # benchmarking, percentiles, --compare plots
│   ├── experiments.py    # controlled experiment matrix + summary
│   ├── optimize.py       # optional Optuna HPO
│   ├── generate_expert_data.py # heuristic -> expert dataset (.npz)
│   ├── behavior_cloning.py     # masked cross-entropy imitation
│   ├── demos.py            # DemoRecorder/DemoDataset: shard-based demo storage
│   ├── collect_demos.py    # headless autonomous demo collection
│   ├── imitation_model.py  # BC network (PPO-compatible keys) + conversion
│   ├── train_imitation.py  # behavioral cloning CLI (masked CE, source weights)
│   ├── augment.py          # symmetry augmentation (basic profile only)
│   ├── dataset_stats.py    # dataset inspection / bias report
│   ├── evaluate_imitation.py # imitation vs Random/Heuristic/PPO comparison
│   ├── profile_performance.py  # env / NN / VecEnv profiler
│   ├── repro.py          # reproducibility metadata -> config.json
│   └── plot_results.py   # reward/score learning curves
├── ui/
│   ├── pygame_app.py     # human mode, AI watch mode, debug panel
│   ├── brain_app.py      # AI Brain mode (live policy heatmaps)
│   └── record_app.py     # record mode + human-feedback mode (demo collection)
├── models/  results/     # training outputs (created on demand)
├── tests/                # pytest: engine, env, agents, generator, imitation (117 tests)
├── main.py               # launcher: human / watch / console / brain
└── requirements.txt
```

## Running the tests

```bash
python -m pytest tests/ -q
```

Covers placement, invalid placement, boundaries, row/column/simultaneous
clearing, scoring (incl. combo bonus), piece regeneration, game-over
detection, valid-action generation, seeded determinism, env reset/step
contract, action-mask consistency, invalid-action handling, deterministic
seeding, and agent behavior (random validity, heuristic line-clear
preference, heuristic-beats-random, RL smoke test).

## GPU requirements

Training benefits from CUDA but is not required — PPO on an 8x8 board also
trains on CPU (slower). The training script auto-detects CUDA and prints
the device. Any recent NVIDIA GPU works; the code was developed on an
RTX 4070 SUPER with torch 2.7.1+cu128.

## Board-aware piece generation

The original generator drew 3 fully random pieces regardless of board
state, so on nearly-full boards it could deal pieces with no legal
placement and force an unfair game over. The generator in
`game/game.py` is now board-aware and is the single source of truth for
human play, all agents, the RL environment and training:

* **Level 1 (hard guarantee):** every generated piece has at least one
  legal placement (`get_playable_piece_types`, rejection sampling with
  `MAX_GENERATION_ATTEMPTS = 100`, then a uniform fallback among playable
  types). If NO type fits, the board is genuinely dead — nothing is
  generated and the game is over.
* **Level 2 (best-effort preference):** sets are checked for sequential
  viability (`is_piece_set_playable` — DFS over all 6 orderings with line
  clears simulated, node-budgeted) and regenerated up to
  `MAX_SET_ATTEMPTS = 50` times. If no viable set is found, the last
  Level-1-valid set is kept (counted in `generation_stats`).
* **Randomness is preserved:** pieces are rejection-sampled from the
  ORIGINAL uniform distribution; the generator never picks "easy" pieces.
  Stress test: `python -m training.test_generator --games 1000`.
* Game-over semantics are unchanged: game over = no legal placement for
  ANY remaining piece (never "one piece doesn't fit").

**This changes the environment distribution.** Scores measured before
and after the fix are NOT comparable:

| Environment | Random | Heuristic | PPO 5M |
|---|---|---|---|
| OLD (unconstrained generator) | 68.5 | 705.3 | 141.9 |
| NEW (board-aware generator) | 115.9 | 1737.0 | *(not retrained yet)* |

The pre-fix model is preserved as `models/main_pre_generator_fix.zip`
(100 eval games each, seeds 1000+; impossible sets were ending games
early, which hurt long games the most).

## Known limitations

* **Curriculum learning — investigated, deliberately not implemented.**
  The only clean lever available without changing the rules is the piece
  distribution (e.g. train on smaller/simpler pieces first). That changes
  the transition distribution the policy learns from: a policy optimized
  for easy pieces is not optimizing the real game, and the gains tend to
  evaporate exactly where the real game gets hard. The empirical evidence
  here also points elsewhere: behavior cloning already bootstraps the
  policy past the weak early phase *without* touching the game
  distribution, and all final evaluation always uses the real
  distribution. If curriculum is revisited, it should be done as an
  auxiliary reward/curriculum on starting boards, never on piece
  frequencies, and re-validated on the real distribution.
* PPO needs millions of steps to reach heuristic-level play; the included
  5M-step run is a starting point, not a ceiling. Longer runs
  (`--timesteps 20000000`+) and/or reward tuning in `RewardConfig` help.
* The piece channels are drawn at a fixed origin; the CNN must learn to
  relate shape to board position. A relative-placement encoding could
  learn faster (future work).
* `SubprocVecEnv` on Windows uses process spawning — always launch
  training via `python -m training.train`, not from a REPL.
* The UI's AI watch mode shows the critic value estimate only for the RL
  agent; the heuristic shows its internal heuristic score instead.
