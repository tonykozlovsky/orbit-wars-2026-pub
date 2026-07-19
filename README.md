# Orbit Wars — IMPALA + Behavior Cloning

This repository contains my solution for the Kaggle **Orbit Wars** competition.

The main idea was to avoid learning the game from scratch. I first trained the policy with behavior cloning on replays from a top player, then fine-tuned it with reinforcement learning using the final game result as the reward. During RL, the agent played against both its current policy and a pool of frozen historical opponents. A moving, delayed copy of the policy was also used as a teacher.

The final models were trained for approximately one week on a single NVIDIA RTX 5090.

## Solution overview

The training pipeline had two main stages:

1. **Behavior cloning**
   - Parse Kaggle replays into model observations and legal actions.
   - Train the policy to imitate actions from a strong player.
   - Use masked cross-entropy so only valid planets and legal action classes contribute to the loss.
   - Use the resulting checkpoint as a strong initialization for reinforcement learning.

2. **Reinforcement learning**
   - Continue training with an asynchronous IMPALA-style actor–learner setup.
   - Use V-trace to correct for the policy lag between rollout actors and the learner.
   - Optimize primarily for the sparse terminal result: win or loss.
   - Mix self-play with games against frozen checkpoints from different stages of training.
   - Regularize the current policy toward a delayed moving teacher with a KL loss.
   - Train on both two-player and four-player games.

In short, behavior cloning taught the model how to play, while RL, historical opponents, and the delayed teacher made it stronger and more stable.

## Why behavior cloning first?

Orbit Wars has a large structured action space and long-term consequences. Pure self-play from a random policy spends a large amount of compute discovering basic behavior: when to expand, how many ships to send, and which attacks are legal or useful.

Behavior cloning provides these fundamentals immediately. The replay pipeline reconstructs the observation before every action and converts the original move into the same per-planet action classes used by the RL policy. This made the transition from supervised learning to RL direct: both stages train the same model and action representation.

The BC implementation is in:

- `python/supervised_learning/behavior_cloning.py`
- `python/src/gym/orbit_kaggle_replay_bc_dataset.py`
- `python/tools/analyze_kaggle_orbit_replays.py`

## RL training

The RL system is based on the IMPALA actor–learner design:

- many CPU actor processes run game simulations;
- inference requests are batched before being sent to the model;
- actors write short rollout sequences into shared buffers;
- a GPU learner consumes these rollouts and updates the policy;
- updated weights are periodically copied back to inference workers;
- V-trace corrects for the fact that actors may have used a slightly older policy.

The main RL objective combines:

- V-trace policy-gradient loss;
- a value-function loss;
- entropy regularization;
- teacher KL regularization.

The reward used for the final policy is intentionally sparse. Intermediate fleet, planet, and production rewards are disabled; the baseline reward is determined by the final game result. This reduces the risk of optimizing a hand-designed proxy instead of winning the game.

Representative final-stage settings included short rollouts, BF16 model forward passes, FP32 policy/value heads, a low learning rate, and a target of hundreds of millions of environment steps. The exact experiment variants are preserved in `python/src/configs/`.

Important RL files:

- `python/run_monobeast.py` — training entry point
- `python/src/torchbeast/monobeast.py` — process and worker orchestration
- `python/src/torchbeast/core/act.py` — rollout actors and opponent selection
- `python/src/torchbeast/core/inference_worker.py` — batched policy inference
- `python/src/torchbeast/core/learn.py` — learner update
- `python/src/torchbeast/core/losses_func_selfplay.py` — V-trace, value, entropy, and teacher losses
- `python/src/gym/reward_wrapper.py` — terminal reward construction

## Opponent curriculum

Naive self-play has a common failure mode: the current policy only learns to exploit its latest version and may forget how to beat older strategies.

To reduce this problem, rollout games use a mixture of:

- the live policy;
- frozen checkpoints collected throughout training;
- strong checkpoints specialized for two-player or four-player games.

The checkpoint pool acts as a simple league. It exposes the learner to opponents with different strengths and styles, improves robustness, and makes regressions easier to detect.

## Delayed moving teacher

The learner is additionally regularized against a teacher policy through KL divergence.

The teacher is not updated on every learner step. Instead, it follows the learner with a fixed delay by loading an older checkpoint from the recent training history. In the final experiments the delay was on the order of hundreds of thousands of environment steps.

This has two useful effects:

- policy updates remain anchored to a known playable strategy;
- the learner can improve through RL without drifting too abruptly or catastrophically forgetting behavior learned earlier.

Unlike a permanently frozen teacher, the moving teacher gradually improves together with the student.

## Model architecture

The policy treats the game state as a structured set of planets and directed planet-to-planet edges.

The observation contains:

- per-planet state, ownership, production, position, and motion features;
- projected fleet arrivals over a temporal horizon;
- pairwise source/destination features such as distance and tactical margins;
- masks for existing planets, valid edges, active players, and legal actions;
- relative player identity features.

Continuous features are normalized, while selected discrete quantities use learned embeddings. Planet, arrival, and edge inputs are encoded separately and fused into a shared hidden representation.

The central network is a stack of planet/edge cross-attention blocks. The released configuration uses:

- hidden size 128;
- 16 transformer-style blocks;
- 4 attention heads;
- separate arrival-history attention;
- no residual or feed-forward dropout.

The policy head predicts an action independently for each source planet. An action class represents a destination together with one of several send-amount buckets. Illegal actions are masked before sampling. The value head pools the structured planet representation into a global estimate of the game outcome.

Model and feature definitions:

- `python/src/models/models.py`
- `python/src/configs/impala_orbit_model_hyperparams.py`
- `python/src/gym/obs_wrapper.py`

## Engineering details

Training throughput was important because the environment and the structured model are both relatively expensive.

The code includes:

- a C++ implementation of the Orbit Wars simulation and observation path;
- asynchronous CPU rollout actors;
- fixed-size batched GPU inference;
- shared-memory rollout buffers;
- BF16 inference and learner forward passes;
- `torch.compile` for training and inference workers;
- strict tensor-shape and action-mask assertions;
- checkpoint benchmarking for two-player and four-player games;
- TensorBoard and optional Weights & Biases logging.

The C++ environment can also be checked against the reference Python implementation. This was useful while optimizing the simulator: a faster environment is only useful if it produces exactly the same game transitions and model inputs.

## Submission inference

The Kaggle submission uses separate packaged policy artifacts for two-player and four-player games. The model is exported for CPU inference with PyTorch AOTInductor and called through a small C++ runner.

At every step, the submission:

1. converts the Kaggle observation into the internal plain representation;
2. updates the cached C++ game state;
3. builds features only for the active player;
4. runs the corresponding 2P or 4P policy artifact;
5. applies the legal-action mask and converts selected classes back into Kaggle moves.

The submission implementation is in `python/kaggle_submission/submission.py`.

## What mattered most

The most important parts of the solution were:

- starting from a strong behavior-cloned policy instead of random self-play;
- using the actual win/loss result rather than relying on dense reward shaping;
- maintaining diversity through a pool of frozen historical opponents;
- stabilizing RL with a delayed moving teacher;
- representing planets, arrivals, and directed edges explicitly;
- making simulation and inference fast enough to collect a large number of games.

The final result was not one isolated trick. It came from combining a good supervised initialization, stable off-policy RL, opponent diversity, a structured model, and an optimized training pipeline.

## Repository note

This repository is a research snapshot of the competition solution. Experiment configs contain checkpoint paths from the original training machine, so those paths must be replaced with local checkpoints before reproducing a run. The configs are included to document the actual experiments and hyperparameters rather than to present a simplified one-command training package.

## License

See [LICENSE](LICENSE).
