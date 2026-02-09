# H-MAPPO Stability Review (post-fix)

## Scope
This document reviews the current two-head H-MAPPO implementation in BenchMARL
and its execution in bsk_rl, with a focus on convergence and stability. It
compares the current design to HOP-PPO (Selector S) and recommends changes that
preserve the two-head structure while improving stability.

## Current Architecture (as implemented)
- Head 1 (Split): continuous vector of task split ratios.
- Head 2 (Routing): continuous resource params + routing logits (high/low).
- Conditioner: FiLM based on Head 1 output (loc + log_std).
- Execution: bsk_rl applies shielding and normalization before acting.
- Routing: environment masks unreachable neighbors and renormalizes.

Key files:
- benchmarl/models/hierarchical_actor.py
- benchmarl/models/conditional_dual_head_actor.py
- bsk_rl/act/stin_hier_hybrid_actions.py
- bsk_rl/sats/hier_computation_satellite.py

## Why this is not equivalent to HOP-PPO Selector S
HOP-PPO uses a discrete head to select a single task index, then a hard selector
extracts task-specific features for the continuous head. This reduces the
continuous head to a smaller, task-conditioned problem. In the current H-MAPPO
code, Head 2 sees global features (FiLM conditioned) rather than a hard-selected
task subset. That means Head 2 is still solving a high-variance global mapping.
Therefore the core stability benefit from Selector S is not present.

## Primary instability drivers (current code)
1) Level-1 simplex constraint is applied in the environment:
   - stin_hier_hybrid_actions.py normalizes Level-1 ratios after the policy
     samples them.
   - This creates log_prob mismatch (policy vs executed action).

2) Action Shield modifies actions post-sampling:
   - Shielding edits actions but log_prob is still computed from the original.

3) Routing mask/renorm happens in the environment:
   - Masking and renormalization are applied after the policy outputs routing
     probabilities, creating distribution drift.

4) Hard gumbel routing:
   - Gumbel-Softmax hard=True makes routing nearly discrete and can amplify
     gradient noise.

These factors together can cause large KL/clip_fraction spikes and unstable
training even if the algorithm (PPO) is correct.

## What you can and cannot get without a Selector S
Without task-level hard selection, you cannot fully reproduce HOP-PPO stability.
You can still improve stability significantly by aligning the policy output
distribution with the executed actions and reducing action post-processing.

## Recommended optimization plan (two-head preserved)
### Phase 1: Make policy outputs valid by construction
- Move Level-1 simplex constraint into policy:
  - Prefer Logit-Normal (Normal + StickBreaking/Softmax) for Head 1; avoid Dirichlet unless alphas are kept > 1.
- Ensure sum=1 and non-negative before actions reach the environment.
- Environment should only correct invalid actions (rare):
  - Keep normalization as a strict fallback only.

### Phase 2: Align routing distribution with execution
- Apply neighbor mask inside policy (masked softmax or masked Dirichlet).
- If masking remains in environment, record and monitor correction frequency.
- Prefer routing_mode="flow" for stability runs to reduce sampling variance.

### Phase 2.5: Soft Selector via attention (routing-only)
- Build a routing attention block over [self + neighbors] features.
- Use Head 1 (loc) as the query to modulate routing context.
- Feed the attention context into Head 2 (MLP) to reduce variance without hard selection.
- Keep Head 1 on full observation; limit Head 2 attention to self + neighbors only.

### Phase 3: Optional Selector-like simplification (if needed)
- Add a discrete task index head and a task-specific feature selector.
- Requires observation reformatting to provide task-level feature blocks.
- This is the only way to match HOP-PPO Selector S behavior.

## Quick stability checks
Track the following during training:
- KL and clip_fraction trends (should stabilize or drop).
- Action correction counters (shield/normalize). Should be near zero.
- Log_prob vs executed action consistency (sanity check in debug runs).

## Implementation map (where changes would land)
- benchmarl/models/hierarchical_actor.py
  - Replace Level-1 distribution with Dirichlet/Softmax.
  - Optionally mask routing logits before Categorical.
- bsk_rl/act/stin_hier_hybrid_actions.py
  - Only normalize Level-1 when invalid (epsilon threshold).
  - Track correction stats.
- bsk_rl/sats/hier_computation_satellite.py
  - Keep routing_mode=flow for stable runs.
  - Optional: gate mask to policy side.

## Expected outcome
After Phase 1+2, the policy and environment distributions will match, which
should reduce variance and allow PPO to converge. This will not fully match
HOP-PPO Selector S stability, but should materially improve training quality.
