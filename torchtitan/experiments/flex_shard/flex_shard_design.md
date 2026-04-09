# FlexShard: One API Across Eager and Compile

**Authors**: Wei Feng, Tianyu Liu, Ailing Zhang

## Front-End API

### `flex_shard()` — the single entry point

```python
flex_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    shard_placement_fn: PlacementFn | dict[str, Placement] | None = None,
    *,
    buckets: list[list[str] | BucketSpec] | None = None,
)
```

One call shards a module. The same function works for eager, JIT, and AOT — no mode-specific API.

<!-- TODO: Add multi-mesh API example showing the case where the flex_shard mesh
     differs from the computation mesh, e.g. (dp_replicate, fsdp_cp) for flex_shard
     but (dp, cp) for computation. The implementation already handles this via
     DTensorAwareParametrization (Phase 5d), but the doc needs a user-facing example. -->

```python
mesh = init_device_mesh("cuda", (world_size,))
model = Transformer(args)

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
buckets = [
    BucketSpec(["tok_embeddings.*"], mp_policy=mp),
    *[BucketSpec([f"layers.{i}.*"], mp_policy=mp, offload_policy=OffloadPolicy()) for i in range(n_layers)],
    BucketSpec(["norm.*", "output.*"], mp_policy=mp),
]

# Shard(0): per-param all-gather, one bucket per transformer block (FSDP2-style)
# per_param_placements is a built-in factory:
#
#   def per_param_placements(named_params, mesh):
#       return {fqn: (Shard(0),) for fqn, _ in named_params}
#
# {"*": Shard(0)} is shorthand — flex_shard resolves it via fnmatch to the same result.
flex_shard(model, mesh, shard_placement_fn=per_param_placements, buckets=buckets)

# FlatShard: flatten params per bucket into one 1D buffer, single all-gather per bucket (FSDP1-style)
# flat_shard_placements is a built-in factory that computes contiguous flat offsets:
#
#   def flat_shard_placements(named_params, mesh):
#       total = sum(p.numel() for _, p in named_params)
#       result, offset = {}, 0
#       for fqn, p in named_params:
#           result[fqn] = (FlatShard(offset, p.numel(), total),)
#           offset += p.numel()
#       return result
#
# Example for a bucket with weight (256x768) + bias (768):
#   weight → FlatShard(flat_offset=0,      numel=196608, total=197376)
#   bias   → FlatShard(flat_offset=196608, numel=768,    total=197376)
# Each rank stores total//world_size elements of the concatenated flat buffer.
flex_shard(model, mesh, shard_placement_fn=flat_shard_placements, buckets=buckets)

# Owned: each param lives fully on one rank, broadcast to all (veScale-style)
# param_boundary_placements is a built-in factory that uses greedy bin-packing:
#
#   def param_boundary_placements(named_params, mesh):
#       assignments = _assign_params_to_ranks(named_params, mesh.size())
#       return {fqn: (Owned(assignments[fqn]),) for fqn, _ in named_params}
#
# _assign_params_to_ranks greedily assigns each param to the least-loaded rank.
# Forward: broadcast from owner. Backward: all-reduce to average gradients.
flex_shard(model, mesh, shard_placement_fn=param_boundary_placements, buckets=buckets)
```

## Core Insight

FlexShard intercepts parameter access so that `module.weight` triggers an all-gather behind the scenes — the model code just reads `self.weight` as usual, unaware of sharding. This works identically across eager, JIT, and AOT. Under the hood, the interceptor (a `@property` on the module class) has two branches:

- **Compiled modes (JIT/AOT)**: falls through to `parametrization.forward()`, which emits `_c10d_functional` ops (per-param, async, FX-traceable). `torch.compile` traces these into the FX graph for compiler passes to optimize.
- **Eager mode**: short-circuits via `_pre_gathered` — batched hooks run `dist.*` collectives (per-bucket, synchronous) before the property getter fires, so `parametrization.forward()` never runs.

The parametrization classes (`ShardParametrization`, `FlatShardParametrization`, etc.) define the communication pattern once. Compiled modes execute them directly; eager mode uses them as the reference implementation but replaces the execution with batched collectives for performance (one NCCL kernel per bucket instead of N per-param kernels).

## Eager vs Compiled: Where Collectives Live

The key architectural difference between eager and compiled modes is where collectives sit relative to the autograd graph:

### Eager: collectives are OUTSIDE autograd, checkpoint controls lifetime

```
  ╔══ checkpoint_wrapper (per layer) ═══════════════════════════════════╗
  ║                                                                     ║
  ║  ┌─ outside autograd (torch.no_grad) ─────────────────────────┐    ║
  ║  │                                                             │    ║
  ║  │  pre_forward_hook:                                          │    ║
  ║  │    Placement.unshard()  ──→  dist.all_gather()  ──→ full_W  │    ║
  ║  │                                                             │    ║
  ║  └─────────────────────────────────────────────────────────────┘    ║
  ║                                                         │           ║
  ║                                             detach().requires_grad_ ║
  ║                                                         │           ║
  ║                                                         ▼           ║
  ║  ┌─ inside autograd ──────────────────────────────────────────┐    ║
  ║  │                                                             │    ║
  ║  │  full_W (leaf)  ──→  F.linear(input, full_W)  ──→  output  │    ║
  ║  │                                                             │    ║
  ║  └─────────────────────────────────────────────────────────────┘    ║
  ║                                                                     ║
  ╚═════════════════════════════════════════════════════════════════════╝
          │                                                 │
          │ after layer forward:                            │
          │ checkpoint replaces full_W                      │
          │ with _Holder ──→ full_W freed by GC             │
          │                                                 │
          ▼                                                 ▼
  ┌─ backward ────────────────────────────────────────────────────────┐
  │                                                                    │
  │  checkpoint unpack_hook fires ──→ re-runs layer forward:           │
  │    pre_forward_hook fires again ──→ new all-gather ──→ new full_W  │
  │    autograd computes grad on new full_W                            │
  │    AccumulateGrad hook fires ──→ _reduce_fn:                       │
  │      Placement.reduce_grad() ──→ dist.reduce_scatter()             │
  │      ──→ write to sharded_param.grad                               │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
```

Autograd sees `full_W` as an ordinary leaf tensor — no knowledge of all-gather or reduce-scatter. Checkpoint controls `full_W`'s lifetime: discards it after each layer's forward, triggers re-all-gather during backward recomputation. This achieves per-layer memory release without autograd being aware of FSDP.

### JIT / AOT: collectives are INSIDE autograd

```
  ┌─ inside autograd ───────────────────────────────────────────┐
  │                                                              │
  │  sharded_W (leaf Parameter)                                  │
  │       │                                                      │
  │       ▼                                                      │
  │  _c10d_functional.all_gather_into_tensor(sharded_W)          │
  │       │                                                      │
  │       ▼                                                      │
  │  _c10d_functional.wait_tensor(full_W)                        │
  │       │                                                      │
  │       ▼                                                      │
  │  F.linear(input, full_W)  ──→  output                       │
  │                                                              │
  │  backward auto-generates:                                    │
  │    reduce_scatter_tensor(grad)  ──→  sharded_W.grad          │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

Autograd sees the all-gather as a differentiable op. The graph pass marks it `MUST_RECOMPUTE` so the compiler frees `full_W` after forward and re-all-gathers in backward.

## Parametrization: The Shared Front End

### How it works

`_register_parametrization()` creates a dynamic subclass of each leaf module's class with property descriptors. When `module.weight` is accessed, the property getter checks for a batched all-gather result first, falling through to the per-param parametrization if none:

```python
# Each layer is wrapped: checkpoint_wrapper(layer, context_fn=reshard_policy)
# checkpoint discards full_W after forward, re-creates via recomputation in backward

class FlexShardLinear_1(nn.Linear):
    @property
    def weight(self):
        # Eager only: _pre_gathered set by pre_forward_hook
        # (hook skips under torch.compiler.is_compiling(), so pre is always None in JIT/AOT)
        pre = parametrization._pre_gathered
        if pre is not None:
            # detach(): disconnect from batched AG (outside autograd graph)
            # requires_grad_(True): make it a leaf so autograd computes grad,
            #   which AccumulateGrad hooks capture for batched reduce-scatter
            # checkpoint treats this leaf like any other tensor — discards it
            #   after layer forward, re-creates it via re-all-gather in backward
            return pre.detach().requires_grad_(True)
        # JIT/AOT: per-param all-gather via _c10d_functional (traced into FX graph)
        return parametrization(self._parameters['weight'])
```

`state_dict()` reads `self._parameters` directly (bypasses the property), so checkpoints always see sharded tensors.

### Placement-specific parametrizations

| Class | Forward collective | Backward | Post-processing |
|-------|-------------------|----------|-----------------|
| `ShardParametrization` | `_c10d_functional.all_gather_into_tensor` | autograd `reduce_scatter_tensor` | chunk+cat (dim!=0), narrow (uneven) |
| `FlatShardParametrization` | `_c10d_functional.all_gather_into_tensor` | autograd `reduce_scatter_tensor` | view to original shape |
| `OwnedParametrization` | `_c10d_functional.broadcast` via `_OwnedBroadcast` | custom `all_reduce` / world_size | none |
| `RaggedShardParametrization` | `_c10d_functional.all_gather_into_tensor` | autograd `reduce_scatter_tensor` | chunk+narrow+cat per rank |

These are functional collectives (`torch.ops._c10d_functional.*`) — async, FX-traceable, with autograd support. The parametrization `forward()` containing these ops is the code that `torch.compile` traces into the FX graph.

**However, eager mode bypasses these entirely.** The property getter short-circuits before calling `parametrization.forward()`:
- The pre_forward_hook calls `Placement.unshard()` → `dist.all_gather()` (synchronous, per-bucket batched)
- The property getter returns the pre-gathered result as a detached leaf — `parametrization.forward()` never runs
- `AccumulateGrad` hooks call `Placement.reduce_grad()` → `dist.reduce_scatter_tensor()` (synchronous, per-bucket batched)

So in practice: **JIT/AOT use `_c10d_functional` (per-param, async). Eager uses `dist.*` (per-bucket, synchronous).**

## Reshard-After-Forward

Goal: **free unsharded parameter memory after each layer's forward, recompute in backward**.

### Eager mode

In eager mode, collectives are **outside** the autograd graph. Three mechanisms work together:

**Forward path** (per layer):

1. **Pre-forward hook** (`_install_batched_allgather_hooks`): calls `Placement.unshard()` — one batched `dist.all_gather` per bucket. Stashes the result on `_pre_gathered`.
2. **Property getter**: sees `_pre_gathered` is set, returns `pre.detach().requires_grad_(True)` — a detached leaf. The per-param `parametrization.forward()` (which calls `_c10d_functional` ops) is **never called** in eager.
3. **Post-forward hook**: registers `AccumulateGrad` hooks on each detached leaf for gradient capture.

**Reshard** (between layers):

Each layer is wrapped in `checkpoint_wrapper` with `_flex_shard_reshard_policy` (`_apply_reshard_checkpoint`). The policy marks collective ops as `MUST_RECOMPUTE`, everything else as `PREFER_RECOMPUTE`. Checkpoint discards all intermediates after each layer's forward — including the detached-leaf unsharded params. No explicit reshard call is needed; checkpoint's discard-and-recompute handles it.

If activation checkpointing (AC) is also applied, the two policies are merged into a single `checkpoint_wrapper`: FlexShard collectives → `MUST_RECOMPUTE`, AC compute ops → `MUST_SAVE`, everything else → `PREFER_RECOMPUTE`.

**Backward path** (per layer):

1. Checkpoint re-runs the layer's forward: the pre-forward hook fires again, doing another batched all-gather.
2. Autograd computes grad on the recomputed detached leaf.
3. `AccumulateGrad` hooks fire → `_reduce_fn` calls `Placement.reduce_grad()` — one batched `dist.reduce_scatter_tensor` per bucket, writing to `sharded_param.grad`.

**Key checkpoint interaction**: `_unsharded_for_reduce` is only stored on the FIRST forward (not during checkpoint recomputation). The original leaf is what checkpoint saves and autograd accumulates grad on. The recomputed leaf is a different object used for the recomputed backward graph.

### JIT / AOT (compiled modes)

In compiled modes, collectives are **inside** the autograd graph. The mechanism is entirely different:

**Forward path** (per layer):

1. **Property getter**: `_pre_gathered` is always `None` (batched hooks are disabled under `torch.compiler.is_compiling()`).
2. **`parametrization.forward()` runs**: calls `_c10d_functional.all_gather_into_tensor` / `broadcast` / `wait_tensor` per param. These are functional collectives that `torch.compile` traces into FX `call_function` nodes.
3. No pre/post-forward hooks run — the compiler sees the full op graph.

**Reshard** (between layers):

A **graph pass** (`flex_shard_reshard_after_fwd_pass`) annotates unshard node sequences with `CheckpointPolicy.MUST_RECOMPUTE`. The Inductor min-cut partitioner uses these annotations to free unsharded params after forward and recompute in backward. No `checkpoint_wrapper` is used.

- JIT: pass registered via `functorch_config` joint graph passes.
- AOT: same pass, applied through `get_joint_custom_passes_from_config()`, registered as `"flex_shard_reshard_after_fwd"` in `AVAILABLE_JOINT_PASSES`.

**Backward path** (per layer):

Autograd sees the all-gather as a differentiable op. The backward of `_c10d_functional.all_gather_into_tensor` auto-generates `reduce_scatter_tensor` — per-param, not batched. The compiler rebatches these via bucketing passes.

### Summary

| | Eager | JIT / AOT |
|---|---|---|
| **All-gather** | `dist.all_gather` (batched per bucket, outside autograd) | `_c10d_functional.all_gather_into_tensor` (per param, inside autograd) |
| **Reduce-scatter** | `dist.reduce_scatter_tensor` (batched per bucket, AccumulateGrad hooks) | `_c10d_functional.reduce_scatter_tensor` (per param, autograd-generated) |
| **Reshard mechanism** | `checkpoint_wrapper` + selective policy | Graph pass + min-cut partitioner |
| **Parametrization.forward()** | Skipped (short-circuited by `_pre_gathered`) | Runs (traced into FX graph) |
| **AC composition** | Merged policy in single `checkpoint_wrapper` | Separate graph pass (naturally composable) |

## Graph Pass Pattern Recognition

The reshard pass recognizes these FX node sequences (used in compiled modes):

```
Shard(0):           placeholder → [_to_copy] → all_gather → wait_tensor
                    → [narrow] → [convert_element_type]

Shard(dim!=0):      placeholder → [_to_copy] → all_gather → wait_tensor
                    → chunk → getitem(0..N) → cat → [narrow]
                    → [convert_element_type]

FlatShard:          placeholder → [_to_copy] → all_gather → wait_tensor
                    → view → [convert_element_type]

Owned:              placeholder → [_to_copy] → broadcast → wait_tensor
                    → [convert_element_type]
```

## Design Decisions

### Why parametrization over hooks?

Parametrization emits `_c10d_functional` ops into the graph, making communication visible to the compiler for reshard annotation, communication scheduling, and overlap optimization. Hooks are opaque to FX tracing.

### Why batched collectives in eager?

Per-param `_c10d_functional` ops emit one NCCL kernel per param (57 AllGathers for a 6-layer model). Batching via `Placement.unshard()` / `reduce_grad()` emits one NCCL kernel per bucket (8 AllGathers + 8 ReduceScatters). In compiled modes, the compiler rebatches per-param ops automatically via bucketing passes.

### Why `AccumulateGrad` hooks for reduce-scatter?

`register_hook` on detached leaf params fires when autograd computes their grad — guaranteed correct timing. Previous approaches (`queue_callback`, `_RegisterPostBackward` on inputs) failed because:
- `queue_callback` fires after the ENTIRE backward, losing per-layer timing
- `_RegisterPostBackward` on inputs doesn't work for modules with no-grad inputs (e.g., Embedding receives Long indices)
- `register_hook` on outputs fires BEFORE param grads are computed

`AccumulateGrad` hooks on the detached leaves fire at exactly the right time. Grads are captured from the hook argument (not `.grad`, which is None at hook time).

## Verified Behavior

Profiler traces confirm batched collectives across all modes:

| Mode | AllGather | ReduceScatter | Mechanism | Profiler Trace |
|------|----------|---------------|-----------|----------------|
| Eager | 17 | 8 | Batched `Placement.unshard()` / `reduce_grad()` | https://fburl.com/dt2ljemb |
| JIT (inductor) | 113 | 57 | Per-param, compiler rebatches | https://fburl.com/1qmg8vmh |
| AOT | 114 | 57 | Per-param, compiler rebatches | https://fburl.com/ss9bqtbv |

Convergence matches across all modes (loss ≈ 5.6-5.7 at step 5).

Repro commands (`batch=8, seq_len=6144, 4 GPUs`):
```bash
# Eager
NGPU=4 MODULE=graph_trainer.flex_shard_llama3 CONFIG=graph_trainer_flex_shard_llama3_debugmodel \
  ./run_train.sh --training.local_batch_size 8 --training.seq_len 6144 \
  --compile.mode None --profiling.enable_profiling --profiling.save_traces_folder profile_eager

# JIT
NGPU=4 MODULE=graph_trainer.flex_shard_llama3 CONFIG=graph_trainer_flex_shard_llama3_debugmodel \
  ./run_train.sh --training.local_batch_size 8 --training.seq_len 6144 \
  --compile.mode jit --compile.backend inductor --profiling.enable_profiling --profiling.save_traces_folder profile_jit

# AOT
NGPU=4 MODULE=graph_trainer.flex_shard_llama3 CONFIG=graph_trainer_flex_shard_llama3_debugmodel \
  ./run_train.sh --training.local_batch_size 8 --training.seq_len 6144 \
  --compile.mode aot --profiling.enable_profiling --profiling.save_traces_folder profile_aot
```

## Appendix

### Why store `_unsharded_for_reduce` only on first forward?

Checkpoint recomputation creates NEW detached leaves, but autograd accumulates grad on the ORIGINAL leaf (the one checkpoint saved via `_Holder`). If `_unsharded_for_reduce` were overwritten during recomputation, `_reduce_fn` would read from the recomputed leaf (which has no grad).

### Composition with SAC

Reshard and SAC use the same `CheckpointPolicy` annotation mechanism. They compose:
- SAC marks activations `PREFER_RECOMPUTE` / `MUST_SAVE`
- Reshard marks unshard sequences `MUST_RECOMPUTE` (overrides SAC)
- `ac_graph_id = 100000` prevents partitioner from treating reshard nodes as part of a user AC region
