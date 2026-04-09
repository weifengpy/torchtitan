# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.flex_shard import (
    BucketSpec,
    flat_shard_placements,
    flex_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
    param_boundary_placements,
    RaggedShard,
    Shard,
)
from torchtitan.experiments.graph_trainer.common_utils import apply_graph_ac
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.flex_shard_llama3.model import (
    _register_dcp_hooks,
    FlexShardLlama3Model,
)
from torchtitan.experiments.graph_trainer.llama3.parallelize import annotate_llama
from torchtitan.models.llama3.parallelize import apply_tp
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def parallelize_llama_flex_shard(
    model: FlexShardLlama3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """
    Apply FlexShard data parallelism, activation checkpointing, and compilation.

    Same structure as graph_trainer/llama3/parallelize.py but uses FlexShard
    instead of SimpleFSDP for data parallelism. Supports multi-mesh
    composition (FSDP + TP) via DTensorAwareParametrization.
    """
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    annotate_llama(model)

    # Apply TP first — parameters become DTensors on the TP mesh.
    # FlexShard's DTensorAwareParametrization handles DTensor inputs.
    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        apply_tp(
            model,
            tp_mesh,
            enable_loss_parallel=not parallelism.disable_loss_parallel,
        )

    if ac_config.mode != "none":
        apply_graph_ac(compile_config, ac_config)

    # Apply FlexShard data parallelism
    if parallel_dims.dp_shard_enabled:
        fsdp_mesh = parallel_dims.get_mesh("fsdp")

        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

        offload_policy = None
        if training.enable_cpu_offload:
            offload_policy = OffloadPolicy(pin_memory=True)

        # One bucket per transformer block + embeddings + norm/output,
        # matching SimpleFSDP's natural bucket boundaries.
        # Offload only transformer layers (embeddings and norm/output are
        # small and not worth the H2D transfer cost).
        buckets = (
            [
                BucketSpec(["tok_embeddings.*"], mp_policy=mp_policy),
            ]
            + [
                BucketSpec(
                    [f"layers.{i}.*"],
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                )
                for i in range(len(model.layers))
            ]
            + [
                BucketSpec(["norm.*", "output.*"], mp_policy=mp_policy),
            ]
        )

        # Select placement policy from config
        placement_name = getattr(compile_config, "shard_placement", "per_param")
        placement_policies = {
            "per_param": {"*": Shard(0)},
            "flat_shard": flat_shard_placements,
            "param_boundary": param_boundary_placements,
        }
        if placement_name == "ragged":
            ws = fsdp_mesh.size()
            local_units = tuple(range(1, ws + 1))
            shard_placement_fn = {"*": RaggedShard((0,), local_units)}
        else:
            shard_placement_fn = placement_policies.get(placement_name, {"*": Shard(0)})

        reshard_after_fwd = get_fsdp_reshard_after_forward_policy(
            parallelism.fsdp_reshard_after_forward, parallel_dims.pp_enabled
        )
        flex_shard(
            model,
            fsdp_mesh,
            shard_placement_fn=shard_placement_fn,
            buckets=buckets,
            reshard_after_forward=reshard_after_fwd,
        )
        _register_dcp_hooks(model)
        logger.info("Applied FlexShard data parallelism to the model")

    # Apply compilation based on mode
    model = apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )

    return model
