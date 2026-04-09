#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FlexShard parametrization (Phase 2a).

Usage:
    # Single-process tests (no GPU/NCCL required):
    python -m pytest test_flex_shard_parametrization.py -v -k "not Distributed"

    # Distributed correctness tests:
    torchrun --nproc_per_node=2 test_flex_shard_parametrization.py
"""

import unittest

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Guard behavior tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestActiveParametrizationGuard(unittest.TestCase):
    """Test _active_parametrization guard and disable_active_parametrization."""

    def test_guard_disabled_returns_raw_shard(self):
        """With guard disabled, ShardParametrization returns input unchanged."""
        from torchtitan.experiments.flex_shard import (
            disable_active_parametrization,
            ShardParametrization,
        )

        param = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        shard = torch.randn(4, 8)
        with disable_active_parametrization():
            result = param(shard)
        self.assertIs(result, shard)

    def test_guard_disabled_returns_raw_flat_shard(self):
        """With guard disabled, FlatShardParametrization returns input unchanged."""
        from torchtitan.experiments.flex_shard import (
            disable_active_parametrization,
            FlatShardParametrization,
        )

        param = FlatShardParametrization(
            group_name="fake",
            world_size=2,
            original_shape=torch.Size([4, 8]),
        )
        flat_shard = torch.randn(16)
        with disable_active_parametrization():
            result = param(flat_shard)
        self.assertIs(result, flat_shard)

    def test_guard_restores_after_context(self):
        """Guard restores to True after context manager exits."""
        import importlib

        fs = importlib.import_module("torchtitan.experiments.flex_shard.flex_shard")

        self.assertTrue(fs._active_parametrization)
        with fs.disable_active_parametrization():
            self.assertFalse(fs._active_parametrization)
        self.assertTrue(fs._active_parametrization)

    def test_guard_restores_on_exception(self):
        """Guard restores to True even if exception is raised."""
        import importlib

        fs = importlib.import_module("torchtitan.experiments.flex_shard.flex_shard")

        try:
            with fs.disable_active_parametrization():
                raise RuntimeError("test")
        except RuntimeError:
            pass
        self.assertTrue(fs._active_parametrization)


# ---------------------------------------------------------------------------
# Property registration tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestRegisterParametrization(unittest.TestCase):
    """Test _register_parametrization creates correct property getters."""

    def test_property_created_on_module(self):
        """Property getter is created on the module's dynamic subclass."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _register_parametrization,
            ShardParametrization,
        )

        module = nn.Linear(8, 4, bias=False)
        param = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        _register_parametrization(module, {"weight": param})

        # The module's class should be a dynamic subclass
        self.assertIn("FlexShard", type(module).__name__)
        # Property should exist on the class
        self.assertIsInstance(type(module).__dict__["weight"], property)

    def test_state_dict_bypasses_property(self):
        """state_dict reads _parameters directly, not through property."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _register_parametrization,
            ShardParametrization,
        )

        module = nn.Linear(8, 4, bias=False)
        original_shape = module.weight.shape

        # Register parametrization with guard disabled so no NCCL needed
        param = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        _register_parametrization(module, {"weight": param})

        # state_dict should return the raw parameter (bypasses property)
        sd = module.state_dict()
        self.assertEqual(sd["weight"].shape, original_shape)

    def test_multiple_params_on_same_module(self):
        """Multiple parameters can be parametrized on the same module."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _register_parametrization,
            ShardParametrization,
        )

        module = nn.Linear(8, 4)  # has weight and bias
        param_w = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        param_b = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        _register_parametrization(module, {"weight": param_w, "bias": param_b})

        self.assertIsInstance(type(module).__dict__["weight"], property)
        self.assertIsInstance(type(module).__dict__["bias"], property)


# ---------------------------------------------------------------------------
# Distributed correctness tests (torchrun only)
# ---------------------------------------------------------------------------


class TestDistributedParametrization(unittest.TestCase):
    """Multi-process correctness tests for parametrized FlexShard.

    Run with: torchrun --nproc_per_node=2 test_flex_shard_parametrization.py
    """

    @classmethod
    def setUpClass(cls):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        cls.rank = torch.distributed.get_rank()
        cls.world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(cls.rank % torch.cuda.device_count())

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.cuda.synchronize()
            torch.distributed.destroy_process_group()

    def test_param_access_triggers_allgather(self):
        """Accessing module.weight returns the full (unsharded) tensor."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import flex_shard

        mesh = init_device_mesh("cuda", (self.world_size,))

        model = nn.Linear(8, 4, bias=False, device="cuda")
        # Broadcast weights so all ranks start with the same full tensor
        torch.distributed.broadcast(model.weight.data, src=0)
        full_ref = model.weight.data.clone()

        flex_shard(model, mesh, register_hooks=False)

        # Accessing model.weight should trigger all-gather via property
        result = model.weight
        torch.testing.assert_close(result, full_ref)

    def test_state_dict_returns_sharded(self):
        """state_dict() returns sharded params, not unsharded."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import flex_shard

        mesh = init_device_mesh("cuda", (self.world_size,))

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        flex_shard(model, mesh, register_hooks=False)

        sd = model.state_dict()
        # state_dict bypasses property, returns local shard
        expected_rows = 4 // self.world_size
        self.assertEqual(sd["weight"].shape, (expected_rows, 8))

    def test_disable_guard_returns_sharded(self):
        """With guard disabled, param access returns raw sharded tensor."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import (
            disable_active_parametrization,
            flex_shard,
        )

        mesh = init_device_mesh("cuda", (self.world_size,))

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        flex_shard(model, mesh, register_hooks=False)

        with disable_active_parametrization():
            result = model.weight
        expected_rows = 4 // self.world_size
        self.assertEqual(result.shape, (expected_rows, 8))

    def test_forward_produces_correct_output(self):
        """Forward pass through parametrized model produces correct results."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import flex_shard

        mesh = init_device_mesh("cuda", (self.world_size,))

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)
        ref_weight = model.weight.data.clone()

        # Reference output
        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)
        ref_output = x @ ref_weight.t()

        flex_shard(model, mesh, register_hooks=False)
        output = model(x)

        torch.testing.assert_close(output, ref_output)

    def test_checkpoint_roundtrip_with_guard(self):
        """Per-rank save/load roundtrip preserves sharded params and forward correctness.

        FlexShard params are plain tensors (not DTensors), so checkpoint uses
        per-rank torch.save/load. disable_active_parametrization ensures
        state_dict access and load_state_dict don't trigger collectives.
        """
        import os
        import shutil
        import tempfile

        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import (
            disable_active_parametrization,
            flex_shard,
        )

        mesh = init_device_mesh("cuda", (self.world_size,))

        # Create and shard model
        torch.manual_seed(42)
        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)
        full_ref = model.weight.data.clone()

        flex_shard(model, mesh, register_hooks=False)

        # state_dict returns sharded params (bypasses property)
        sd_before = {k: v.clone() for k, v in model.state_dict().items()}
        expected_rows = 4 // self.world_size
        self.assertEqual(sd_before["weight"].shape, (expected_rows, 8))

        # Guard also returns sharded params via param access
        with disable_active_parametrization():
            guarded_weight = model.weight
        self.assertEqual(guarded_weight.shape, (expected_rows, 8))

        # Share tmpdir across ranks
        obj_list = [tempfile.mkdtemp() if self.rank == 0 else ""]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        tmpdir = obj_list[0]

        try:
            # Per-rank save
            torch.save(
                model.state_dict(),
                os.path.join(tmpdir, f"rank_{self.rank}.pt"),
            )
            torch.distributed.barrier()

            # Create fresh model with different weights, shard it
            torch.manual_seed(99)
            model2 = nn.Linear(8, 4, bias=False, device="cuda")
            torch.distributed.broadcast(model2.weight.data, src=0)
            flex_shard(model2, mesh, register_hooks=False)

            # Per-rank load
            sd2 = torch.load(
                os.path.join(tmpdir, f"rank_{self.rank}.pt"),
                weights_only=True,
                map_location=f"cuda:{self.rank}",
            )
            model2.load_state_dict(sd2)
        finally:
            torch.distributed.barrier()
            if self.rank == 0:
                shutil.rmtree(tmpdir, ignore_errors=True)

        # Sharded params should match
        sd_after = model2.state_dict()
        torch.testing.assert_close(sd_after["weight"], sd_before["weight"])

        # Forward should produce the same result as original full weights
        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)
        ref_output = x @ full_ref.t()
        output = model2(x)
        torch.testing.assert_close(output, ref_output)

    def _to_dtensor_sd(self, model, mesh):
        """Wrap FlexShard state_dict as DTensors for DCP.

        Maps FlexShard placements to DTensor Shard(0):
        - Shard(dim): each rank holds a chunk along dim → DTensor Shard(dim)
        - FlatShard: in parametrization mode, decomposed to per-param
          FlatShard(0, numel, numel), i.e. 1D Shard(0) → DTensor Shard(0)
        """
        from torch.distributed.tensor import DTensor, Shard as DTensorShard

        from torchtitan.experiments.flex_shard import Shard as FlexShard
        from torchtitan.experiments.flex_shard.flex_shard import FlatShard

        sd = {}
        plain_sd = model.state_dict()
        fqn_to_placement = {}
        for ds in model.dstorages:
            for fqn, info in ds.param_infos.items():
                fqn_to_placement[fqn] = info.placements
        for k, v in plain_sd.items():
            placements = fqn_to_placement.get(k)
            if placements is not None:
                dt_placements = []
                for p in placements:
                    if isinstance(p, FlexShard):
                        dt_placements.append(DTensorShard(p.dim))
                    elif isinstance(p, FlatShard):
                        # Per-param FlatShard(0, numel, numel) is 1D Shard(0)
                        dt_placements.append(DTensorShard(0))
                    else:
                        raise ValueError(f"Unsupported placement {p}")
                sd[k] = DTensor.from_local(v, mesh, dt_placements, run_check=False)
            else:
                sd[k] = v
        return sd

    def _dcp_roundtrip(self, model, model2, mesh, full_ref):
        """Save model via DCP, load into model2, verify correctness."""
        import shutil
        import tempfile

        import torch.distributed.checkpoint as dcp
        from torch.distributed.tensor import DTensor

        sd_before_clone = {k: v.clone() for k, v in model.state_dict().items()}

        obj_list = [tempfile.mkdtemp() if self.rank == 0 else ""]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        tmpdir = obj_list[0]

        try:
            dcp.save(self._to_dtensor_sd(model, mesh), checkpoint_id=tmpdir)

            sd2 = self._to_dtensor_sd(model2, mesh)
            dcp.load(sd2, checkpoint_id=tmpdir)

            plain_sd2 = model2.state_dict()
            for k, v in sd2.items():
                local = v.to_local() if isinstance(v, DTensor) else v
                plain_sd2[k].copy_(local)
        finally:
            torch.distributed.barrier()
            if self.rank == 0:
                shutil.rmtree(tmpdir, ignore_errors=True)

        # Sharded params should match original
        sd_after = model2.state_dict()
        for k in sd_before_clone:
            torch.testing.assert_close(
                sd_after[k], sd_before_clone[k], msg=f"{k} mismatch"
            )

        # Forward should produce the same result as original full weights
        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)
        ref_output = x @ full_ref.t()
        output = model2(x)
        torch.testing.assert_close(output, ref_output)

    def test_dcp_save_load_shard(self):
        """DCP roundtrip with Shard(0) (FSDP2-style)."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import flex_shard

        mesh = init_device_mesh("cuda", (self.world_size,))

        torch.manual_seed(42)
        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)
        full_ref = model.weight.data.clone()
        flex_shard(model, mesh, register_hooks=False)

        torch.manual_seed(99)
        model2 = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model2.weight.data, src=0)
        flex_shard(model2, mesh, register_hooks=False)

        self._dcp_roundtrip(model, model2, mesh, full_ref)

    def test_dcp_save_load_flat_shard(self):
        """DCP roundtrip with FlatShard (FSDP1-style)."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import flat_shard_placements, flex_shard

        mesh = init_device_mesh("cuda", (self.world_size,))

        torch.manual_seed(42)
        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)
        full_ref = model.weight.data.clone()
        flex_shard(
            model, mesh, shard_placement_fn=flat_shard_placements, register_hooks=False
        )

        torch.manual_seed(99)
        model2 = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model2.weight.data, src=0)
        flex_shard(
            model2, mesh, shard_placement_fn=flat_shard_placements, register_hooks=False
        )

        self._dcp_roundtrip(model, model2, mesh, full_ref)


if __name__ == "__main__":
    unittest.main()
