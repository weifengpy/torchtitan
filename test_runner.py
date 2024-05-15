# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


parser = argparse.ArgumentParser()
parser.add_argument("output_dir")
args = parser.parse_args()


@dataclass
class OverrideDefinitions:
    """
    This class is used to define the override definitions for the integration tests.
    """

    override_args: Sequence[Sequence[str]] = tuple(tuple(" "))
    test_descr: str = "default"
    requires_seed_checkpoint: bool = False
    ngpu: int = 4


CONFIG_DIR = "./train_configs"

"""
key is the config file name and value is a list of OverrideDefinitions
that is used to generate variations of integration tests based on the
same root config file.
"""
integration_tests_flavors = defaultdict(list)
integration_tests_flavors["debug_model.toml"] = [
    OverrideDefinitions(
        [
            [
                f"--job.dump_folder {args.output_dir}/default/",
            ],
        ],
        "Default",
    ),
    OverrideDefinitions(
        [
            [
                "--training.compile",
                f"--job.dump_folder {args.output_dir}/1d_compile/",
            ],
        ],
        "1D compile",
    ),
    OverrideDefinitions(
        [
            [
                "--training.tensor_parallel_degree 2 --model.norm_type=rmsnorm",
                f"--job.dump_folder {args.output_dir}/eager_2d/",
            ],
        ],
        "Eager mode 2DParallel",
    ),
    OverrideDefinitions(
        [
            [
                "--checkpoint.enable_checkpoint",
                f"--job.dump_folder {args.output_dir}/full_checkpoint/",
            ],
            [
                "--checkpoint.enable_checkpoint",
                f"--job.dump_folder {args.output_dir}/full_checkpoint/",
                "--training.steps 20",
            ],
        ],
        "Checkpoint Integration Test - Save Load Full Checkpoint",
    ),
    OverrideDefinitions(
        [
            [
                "--checkpoint.enable_checkpoint",
                f"--job.dump_folder {args.output_dir}/model_weights_only_fp32/",
                "--checkpoint.model_weights_only",
            ],
        ],
        "Checkpoint Integration Test - Save Model Weights Only fp32",
    ),
    OverrideDefinitions(
        [
            [
                "--checkpoint.enable_checkpoint",
                f"--job.dump_folder {args.output_dir}/model_weights_only_bf16/",
                "--checkpoint.model_weights_only",
                "--checkpoint.export_dtype bfloat16",
            ],
        ],
        "Checkpoint Integration Test - Save Model Weights Only bf16",
    ),
    OverrideDefinitions(
        [
            [
                "--checkpoint.enable_checkpoint",
                f"--job.dump_folder {args.output_dir}/pp/",
                "--experimental.pipeline_parallel_degree 2",
                "--experimental.pipeline_parallel_split_points layers.1",
                "--training.data_parallel_degree 1",
                "--model.norm_type rmsnorm",  # TODO fix fused_rmsnorm issue
            ],
        ],
        "PP 1D test",
        requires_seed_checkpoint=True,
        ngpu=2,
    ),
    OverrideDefinitions(
        [
            [
                "--checkpoint.enable_checkpoint",
                f"--job.dump_folder {args.output_dir}/pp_dp/",
                "--experimental.pipeline_parallel_degree 2",
                "--experimental.pipeline_parallel_split_points layers.1",
                "--training.data_parallel_degree 2",
                "--model.norm_type fused_rmsnorm",
            ],
        ],
        "PP+DP 2D test",
        requires_seed_checkpoint=True,
    ),
    OverrideDefinitions(
        [
            [
                "--checkpoint.enable_checkpoint",
                f"--job.dump_folder {args.output_dir}/pp_tp/",
                "--experimental.pipeline_parallel_degree 2",
                "--experimental.pipeline_parallel_split_points layers.1",
                "--training.tensor_parallel_degree 2",
                "--model.norm_type rmsnorm",  # TODO fix fused_rmsnorm issue
            ],
        ],
        "PP+TP 2D test",
        requires_seed_checkpoint=True,
    ),
    # oh.. not enough GPUs?
    # OverrideDefinitions(
    #     [
    #         [
    #             "--checkpoint.enable_checkpoint",
    # f"--job.dump_folder {args.output_dir}/pp_dp_tp/",
    #             "--experimental.pipeline_parallel_degree 2",
    #             "--experimental.pipeline_parallel_split_points layers.1",
    #             "--training.data_parallel_degree 2",
    #             "--training.tensor_parallel_degree 2",
    #             "--model.norm_type rmsnorm",  # TODO fix fused_rmsnorm issue
    #         ],
    #     ],
    #     "PP+DP+TP 3D test",
    #     requires_seed_checkpoint=True,
    # ),
]


def _run_cmd(cmd):
    return subprocess.run(
        [cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
    )


def run_test(test_flavor: OverrideDefinitions, full_path: str):
    # run_test supports sequence of tests.
    for override_arg in test_flavor.override_args:

        cmd = f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK=0,1,2,3 ./run_llama_train.sh"
        if override_arg:
            cmd += (
                " " + " ".join(override_arg)
            )
        print(
            f"=====Integration test, flavor : {test_flavor.test_descr}, command : {cmd}====="
        )

        if test_flavor.requires_seed_checkpoint:
            dump_folder_arg = None
            for arg in override_arg:
                if "--job.dump_folder" in arg:
                    dump_folder_arg = arg
            assert (
                dump_folder_arg is not None
            ), "Can't use seed checkpoint if folder is not specified"
            print("Creating seed checkpoint")
            result = _run_cmd(
                f"CONFIG_FILE={full_path} ./create_seed_checkpoint.sh {dump_folder_arg}"
            )
            print(result.stdout)

        result = _run_cmd(cmd)
        print(result.stdout)
        if result.returncode != 0:
            raise Exception(
                f"Integration test failed, flavor : {test_flavor.test_descr}, command : {cmd}"
            )


for config_file in os.listdir(CONFIG_DIR):
    if config_file.endswith(".toml"):
        full_path = os.path.join(CONFIG_DIR, config_file)
        with open(full_path, "rb") as f:
            config = tomllib.load(f)
            is_integration_test = config["job"].get("use_for_integration_test", False)
            if is_integration_test:
                for test_flavor in integration_tests_flavors[config_file]:
                    run_test(test_flavor, full_path)
