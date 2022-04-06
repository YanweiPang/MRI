# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import Phase_selection.activemri.baselines.ddqn as ddqn
import Phase_selection.activemri.envs as envs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=70)
    parser.add_argument("--num_parallel_episodes", type=int, default=4)
    parser.add_argument("--training_dir", type=str, default="path/to/checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--extreme_acc", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    env = envs.MICCAI2020Env(
        args.num_parallel_episodes,
        args.budget,
        extreme=args.extreme_acc,
        seed=args.seed,
    )
    tester = ddqn.DDQNTester(env, args.training_dir, args.device)
    tester()
