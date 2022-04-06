# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import Phase_selection.activemri.baselines as baselines
import Phase_selection.activemri.envs as envs

import time
import Reconstruction.fastmri

import torch
import Phase_selection.activemri.data.singlecoil_knee_data as scknee_data

# from memory_profiler import profile


def add2masks(masks, file_id, slice_id, mask):
    """
    The mask here is from envs, but the "fastmri.ifftshift" is used in sckneedata and the process of reconstruction,
    that is to say, mask and k-space in envs are all not shifted (low frequency in corners), so if the original k-space
    will be masked, the mask which used must convert to original mask, the "fastmri.ifftshift" is used to convert it.
    Args:
        masks: the dict to save all masks of current baseline for next training of reconstruction network
        file_id: file names in current batch
        slice_id: slice ids in current batch and current files
        mask: the trajectories returned by current baseline, also note as mask

    Returns: masks, the same as the parameter "masks" in input.

    """
    mask = Reconstruction.fastmri.ifftshift(mask)
    if file_id in masks.keys():
        masks[file_id].update({slice_id: mask})
    else:
        masks.update({file_id: {slice_id: mask}})

    return masks

def add2trajectories(trajectories, file_id, slice_id, trajectory, width):
    """
    The mask here is from envs, but the "fastmri.ifftshift" is used in sckneedata and the process of reconstruction,
    that is to say, mask and k-space in envs are all not shifted (low frequency in corners), so if the original k-space
    will be masked, the mask which used must convert to original mask, the "fastmri.ifftshift" is used to convert it.
    Args:
        masks: the dict to save all masks of current baseline for next training of reconstruction network
        file_id: file names in current batch
        slice_id: slice ids in current batch and current files
        mask: the trajectories returned by current baseline, also note as mask

    Returns: masks, the same as the parameter "masks" in input.

    """
    trajectory = (trajectory + width / 2) % width
    if file_id in trajectories.keys():
        trajectories[file_id].update({slice_id: trajectory})
    else:
        trajectories.update({file_id: {slice_id: trajectory}})

    return trajectories

# @profile
def evaluate(
    env: envs.envs.ActiveMRIEnv,
    policy: baselines.Policy,
    num_episodes: int,
    seed: int,
    split: str,
    verbose: Optional[bool] = False,
) -> Tuple[Dict[str, Any], List[Tuple[Any, Any]], Union[dict, Dict]]:
    env.seed(seed)
    if split == "test":
        env.set_test()
    elif split == "val":
        env.set_val()
    else:
        raise ValueError(f"Invalid evaluation split: {split}.")

    score_keys = env.score_keys()
    all_scores = dict(
        (k, np.zeros((num_episodes * env.num_parallel_episodes, env.budget + 1)))
        for k in score_keys
    )
    temp_trajectories = np.zeros((4, env.budget))
    all_trajectories = dict()
    all_img_ids = []
    trajectories_written = 0

    time_s = time.time()
    for episode in range(num_episodes):
        step = 0
        obs, meta = env.reset()
        if not obs:
            break  # no more images
        # in case the last batch is smaller
        actual_batch_size = len(obs["reconstruction"])
        if verbose:
            msg = ", ".join(
                [
                    f"({meta['fname'][i]}, {meta['slice_id'][i]})"
                    for i in range(actual_batch_size)
                ]
            )
            time_n = time.time()
            period = (time_n - time_s) / (episode + 1e-10)
            ETA = (num_episodes - episode) * period
            ETA_h = int(ETA / 3600)
            ETA_m = int((ETA - ETA_h * 3600) / 60)
            ETA_s = int(ETA % 60)
            print(f"Read images: {msg} ETA: {ETA_h}:{ETA_m}:{ETA_s} Time per episode: {period}")
        for i in range(actual_batch_size):
            all_img_ids.append((meta["fname"][i], meta["slice_id"][i]))
        batch_idx = slice(
            trajectories_written, trajectories_written + actual_batch_size
        )
        for k in score_keys:
            all_scores[k][batch_idx, step] = meta["current_score"][k]
        trajectories_written += actual_batch_size
        all_done = False
        while not all_done:
            step += 1
            action = policy.get_action(obs)
            obs, reward, done, meta_back = env.step(action)
            for k in score_keys:
                all_scores[k][batch_idx, step] = meta_back["current_score"][k]
            all_done = all(done)
            temp_trajectories[:actual_batch_size, step - 1] = action

        obs_width = obs["reconstruction"].size()[2]
        init_mask = meta["current_mask"].squeeze(1)[0]
        init_mask[scknee_data.MICCAI2020Data.START_PADDING: scknee_data.MICCAI2020Data.END_PADDING] = 0
        init_idx = torch.nonzero(init_mask).squeeze(1).repeat(4, 1)
        fused_trajectories = torch.cat([init_idx, torch.tensor(temp_trajectories)], dim= 1)

        for i in range(actual_batch_size):
            all_trajectories = add2trajectories(all_trajectories, meta["fname"][i], meta["slice_id"][i], fused_trajectories[i], obs_width)

    for k in score_keys:
        all_scores[k] = all_scores[k][: len(all_img_ids), :]
    return all_scores, all_img_ids, all_trajectories
