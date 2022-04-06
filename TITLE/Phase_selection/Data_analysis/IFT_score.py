import h5py
import numpy as np

import Reconstruction.fastmri
from Reconstruction.fastmri.data import transforms as T
import os
import Phase_selection.activemri
import torch
from typing import Dict, Optional, Sequence, Tuple, Union
import pickle


def _compute_score_given_tensors(
        reconstruction: torch.Tensor, ground_truth: torch.Tensor
) -> Dict[str, np.ndarray]:
    mse = Phase_selection.activemri.envs.util.compute_mse(reconstruction, ground_truth)
    nmse = Phase_selection.activemri.envs.util.compute_nmse(reconstruction, ground_truth)
    ssim = Phase_selection.activemri.envs.util.compute_ssim(reconstruction, ground_truth)
    psnr = Phase_selection.activemri.envs.util.compute_psnr(reconstruction, ground_truth)

    return {"mse": mse, "nmse": nmse, "ssim": ssim, "psnr": psnr}


def ifft_permute_maybe_shift(
    x: torch.Tensor, normalized: bool = False, ifft_shift: bool = False
) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1)
    y = torch.ifft(x, 2, normalized=normalized)
    if ifft_shift:
        y = Reconstruction.fastmri.ifftshift(y, dim=(1, 2))
    return y.permute(0, 3, 1, 2)


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.
    """

    def __init__(self, center_fractions: Sequence[float], accelerations: Sequence[int]):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
        """
        if not len(center_fractions) == len(accelerations):
            raise ValueError(
                "Number of center fractions should match number of accelerations"
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()  # pylint: disable=no-member

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None, **img_id,
    ) -> torch.Tensor:
        raise NotImplementedError

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        return center_fraction, acceleration


class TrajectoryMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the RandomMaskFunc object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.

    read low frequency setting, and the sequence of the trajectory, to finish the 4x or other acceleration
    """

    def __init__(self, center_fractions, accelerations, trajectory_dir):
        super(TrajectoryMaskFunc, self).__init__(center_fractions, accelerations)
        self.trajectory_dir = trajectory_dir
        print(trajectory_dir)
        with open(self.trajectory_dir, "rb") as f:
            self.trajectories = pickle.load(f)

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None, **img_id
    ) -> torch.Tensor:
        """
        Create the mask.

        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.

        Returns:
            A mask of the specified shape.
        """

        # add dir of pkl, index the img_id to get trajectory
        num_cols = shape[-2]

        center_fraction, acceleration = self.choose_acceleration()
        RF_num = num_cols / acceleration
        fname = img_id["fname"]
        slice_id = img_id["slice_num"]
        trajectory = self.trajectories[fname][slice_id]
        mask_idx = trajectory[:int(RF_num)].long()
        mask = torch.zeros(num_cols)
        mask[mask_idx] = 1
        mask = mask.unsqueeze(1).unsqueeze(0)

        return mask


def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
    **img_id,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data
            mask: The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed, **img_id)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask


center_fractions = [0.08]
accelerations = [3.68]
method = "TIPS" # TIPS ss-ddqn-org(Pineda) evaluator(Zhang) random
if method is not "random":
    trajectory_dir = "../trajectories4IFTRecon/" + method + "/" + method + "_trajectories.pkl"
    mask_func = TrajectoryMaskFunc(center_fractions, accelerations, trajectory_dir)
    file_list = os.listdir('path/to/active_data/knee_singlecoil_test')
    ssim = []
    nmse = []
    psnr = []
    mse = []
    for fname in file_list:
        file = 'path/to/active_data/knee_singlecoil_test/' + fname
        print(file)
        hf = h5py.File(file)
        volume_kspace = hf['kspace'][()]
        for slice_num, slice_kspace in enumerate(volume_kspace):
            img_id = {"fname": fname, "slice_num": slice_num}
            slice_kspace = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
            masked_kspace, mask = apply_mask(slice_kspace, mask_func, seed=0, padding=None, **img_id)
            slice_image = Reconstruction.fastmri.ifft2c(slice_kspace)  # Apply Inverse Fourier Transform to get the complex image
            slice_image_abs = Reconstruction.fastmri.complex_abs(slice_image)  # Compute absolute value to get a real image
            slice_image_masked = Reconstruction.fastmri.ifft2c(masked_kspace)  # Apply Inverse Fourier Transform to get the complex image
            slice_image_masked_abs = Reconstruction.fastmri.complex_abs(slice_image_masked)  # Compute absolute value to get a real image

            IFT_recon = slice_image_masked_abs.unsqueeze(0)
            gt = slice_image_abs.unsqueeze(0)

            IFT_recon = Phase_selection.activemri.data.transforms.center_crop(
                IFT_recon, (320, 320)
            )
            gt = Phase_selection.activemri.data.transforms.center_crop(
                gt, (320, 320)
            )

            scores = _compute_score_given_tensors(IFT_recon, gt)
            ssim_temp = scores["ssim"]
            psnr_temp = scores["psnr"]
            nmse_temp = scores["nmse"]
            mse_temp = scores["mse"]
            ssim.append(ssim_temp)
            psnr.append(psnr_temp)
            nmse.append(nmse_temp)
            mse.append(mse_temp)

    print("ssim mean: ", np.array(ssim).mean())
    print("psnr mean: ", np.array(psnr).mean())
    print("nmse mean: ", np.array(nmse).mean())
    print("mse mean: ", np.array(mse).mean())
else:
    idx = 70
    pkl_file = "../trajectories4IFTRecon/" + method + "/img_ids.pkl"
    scs_file = "../trajectories4IFTRecon/" + method + "/scores.npy"
    with open(pkl_file, "rb") as f:
        img_list = pickle.load(f)
    scores = np.load(scs_file, allow_pickle=True).item()

    ssim_scores = scores['ssim']
    psnr_scores = scores['psnr']
    mse_scores = scores['mse']
    nmse_scores = scores['nmse']

    ssim_idx = ssim_scores[:, idx]
    psnr_idx = psnr_scores[:, idx]
    mse_idx = mse_scores[:, idx]
    nmse_idx = nmse_scores[:, idx]

    ssim_mean = ssim_idx.mean()
    psnr_mean = psnr_idx.mean()
    mse_mean = mse_idx.mean()
    nmse_mean = nmse_idx.mean()

    print("Results of {}".format(method))
    print("ssim of {} RF is: ".format(idx), ssim_mean)
    print("psnr of {} RF is: ".format(idx), psnr_mean)
    print("mse of {} RF is: ".format(idx), mse_mean)
    print("nmse of {} RF is: ".format(idx), nmse_mean)






