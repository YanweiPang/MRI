# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import Reconstruction.fastmri.models
import torch

import Phase_selection.activemri.models


# noinspection PyAbstractClass
class Unet(Phase_selection.activemri.models.Reconstructor):
    def __init__(
        self, in_chans=2, out_chans=2, chans=32, num_pool_layers=4, drop_prob=0.0
    ):
        super().__init__()
        self.unet = Reconstruction.fastmri.models.Unet(
            in_chans,
            out_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
        )

    def forward(  # type: ignore
        self, image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> Dict[str, Any]:
        output = self.unet(image).squeeze(1)
        std = std.unsqueeze(1).unsqueeze(2)
        mean = mean.unsqueeze(1).unsqueeze(2)
        reconstruction = output * std + mean
        return {"reconstruction": reconstruction}

    def init_from_checkpoint(self, checkpoint: Dict[str, Any]) -> Optional[Any]:
        self.load_state_dict(checkpoint["state_dict"])
        return None
