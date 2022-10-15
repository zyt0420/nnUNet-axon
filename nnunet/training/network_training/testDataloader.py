#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset, load_dataset, DataLoader3D
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from torch.optim import lr_scheduler


def initialize(training=True, force_load_plans=False):
    """
    - replaced get_default_augmentation with get_moreDA_augmentation
    - enforce to only run this code once
    - loss function wrapper for deep supervision

    :param training:
    :param force_load_plans:
    :return:
    """
    plans = load_pickle("/media/root/data4/zyt/nnUNet-axon/DATASET/nnUNet_preprocessed/Task7090_Renlab_aug_new/nnUNetPlansv2.1_16GB_plans_3D.pkl")
    dataset_directory = "/media/root/data4/zyt/nnUNet-axon/DATASET/nnUNet_preprocessed/Task7090_Renlab_aug_new/"
    folder_with_preprocessed_data = join(dataset_directory, plans['data_identifier'] + "_stage%d" % 0)
    unpack_dataset(folder_with_preprocessed_data)
    dataset = load_dataset(folder_with_preprocessed_data)
    dl = DataLoader3D(dataset, np.array(plans['patch_size']).astype(int), np.array(plans['patch_size']).astype(int), 2,
                      oversample_foreground_percent=0.33)
    print(np.array(plans['patch_size']))


if __name__ == "__main__":
    initialize()
