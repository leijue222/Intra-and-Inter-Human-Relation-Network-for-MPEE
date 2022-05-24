# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
# from .coco_ochuman import COCODataset as coco
from .coco import COCODataset as coco
from .crowdpose import CROWDPOSEDataset as crowdpose
from .ochuman import OCHumanDataset as OCHuman