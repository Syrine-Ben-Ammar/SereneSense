# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 SereneSense Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Legacy Models Module
====================
Legacy CNN and CRNN models for comparative analysis with modern transformer-based approaches.
Preserves the original architecture from the OLD notebooks for benchmarking purposes.

Models:
    - CNNMFCCModel: 2D-CNN on MFCC features (242K parameters)
    - CRNNMFCCModel: CRNN with BiLSTM on MFCC features (1.5M parameters)

Features:
    - MFCC feature extraction (40 coefficients + delta + delta-delta)
    - Legacy SpecAugment implementation
    - Configuration-driven setup
    - Compatible with modern SereneSense training pipeline
"""

from .base_legacy_model import BaseLegacyModel
from .cnn_mfcc import CNNMFCCModel
from .crnn_mfcc import CRNNMFCCModel
from .legacy_config import LegacyModelConfig, LegacyModelType

__all__ = [
    'BaseLegacyModel',
    'LegacyModelConfig',
    'CNNMFCCModel',
    'CRNNMFCCModel',
    'LegacyModelType',
]
