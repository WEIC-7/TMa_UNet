# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from abc import abstractmethod
from math import ceil, sqrt

import torch

from ..transform import RandomizableTransform

__all__ = ["MixUp", "CutMix", "CutOut", "Mixer"]


class Mixer(RandomizableTransform):

    def __init__(self, batch_size: int, alpha: float = 1.0) -> None:
        """
        Mixer is a base class providing the basic logic for the mixup-class of
        augmentations. In all cases, we need to sample the mixing weights for each
        sample (lambda in the notation used in the papers). Also, pairs of samples
        being mixed are picked by randomly shuffling the batch samples.

        Args:
            batch_size (int): number of samples per batch. That is, samples are expected tp
                be of size batchsize x channels [x depth] x height x width.
            alpha (float, optional): mixing weights are sampled from the Beta(alpha, alpha)
                distribution. Defaults to 1.0, the uniform distribution.
        """
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"Expected positive number, but got {alpha = }")
        self.alpha = alpha
        self.batch_size = batch_size

    @abstractmethod
    def apply(self, data: torch.Tensor):
        raise NotImplementedError()

    def randomize(self, data=None) -> None:
        """
        Sometimes you need may to apply the same transform to different tensors.
        The idea is to get a sample and then apply it with apply() as often
        as needed. You need to call this method everytime you apply the transform to a new
        batch.
        """
        self._params = (
            torch.from_numpy(self.R.beta(self.alpha, self.alpha, self.batch_size)).type(torch.float32),
            self.R.permutation(self.batch_size),
        )


class MixUp(Mixer):
    """MixUp as described in:
    Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz.
    mixup: Beyond Empirical Risk Minimization, ICLR 2018

    Class derived from :py:class:`monai.transforms.Mixer`. See corresponding
    documentation for details on the constructor parameters.
    """

    def apply(self, data: torch.Tensor):
        weight, perm = self._params
        nsamples, *dims = data.shape
        if len(weight) != nsamples:
            raise ValueError(f"Expected batch of size: {len(weight)}, but got {nsamples}")

        if len(dims) not in [3, 4]:
            raise ValueError("Unexpected number of dimensions")

        mixweight = weight[(Ellipsis,) + (None,) * len(dims)]
        return mixweight * data + (1 - mixweight) * data[perm, ...]

    def __call__(self, data: torch.Tensor, labels: torch.Tensor | None = None):
        self.randomize()
        if labels is None:
            return self.apply(data)
        return self.apply(data), self.apply(labels)


class CutMix(Mixer):
    """CutMix augmentation as described in:
        Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon Yoo.
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features,
        ICCV 2019

        Class derived from :py:class:`monai.transforms.Mixer`. See corresponding
        documentation for details on the constructor parameters. Here, alpha not only determines
        the mixing weight but also the size of the random rectangles used during for mixing.
        Please refer to the paper for details.

        The most common use case is something close to:

    .. code-block:: python

        cm = CutMix(batch_size=8, alpha=0.5)
        for batch in loader:
            images, labels = batch
            augimg, auglabels = cm(images, labels)
            output = model(augimg)
            loss = loss_function(output, auglabels)
            ...

    """

    def apply(self, data: torch.Tensor):
        weights, perm = self._params
        nsamples, _, *dims = data.shape
        if len(weights) != nsamples:
            raise ValueError(f"Expected batch of size: {len(weights)}, but got {nsamples}")

        mask = torch.ones_like(data)
        for s, weight in enumerate(weights):
            coords = [torch.randint(0, d, size=(1,)) for d in dims]
            lengths = [d * sqrt(1 - weight) for d in dims]
            idx = [slice(None)] + [slice(c, min(ceil(c + ln), d)) for c, ln, d in zip(coords, lengths, dims)]
            mask[s][idx] = 0

        return mask * data + (1 - mask) * data[perm, ...]

    def apply_on_labels(self, labels: torch.Tensor):
        weights, perm = self._params
        nsamples, *dims = labels.shape
        if len(weights) != nsamples:
            raise ValueError(f"Expected batch of size: {len(weights)}, but got {nsamples}")

        mixweight = weights[(Ellipsis,) + (None,) * len(dims)]
        return mixweight * labels + (1 - mixweight) * labels[perm, ...]

    def __call__(self, data: torch.Tensor, labels: torch.Tensor | None = None):
        self.randomize()
        augmented = self.apply(data)
        return (augmented, self.apply_on_labels(labels)) if labels is not None else augmented


class CutOut(Mixer):
    """Cutout as described in the paper:
    Terrance DeVries, Graham W. Taylor.
    Improved Regularization of Convolutional Neural Networks with Cutout,
    arXiv:1708.04552

    Class derived from :py:class:`monai.transforms.Mixer`. See corresponding
    documentation for details on the constructor parameters. Here, alpha not only determines
    the mixing weight but also the size of the random rectangles being cut put.
    Please refer to the paper for details.
    """

    def apply(self, data: torch.Tensor):
        weights, _ = self._params
        nsamples, _, *dims = data.shape
        if len(weights) != nsamples:
            raise ValueError(f"Expected batch of size: {len(weights)}, but got {nsamples}")

        mask = torch.ones_like(data)
        for s, weight in enumerate(weights):
            coords = [torch.randint(0, d, size=(1,)) for d in dims]
            lengths = [d * sqrt(1 - weight) for d in dims]
            idx = [slice(None)] + [slice(c, min(ceil(c + ln), d)) for c, ln, d in zip(coords, lengths, dims)]
            mask[s][idx] = 0

        return mask * data

    def __call__(self, data: torch.Tensor):
        self.randomize()
        return self.apply(data)
