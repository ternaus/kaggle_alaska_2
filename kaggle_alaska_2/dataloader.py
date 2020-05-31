import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import albumentations as albu
import torch
from iglovikov_helper_functions.utils.image_utils import load_rgb
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils.data import Dataset


class Alaska2Dataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform: albu.Compose, stratified: bool = False) -> None:
        self.samples = samples
        self.transform = transform
        self.stratified = stratified
        self.grouped = self.group_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def group_samples(self) -> Dict[int, List[Path]]:
        result: Dict[int, List[Path]] = {0: [], 1: []}

        for image_path, target in self.samples:
            result[target] += [image_path]
        return result

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.stratified:
            target = random.choice([0, 1])
            image_paths = self.grouped[target]
            image_path = random.choice(image_paths)
        else:
            image_path, target = self.samples[idx]

        image = load_rgb(image_path, lib="jpeg4py")

        # apply augmentations
        image = self.transform(image=image)["image"]

        return {
            "image_id": image_path.stem,
            "features": tensor_from_rgb_image(image),
            "targets": torch.Tensor([target]),
        }


class AlaskaTest2Dataset(Dataset):
    def __init__(self, samples: List[Path], transform: albu.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.samples[idx]

        image = load_rgb(image_path, lib="jpeg4py")

        # apply augmentations
        image = self.transform(image=image)["image"]

        return {"image_id": image_path.stem, "features": tensor_from_rgb_image(image)}
