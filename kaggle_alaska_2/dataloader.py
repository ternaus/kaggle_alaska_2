from pathlib import Path
from typing import List, Tuple, Dict, Any

import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils.data import Dataset


class Alaska2Dataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, Path]], transform: albu.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path, target = self.samples[idx]

        image = load_rgb(image_path)

        # apply augmentations
        image = self.transform(image=image)["image"]

        return {"image_id": image_path.stem, "features": tensor_from_rgb_image(image), "targets": target}


class AlaskaTest2Dataset(Dataset):
    def __init__(self, samples: List[Path], transform: albu.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.samples[idx]

        image = load_rgb(image_path)

        # apply augmentations
        image = self.transform(image=image)["image"]

        return {"image_id": image_path.stem, "features": tensor_from_rgb_image(image)}
