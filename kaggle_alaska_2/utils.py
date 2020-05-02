from pathlib import Path
from typing import Union, Dict, List, Tuple

folder2label = {"Cover": 0, "JMiPOD": 1, "JUNIWARD": 1, "UERD": 1}


def get_id2_file_paths(file_path: Union[str, Path]) -> Dict[str, Path]:
    return {file.stem: file for file in Path(file_path).glob("*")}


def get_samples(image_path: Path) -> List[Tuple[Path, int]]:
    label = folder2label[image_path.name]
    result = [(x, label) for x in sorted(image_path.glob("*.jpg"))]
    return result
