import os
from dataclasses import dataclass
import yaml
from typing import Dict, Optional
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

@dataclass
class DataFrameInitializer:
    base_path: str
    classes_path: Optional[str] = None

    CLASSES = {
        1: 'glass',
        2: 'paper',
        3: 'cardboard',
        4: 'plastic',
        5: 'metal',
        6: 'trash'
    }

    def from_file(self, filepath: str) -> pd.DataFrame:
        rows = []
        classes = self._load_classes()
        with open(os.path.join(self.base_path, filepath), "r") as f:
            for line in f:
                img, cls = line.strip().split(" ")
                class_idx = int(cls) - 1
                cls_name = classes[int(cls)]
                full_path = os.path.join(self.base_path, "garbage_classification", cls_name, img)
                rows.append((img, int(cls), class_idx, cls_name, full_path))

        df = pd.DataFrame(rows, columns=["image_name", "class", "class_idx", "class_name", "image_path"])
        return df


    def _load_classes(self) -> Dict[int, str]:
        if not self.classes_path:
            return self.CLASSES

        with open(self.classes_path, "r") as f:
            classes = yaml.safe_load(f)

        return classes

    @classmethod
    def load_classes(self):
        return {k-1: v for k, v in self.CLASSES.items()}

class GarbageClassificationDataset(Dataset):

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img = Image.open(item['image_path']).convert('RGB')
        label = torch.tensor(int(item['class_idx']), dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label