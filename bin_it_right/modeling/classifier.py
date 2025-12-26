import os
from typing import Dict

from bin_it_right.modeling.pytorch import get_device, GarbageClassificationCNN, GarbageClassificationPretrained
from bin_it_right.dataset import DataFrameInitializer

import torch
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

@dataclass
class ClassificationResponse:
    predicted_class: str
    classes_distribution: Dict[str, float]

@dataclass
class TrashClassifier:
    base_path: str

    def classify(self, model_type: str, image: Image):
        device = get_device()
        if model_type == 'raw':
            model = GarbageClassificationCNN(num_classes=6)
            model_name = 'best_model_raw.pt'
        elif model_type == 'pretrained':
            model = GarbageClassificationPretrained()
            model_name = 'best_model_pretrained.pt'

        checkpoint = torch.load(
            os.path.join(self.base_path, model_name),
            map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        val_transform = self.get_val_transform()
        x = val_transform(image).unsqueeze(0).to(device)
        classes = DataFrameInitializer.load_classes()

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0] 

        pred_idx = int(torch.argmax(probs).item())
        pred_class = classes.get(pred_idx, f"class_{pred_idx}")
        result = {}
        for idx, p in enumerate(probs.cpu().numpy()):
            name = classes.get(idx, f"class_{idx}")
            result[name] = p

        return ClassificationResponse(
            predicted_class=pred_class,
            classes_distribution=result
        )

    def get_val_transform(self) -> transforms.Compose:
        input_size = 200

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])