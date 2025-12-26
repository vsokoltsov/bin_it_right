import os
import click
from torchvision import transforms
from bin_it_right.modeling.pytorch import get_device, GarbageClassificationCNN, GarbageClassificationPretrained
from bin_it_right.modeling.image_transformers import get_val_transform
import torch
from PIL import Image
from bin_it_right.dataset import DataFrameInitializer
import torch.nn.functional as F

@click.command()
@click.argument("model_path")
@click.argument("image_path")
@click.option('--model-type', default='raw', help='Type of the model. Available options are: `raw` and `pretrained`')
def predict_image(model_path, image_path, model_type):
    if model_type not in ['raw', 'pretrained']:
        raise ValueError(f"'model_type' parameter has wrong value {model_type}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = get_device()
    click.echo(f"Using device: {device}")

    if model_type == 'raw':
        model = GarbageClassificationCNN(num_classes=6)
    elif model_type == 'pretrained':
        model = GarbageClassificationPretrained()

    checkpoint = torch.load(
        model_path,
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_transform = get_val_transform()
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    x = val_transform(img).unsqueeze(0).to(device)

    classes = DataFrameInitializer.load_classes()

    with torch.no_grad():
        logits = model(x)             # [1, num_classes]
        probs = F.softmax(logits, dim=1)[0] 

    pred_idx = int(torch.argmax(probs).item())
    pred_class = classes.get(pred_idx, f"class_{pred_idx}")
    click.echo(f"Image: {image_path}")
    click.echo(f"Predicted class index: {pred_idx}")
    click.echo(f"Predicted class name:  {pred_class}")
    click.echo("Probabilities:")
    for idx, p in enumerate(probs.cpu().numpy()):
        name = classes.get(idx, f"class_{idx}")
        click.echo(f"  {idx}: {name:10s} -> {p:.4f}")

if __name__ == '__main__':
    predict_image()