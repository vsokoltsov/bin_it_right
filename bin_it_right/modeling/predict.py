import os
import click
from torchvision import transforms
from bin_it_right.modeling.pytorch import get_device, GarbageClassificationCNN
import torch
from PIL import Image
from bin_it_right.dataset import DataFrameInitializer
import torch.nn.functional as F

def get_val_transform():
    input_size = 200

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

@click.command()
@click.argument("model_path")
@click.argument("image_path")
def predict_image(model_path, image_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = get_device()
    click.echo(f"Using device: {device}")

    model = GarbageClassificationCNN(num_classes=6)
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