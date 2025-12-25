import click

from bin_it_right.dataset import DataFrameInitializer, GarbageClassificationDataset
from bin_it_right.modeling.pytorch import get_device, GarbageClassificationCNN, GarbageClassificationPretrained

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Dict, List

def get_transformers():
    input_size = 200

    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return {
        'train': train_transforms,
        'val': val_transforms
    }

def train_eval_model(
    model: nn.Module,
    num_epochs: int,
    best_model_path: str,
    optimizer,
    criterion,
    device,
    train_loader,
    val_loader
) -> pd.DataFrame:
    rows = []
    best_val_acc = 0.0
    best_model_state = None

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for _, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }
            torch.save(best_model_state, best_model_path)
            print(f"  >> New best model! Val Acc = {val_acc:.4f}")

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}')
        scheduler.step(val_loss)

        rows.append((epoch, train_loss, train_acc, val_loss, val_acc))
        
    return pd.DataFrame(rows, columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

def test_model(model: nn.Module, test_loader, device) -> Dict[str, List[float]]:
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    return {
        'labels': all_labels,
        'preds': all_preds
    }


@click.command()
@click.argument('dataset_path')
@click.argument('model_path')
@click.option('--model', default='raw', help='Type of the model. Available options are: `raw` and `pretrained`')
@click.option('--model-provider', default='pytorch')
@click.option('--epochs', default=60)
def train_classifier(dataset_path, model_path, model, model_provider, epochs):
    click.echo(f"Dataset is located in {dataset_path}")
    dfi = DataFrameInitializer(base_path = dataset_path)
    df_train  = dfi.from_file("one-indexed-files-notrash_train.txt")
    df_val  = dfi.from_file("one-indexed-files-notrash_val.txt")
    df_test  = dfi.from_file("one-indexed-files-notrash_test.txt")

    trnsfrms = get_transformers()

    dataset_train = GarbageClassificationDataset(df=df_train, transform=trnsfrms['train'])
    dataset_val = GarbageClassificationDataset(df=df_val, transform=trnsfrms['val'])
    dataset_test = GarbageClassificationDataset(df=df_test, transform=trnsfrms['val'])

    train_loader = DataLoader(dataset_train, batch_size=32)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

    device = get_device()
    model = GarbageClassificationCNN()
    model.to(device)

    class_counts = df_train["class"].value_counts().sort_index()

    alpha = 0.5
    class_weights = (1.0 / class_counts) ** alpha
    # normalize, so that average weight is ~1
    class_weights = class_weights / class_weights.mean()

    # cast to tensor
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    train_eval_model(
        model=model,
        num_epochs=epochs,
        best_model_path=model_path,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader
    )

    model_test = GarbageClassificationCNN(num_classes=6)
    checkpoint = torch.load(
        model_path,
        map_location=device
    )
    model_test.load_state_dict(checkpoint["model_state_dict"])
    model_test.to(device)
    model_test.eval()

    test_data = test_model(model=model_test, test_loader=test_loader, device=device)

    click.echo(f"Test accuracy: {(test_data['labels'] == test_data['preds']).mean()}")
    

if __name__ == '__main__':
    train_classifier()