import os
import random
import numpy as np
import cv2
import json

from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_DIR = "dataset"
CLASS_NAMES = ["Des", "Nodes"]
NUM_CLASSES = 2
NUM_FRAMES = 16 # MViT v2 s por defecto usa 16 frames
IMAGE_SIZE = 224

BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

SEED = 42
NUM_WORKERS = 4

# ============================================================
# REPRODUCIBILIDAD
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------
fecha = datetime.now()
dir_salida = str(fecha.year) + str(fecha.month) +str(fecha.day) + str("-") + str(fecha.hour) + str(fecha.minute) + str(fecha.second) + str("_MVIT")
OUTPUT_DIR = Path(dir_salida)

# ============================================================
# FUNCIONES Y CLASES
# ============================================================
def create_output_directory():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_video_paths(dataset_dir):
    dataset_dir = Path(dataset_dir)
    videos = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            continue
        for extension in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
            for video_path in class_dir.glob(extension):
                videos.append((str(video_path), class_id))
    return videos

# Transformaciones (MViT espera [C, T, H, W])
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]) # Kinetics mean/std
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def read_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        if total_frames > 0:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        else:
            indices = np.zeros(self.num_frames, dtype=int)
        current_index = 0
        target_indices = set(indices.tolist())
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_index in target_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            current_index += 1
        cap.release()
        while len(frames) < self.num_frames:
            if len(frames) > 0:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
        return frames[:self.num_frames]

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label = self.labels[index]
        frames = self.read_video(video_path)
        processed_frames = []
        seed = np.random.randint(2147483647)
        for frame in frames:
            if self.transform is not None:
                torch.manual_seed(seed)
                frame = self.transform(frame)
            processed_frames.append(frame)
        video = torch.stack(processed_frames) # [T, C, H, W]
        # MViT espera [C, T, H, W]
        video = video.permute(1, 0, 2, 3) 
        label = torch.tensor(label, dtype=torch.long)
        return video, label

# ============================================================
# MViT MODEL (Pre-entrenado)
# ============================================================
class VideoMViTClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        weights = MViT_V2_S_Weights.DEFAULT
        self.backbone = mvit_v2_s(weights=weights)
        
        # Sustituimos la cabeza por una de 2 clases
        in_features = self.backbone.head[1].in_features
        self.backbone.head[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    return accuracy, precision, recall, f1

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_predictions, all_labels = [], []
    for videos, labels in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(videos)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predictions = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * videos.size(0)
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_predictions)
    return epoch_loss, metrics

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_predictions, all_labels = [], []
    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)
        outputs = model(videos)
        loss = criterion(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * videos.size(0)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_predictions)
    return epoch_loss, metrics

@torch.no_grad()
def test_model(model, loader):
    model.eval()
    all_predictions, all_labels = [], []
    for videos, labels in loader:
        videos = videos.to(device)
        outputs = model(videos)
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_predictions)

def plot_and_save_curves(history, y_true, y_pred, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label='Train Loss')
    plt.plot(epochs, history["val_loss"], label='Validation Loss')
    plt.title('Pérdida durante Entrenamiento y Validación')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()

    # 2. Accuracy Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_accuracy"], label='Train Accuracy')
    plt.plot(epochs, history["val_accuracy"], label='Validation Accuracy')
    plt.title('Precisión (Accuracy) durante Entrenamiento y Validación')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "accuracy_curve.png", dpi=150)
    plt.close()
    
    # 3. F1 Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_f1"], label='Train F1 Score')
    plt.plot(epochs, history["val_f1"], label='Validation F1 Score')
    plt.title('F1 Score durante Entrenamiento y Validación')
    plt.xlabel('Época')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "f1_curve.png", dpi=150)
    plt.close()

    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Matriz de Confusión - Test")
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Real")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

if __name__ == '__main__':
    create_output_directory()
    print("=" * 70)
    print("DEVICE:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 70)

    videos = get_video_paths(DATASET_DIR)
    print(f"\nNúmero total de videos: {len(videos)}")
    video_paths = [x[0] for x in videos]
    labels = [x[1] for x in videos]

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        video_paths, labels, test_size=0.15, stratify=labels, random_state=SEED
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.17647, stratify=train_val_labels, random_state=SEED
    )

    train_dataset = VideoDataset(train_paths, train_labels, NUM_FRAMES, train_transform)
    val_dataset = VideoDataset(val_paths, val_labels, NUM_FRAMES, test_transform)
    test_dataset = VideoDataset(test_paths, test_labels, NUM_FRAMES, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Crear el modelo
    model = VideoMViTClassifier(num_classes=NUM_CLASSES).to(device)

    # Congelamiento Parcial (Congelar los primeros 12 bloques de 16)
    for i, block in enumerate(model.backbone.blocks):
        if i < 12:
            for param in block.parameters():
                param.requires_grad = False

    # Calcular e imprimir parámetros (Totales vs Entrenables)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParámetros totales MViT: {total_params:,}")
    print(f"Parámetros entrenables:  {trainable_params:,}") # <- Esto te confirmará el congelamiento

    # Definir función de pérdida (NUNCA QUITARLA)
    criterion = nn.CrossEntropyLoss()
    
    # Definir el optimizador (ATENCIÓN: Pasar solo los parámetros entrenables)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), # <- Cambio crucial aquí
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [], "train_f1": [], "val_f1": []}

    print("\nINICIANDO ENTRENAMIENTO...")
    for epoch in range(EPOCHS):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = validate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_metrics[0])
        history["val_accuracy"].append(val_metrics[0])
        history["train_f1"].append(train_metrics[3])
        history["val_f1"].append(val_metrics[3])

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc:  {train_metrics[0]:.4f} | Val Acc:  {val_metrics[0]:.4f}")
        print(f"Train F1:   {train_metrics[3]:.4f} | Val F1:   {val_metrics[3]:.4f}")

        if val_metrics[3] > best_f1:
            best_f1 = val_metrics[3]
            torch.save({"model_state_dict": model.state_dict()}, OUTPUT_DIR / "best_mvit.pth")
            print("✓ Mejor modelo guardado")
            
    print("\nENTRENAMIENTO TERMINADO")

    # Guardar historial
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    # Evaluación final y gráficas
    checkpoint = torch.load(OUTPUT_DIR / "best_mvit.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred = test_model(model, test_loader)
    
    print("\nRESULTADOS FINALES EN TEST")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    print("\nGenerando gráficas y reportes...")
    plot_and_save_curves(history, y_true, y_pred, OUTPUT_DIR)
    
    print("\nArchivos guardados en:", OUTPUT_DIR)
