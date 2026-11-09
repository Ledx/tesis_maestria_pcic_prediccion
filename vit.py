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

# Clases
CLASS_NAMES = ["Des", "Nodes"]
NUM_CLASSES = 2

# Número de frames utilizados por video
NUM_FRAMES = 16

# Resolución de entrada
IMAGE_SIZE = 224

# Entrenamiento
BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

# ViT
PATCH_SIZE = 16
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
MLP_DIM = 512
DROPOUT = 0.1

# Random seed
SEED = 42

# Número de workers
NUM_WORKERS = 4

# ============================================================
# REPRODUCIBILIDAD
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------
fecha = datetime.now()
dir_salida = str(fecha.year) + str(fecha.month) +str(fecha.day) + str("-") + str(fecha.hour) + str(fecha.minute) + str(fecha.second) + str("_VIT")
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
            raise FileNotFoundError(
                f"No existe la carpeta: {class_dir}"
            )

        for extension in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
            for video_path in class_dir.glob(extension):
                videos.append((str(video_path), class_id))

    return videos


# Transformaciones
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
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

        if len(frames) == 0:
            raise RuntimeError(f"No se pudo leer el video: {video_path}")

        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy())

        frames = frames[:self.num_frames]
        return frames

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label = self.labels[index]
        frames = self.read_video(video_path)

        processed_frames = []
        for frame in frames:
            if self.transform is not None:
                frame = self.transform(frame)
            processed_frames.append(frame)

        video = torch.stack(processed_frames)
        label = torch.tensor(label, dtype=torch.long)
        return video, label


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self, image_size=224, patch_size=16, num_classes=2,
        embed_dim=256, num_heads=8, num_layers=6, mlp_dim=512, dropout=0.1
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            image_size=image_size, patch_size=patch_size, embed_dim=embed_dim
        )
        num_patches = self.patch_embedding.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)
        return logits


class VideoViT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.vit = VisionTransformer(
            image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=EMBED_DIM,
            embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
            mlp_dim=MLP_DIM, dropout=DROPOUT
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, video):
        B, T, C, H, W = video.shape
        video = video.view(B * T, C, H, W)
        frame_features = self.vit(video)
        frame_features = frame_features.view(B, T, EMBED_DIM)
        video_features = frame_features.mean(dim=1)
        output = self.classifier(video_features)
        return output


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


# ============================================================
# PUNTO DE ENTRADA PRINCIPAL (SOLUCIÓN AL ERROR)
# ============================================================

if __name__ == '__main__':

    create_output_directory()
    print("=" * 70)
    print("DEVICE:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 70)

    # 1. Obtener videos
    videos = get_video_paths(DATASET_DIR)
    print(f"\nNúmero total de videos: {len(videos)}")
    for class_id, class_name in enumerate(CLASS_NAMES):
        count = sum(1 for _, label in videos if label == class_id)
        print(f"Clase {class_id} ({class_name}): {count} videos")

    # 2. División de dataset
    video_paths = [x[0] for x in videos]
    labels = [x[1] for x in videos]

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        video_paths, labels, test_size=0.15, stratify=labels, random_state=SEED
    )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.17647, stratify=train_val_labels, random_state=SEED
    )

    print("\n" + "=" * 70)
    print("DIVISIÓN DEL DATASET")
    print("=" * 70)
    print(f"Train      : {len(train_paths)} videos")
    print(f"Validation : {len(val_paths)} videos")
    print(f"Test       : {len(test_paths)} videos")

    # 3. Datasets y DataLoaders
    train_dataset = VideoDataset(train_paths, train_labels, NUM_FRAMES, train_transform)
    val_dataset = VideoDataset(val_paths, val_labels, NUM_FRAMES, test_transform)
    test_dataset = VideoDataset(test_paths, test_labels, NUM_FRAMES, test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # 4. Crear modelo, optimizer, scheduler, criterion
    model = VideoViT(num_classes=NUM_CLASSES).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 70)
    print("MODELO")
    print("=" * 70)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5. Entrenamiento
    best_f1 = 0.0
    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_f1": [], "val_f1": []
    }

    print("\n" + "=" * 70)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 70)

    for epoch in range(EPOCHS):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = validate(model, val_loader, criterion)
        scheduler.step()

        train_acc, _, _, train_f1 = train_metrics
        val_acc, _, _, val_f1 = val_metrics

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc:  {train_acc:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"Train F1:   {train_f1:.4f} | Val F1:     {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1
            }, OUTPUT_DIR / "best_vit_video.pth")
            print("✓ Mejor modelo guardado")

    # 6. Evaluación en Test
    print("\n" + "=" * 70)
    print("CARGANDO MEJOR MODELO Y EVALUANDO")
    print("=" * 70)

    checkpoint = torch.load(OUTPUT_DIR / "best_vit_video.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = test_model(model, test_loader)

    test_accuracy = accuracy_score(y_true, y_pred)
    test_precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    test_recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    test_f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

    print(f"\nAccuracy : {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")
    print(f"F1 Score : {test_f1:.4f}")

    print("\nMATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(y_true, y_pred))

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # Guardar el historial en JSON
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    # ============================================================
    # GENERACIÓN DE GRÁFICAS (CURVAS Y MATRIZ DE CONFUSIÓN)
    # ============================================================
    print("\nGenerando gráficas...")


    # 1. Guardar Matriz de Confusión como Imagen
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES)
    plt.title("Matriz de Confusión - Evaluación Test")
    plt.xlabel("Predicción del Modelo")
    plt.ylabel("Etiqueta Real")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # 2. Guardar Curvas de Pérdida (Loss)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", color='blue', linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", color='orange', linewidth=2)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida (Loss)")
    plt.title("Curva de Pérdida en Entrenamiento y Validación")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=200)
    plt.close()

    # 3. Guardar Curvas de Precisión (Accuracy)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_accuracy"], label="Train Accuracy", color='green', linewidth=2)
    plt.plot(history["val_accuracy"], label="Validation Accuracy", color='red', linewidth=2)
    plt.xlabel("Épocas")
    plt.ylabel("Precisión (Accuracy)")
    plt.title("Curva de Precisión en Entrenamiento y Validación")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=200)
    plt.close()

    print("\n" + "=" * 70)
    print("EJECUCIÓN TERMINADA CON ÉXITO")
    print("=" * 70)
    print("Modelo guardado como       : best_vit_video.pth")
    print("Historial guardado en      : training_history.json")
    print("Curva de Pérdida guardada  : loss_curve.png")
    print("Curva de Accuracy guardada : accuracy_curve.png")
    print("Matriz de Confusión visual : confusion_matrix.png")