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
from torchvision.models import vit_b_16, ViT_B_16_Weights


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

# ViViT (Factorized Encoder)
PATCH_SIZE = 16
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS_SPATIAL = 4
NUM_LAYERS_TEMPORAL = 4
MLP_DIM = 512
DROPOUT = 0.3

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------
fecha = datetime.now()
dir_salida = str(fecha.year) + str(fecha.month) +str(fecha.day) + str("-") + str(fecha.hour) + str(fecha.minute) + str(fecha.second) + str("_VIVIT")
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
            raise FileNotFoundError(f"No existe la carpeta: {class_dir}")
        for extension in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
            for video_path in class_dir.glob(extension):
                videos.append((str(video_path), class_id))
    return videos

# ==========================================
# AUMENTO DE DATOS AGRESIVO (ENTRENAMIENTO)
# ==========================================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # Recortes aleatorios grandes que obligan a la red a no depender de los bordes
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)), 
    # Volteo y rotaciones
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    # Distorsión agresiva de color y contraste
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        seed = np.random.randint(2147483647)
        for frame in frames:
            if self.transform is not None:
                torch.manual_seed(seed)
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
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x



class ViViT(nn.Module):
    def __init__(self, num_frames=16, num_classes=2, num_temporal_layers=2, freeze_spatial=True):
        super().__init__()
        
        # ==========================================
        # 1. ENCODER ESPACIAL (Preentrenado en ImageNet)
        # ==========================================
        self.spatial_vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        embed_dim = 768 # Dimensión interna del vit_b_16
        
        # Reemplazamos la cabeza final (1000 clases) por Identity 
        # para obtener directamente el vector de características de 768 dims
        self.spatial_vit.heads = nn.Identity()

        # Congelar los pesos espaciales (Recomendado para datasets pequeños)
        if freeze_spatial:
            for param in self.spatial_vit.parameters():
                param.requires_grad = False
                
        # ==========================================
        # 2. ENCODER TEMPORAL (Entrenado desde cero)
        # ==========================================
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, num_frames + 1, embed_dim))
        
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            dim_feedforward=2048,
            dropout=0.4, # Dropout alto para regularizar
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=num_temporal_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.4)
        
        # ==========================================
        # 3. CABEZA DE CLASIFICACIÓN
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64), 
            nn.GELU(),
            nn.Dropout(0.6), 
            nn.Linear(64, num_classes)
        )

    def forward(self, video):
        # video shape: [Batch, Tiempo, Canales, Alto, Ancho]
        B, T, C, H, W = video.shape
        
        # --- SPATIAL ENCODER ---
        x = video.view(B * T, C, H, W)
        
        # Extraemos las características de cada frame. 
        # Salida: [Batch * Tiempo, 768]
        spatial_features = self.spatial_vit(x) 
        
        # --- TEMPORAL ENCODER ---
        # Reconstruimos la dimensión temporal: [Batch, Tiempo, 768]
        x_temp = spatial_features.view(B, T, -1) 
        
        # Añadimos el token CLS temporal y las posiciones temporales
        temp_cls_tokens = self.temporal_cls_token.expand(B, -1, -1)
        x_temp = torch.cat((temp_cls_tokens, x_temp), dim=1) # [B, T+1, 768]
        x_temp = x_temp + self.temporal_pos_embedding
        x_temp = self.dropout(x_temp)
        
        # Modelamos la relación entre los fotogramas
        x_temp = self.temporal_transformer(x_temp) 
        
        # Extraemos la representación de todo el video (Token CLS temporal)
        video_features = x_temp[:, 0] # [B, 768]
        video_features = self.norm(video_features)
        
        # --- CLASIFICACIÓN FINAl ---
        logits = self.classifier(video_features)
        return logits

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

    # Instanciar el nuevo modelo
    model = ViViT(
        num_frames=NUM_FRAMES, 
        num_classes=NUM_CLASSES,
        num_temporal_layers=2, # 2 capas son suficientes para 16 frames
        freeze_spatial=True    # Evita el colapso inicial
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    
    # ATENCIÓN AQUÍ: Pasamos al optimizador SOLO los parámetros que requieren gradiente
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0.0
    # ==========================================
    # CONFIGURACIÓN DE EARLY STOPPING ESTRICTO
    # ==========================================
    patience = 5  # Número máximo de épocas sin mejora permitidas
    patience_counter = 0
    best_val_loss = float('inf') # Ahora guardamos el modelo minimizando el Loss
    
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

        # ==========================================
        # LÓGICA DE DETENCIÓN
        # ==========================================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 # Reiniciar contador al mejorar
            
            # Guardar el modelo basándose en la menor pérdida
            torch.save({"model_state_dict": model.state_dict()}, OUTPUT_DIR / "best_vivit.pth")
            print(f"✓ Mejor modelo guardado (Val Loss bajó a {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⚠ Val Loss no mejoró. Paciencia: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("\n" + "="*50)
                print(f"[!] EARLY STOPPING ACTIVADO en la época {epoch + 1}.")
                print(f"[!] La red empezó a sobreajustarse. Entrenamiento detenido.")
                print("="*50)
                break # Rompe el ciclo y pasa directamente a la evaluación final

    print("\nENTRENAMIENTO TERMINADO")

    # Guardar historial
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    # Evaluación final y gráficas
    checkpoint = torch.load(OUTPUT_DIR / "best_vivit.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred = test_model(model, test_loader)
    
    print("\nRESULTADOS FINALES EN TEST")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    print("\nGenerando gráficas y reportes...")
    plot_and_save_curves(history, y_true, y_pred, OUTPUT_DIR)
    
    print("\nArchivos guardados en:", OUTPUT_DIR)
