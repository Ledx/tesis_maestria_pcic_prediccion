"""
Entrenamiento binario con Video Swin Transformer 3D de Torchvision.
Estructura esperada:
dataset/
├── Des/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
└── Nodes/
    ├── video_101.mp4
    ├── video_102.mp4
    └── ...
Clases:
    Des   -> 0
    Nodes -> 1
Instalación:
    pip install torch torchvision opencv-python numpy matplotlib scikit-learn tqdm
Ejecución:
    python train_video_swin3d.py
"""
from pathlib import Path
import json
import random
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import (
    swin3d_t,
    Swin3D_T_Weights,
)
# ============================================================
# CONFIGURACIÓN
# ============================================================
DATASET_DIR = Path("dataset")
fecha = datetime.now()
dir_salida = str(fecha.year) + str(fecha.month) +str(fecha.day) + str("-") + str(fecha.hour) + str(fecha.minute) + str(fecha.second)  + str("_SWIN")
OUTPUT_DIR = Path(dir_salida)
CLASS_NAMES = ["Des", "Nodes"]
CLASS_TO_INDEX = {
    "Des": 0,
    "Nodes": 1,
}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".mpeg",
    ".mpg",
}
IMAGE_SIZE = 224
# Número de fotogramas usados por clip
NUM_FRAMES = 32
# Si el video tiene menos de NUM_FRAMES fotogramas,
# se repiten fotogramas.
MIN_VIDEO_FRAMES = 1
BATCH_SIZE = 2
NUM_WORKERS = 0
NUM_EPOCHS = 25
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
# Durante las primeras épocas se congela el backbone.
FREEZE_BACKBONE = True
# Número de épocas con backbone congelado.
FREEZE_EPOCHS = 15
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
# ============================================================
# REPRODUCIBILIDAD
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ============================================================
# UTILIDADES
# ============================================================
def create_output_directory():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
def get_video_files():
    """
    Obtiene todos los videos de las carpetas Des y Nodes.
    """
    videos = []
    for class_name in CLASS_NAMES:
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"No existe la carpeta: {class_dir}"
            )
        for video_path in sorted(class_dir.rglob("*")):
            if video_path.is_file():
                if video_path.suffix.lower() in VIDEO_EXTENSIONS:
                    videos.append(
                        {
                            "path": str(video_path),
                            "label": CLASS_TO_INDEX[class_name],
                            "class_name": class_name,
                        }
                    )
    if len(videos) == 0:
        raise RuntimeError(
            f"No se encontraron videos en {DATASET_DIR}"
        )
    return videos
def print_dataset_summary(videos, name):
    labels = [item["label"] for item in videos]
    print(f"\n{name}")
    print("-" * 50)
    print(f"Total de videos: {len(videos)}")
    for class_name, class_index in CLASS_TO_INDEX.items():
        count = labels.count(class_index)
        print(f"{class_name}: {count}")
def read_video_frames(
    video_path: str,
    num_frames: int = NUM_FRAMES,
    image_size: int = IMAGE_SIZE,
):
    """
    Lee un video y devuelve un tensor con forma:
        [T, C, H, W]
    donde:
        T = número de frames
        C = 3 canales RGB
        H = alto
        W = ancho
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(
            f"No se pudo abrir el video: {video_path}"
        )
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(
            f"El video no contiene frames válidos: {video_path}"
        )
    # Seleccionamos frames uniformemente distribuidos.
    selected_indices = np.linspace(
        0,
        total_frames - 1,
        num=num_frames,
        dtype=np.int32,
    )
    frames = []
    for frame_index in selected_indices:
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index),
        )
        success, frame = capture.read()
        if not success:
            continue
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )
        frame = cv2.resize(
            frame,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
        frame = frame.astype(np.float32) / 255.0
        # HWC -> CHW
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)
    capture.release()
    if len(frames) == 0:
        raise RuntimeError(
            f"No se pudieron leer frames de: {video_path}"
        )
    # Si faltan frames, repetimos el último.
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    frames = frames[:num_frames]
    video = np.stack(frames, axis=0)
    # Normalización ImageNet.
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    ).reshape(1, 3, 1, 1)
    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    ).reshape(1, 3, 1, 1)
    video = (video - mean) / std
    # [T, C, H, W]
    return torch.tensor(
        video,
        dtype=torch.float32,
    )
# ============================================================
# DATASET
# ============================================================
class VideoDataset(Dataset):
    def __init__(
        self,
        samples,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
    ):
        self.samples = samples
        self.num_frames = num_frames
        self.image_size = image_size
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        sample = self.samples[index]
        video_path = sample["path"]
        label = sample["label"]
        try:
            video = read_video_frames(
                video_path=video_path,
                num_frames=self.num_frames,
                image_size=self.image_size,
            )
        except Exception as error:
            raise RuntimeError(
                f"Error leyendo {video_path}: {error}"
            ) from error
        return video, torch.tensor(
            label,
            dtype=torch.long,
        )
# ============================================================
# MODELO
# ============================================================
class VideoSwinClassifier(nn.Module):
    """
    Clasificador basado exclusivamente en Video Swin 3D
    de Torchvision.
    Entrada esperada:
        [B, T, C, H, W]
    Video Swin 3D espera:
        [B, C, T, H, W]
    """
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        freeze_backbone=True,
    ):
        super().__init__()
        if pretrained:
            weights = Swin3D_T_Weights.DEFAULT
        else:
            weights = None
        self.backbone = swin3d_t(
            weights=weights,
            progress=True,
        )
        # El modelo original clasifica 400 clases de Kinetics.
        # Sustituimos la cabeza por una de 2 clases.
        input_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(input_features, num_classes)
        )
        if freeze_backbone:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad = False
            # La nueva cabeza sí debe entrenarse.
            for parameter in self.backbone.head.parameters():
                parameter.requires_grad = True
    def unfreeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
    def forward(self, x):
        # Entrada recibida:
        # [B, T, C, H, W]
        # Video Swin 3D necesita:
        # [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        return self.backbone(x)
# ============================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================
def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    progress_bar = tqdm(
        loader,
        desc="Entrenamiento",
        leave=False,
    )
    for videos, labels in progress_bar:
        videos = videos.to(
            DEVICE,
            non_blocking=True,
        )
        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(
            outputs,
            labels,
        )
        loss.backward()
        optimizer.step()
        predictions = torch.argmax(
            outputs,
            dim=1,
        )
        total_loss += loss.item() * videos.size(0)
        all_labels.extend(
            labels.detach().cpu().numpy().tolist()
        )
        all_predictions.extend(
            predictions.detach().cpu().numpy().tolist()
        )
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )
    epoch_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )
    metrics["loss"] = epoch_loss
    return metrics
@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    description="Validación",
):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    progress_bar = tqdm(
        loader,
        desc=description,
        leave=False,
    )
    for videos, labels in progress_bar:
        videos = videos.to(
            DEVICE,
            non_blocking=True,
        )
        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )
        outputs = model(videos)
        loss = criterion(
            outputs,
            labels,
        )
        predictions = torch.argmax(
            outputs,
            dim=1,
        )
        total_loss += loss.item() * videos.size(0)
        all_labels.extend(
            labels.detach().cpu().numpy().tolist()
        )
        all_predictions.extend(
            predictions.detach().cpu().numpy().tolist()
        )
    epoch_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )
    metrics["loss"] = epoch_loss
    return metrics, all_labels, all_predictions
# ============================================================
# GRÁFICAS
# ============================================================
def save_training_curves(history):
    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs,
        history["train_loss"],
        label="Entrenamiento",
    )
    plt.plot(
        epochs,
        history["val_loss"],
        label="Validación",
    )
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Pérdida durante el entrenamiento")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "loss_curve.png",
        dpi=150,
    )
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Accuracy entrenamiento",
    )
    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Accuracy validación",
    )
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Accuracy durante el entrenamiento")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "accuracy_curve.png",
        dpi=150,
    )
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs,
        history["train_f1"],
        label="F1 entrenamiento",
    )
    plt.plot(
        epochs,
        history["val_f1"],
        label="F1 validación",
    )
    plt.xlabel("Época")
    plt.ylabel("F1 macro")
    plt.title("F1 durante el entrenamiento")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "f1_curve.png",
        dpi=150,
    )
    plt.close()
def save_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(
        y_true,
        y_pred,
    )
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.title("Matriz de confusión - Test")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(
        tick_marks,
        CLASS_NAMES,
        rotation=45,
    )
    plt.yticks(
        tick_marks,
        CLASS_NAMES,
    )
    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )
    plt.ylabel("Etiqueta real")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "confusion_matrix_test.png",
        dpi=150,
    )
    plt.close()
# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(RANDOM_SEED)
    create_output_directory()
    print("=" * 70)
    print("CLASIFICACIÓN BINARIA CON VIDEO SWIN TRANSFORMER 3D")
    print("=" * 70)
    print(f"Dispositivo: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {DATASET_DIR.resolve()}")
    print(f"Frames por clip: {NUM_FRAMES}")
    print(f"Tamaño espacial: {IMAGE_SIZE} x {IMAGE_SIZE}")
    videos = get_video_files()
    print_dataset_summary(
        videos,
        "Dataset completo",
    )
    labels = [
        sample["label"]
        for sample in videos
    ]
    # Primera división:
    # train = 70 %
    # temporal = 30 %
    train_samples, temporary_samples = train_test_split(
        videos,
        test_size=(1.0 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    temporary_labels = [
        sample["label"]
        for sample in temporary_samples
    ]
    # Segunda división:
    # validation = 15 %
    # test = 15 %
    relative_test_ratio = TEST_RATIO / (
        VAL_RATIO + TEST_RATIO
    )
    val_samples, test_samples = train_test_split(
        temporary_samples,
        test_size=relative_test_ratio,
        random_state=RANDOM_SEED,
        stratify=temporary_labels,
    )
    print_dataset_summary(
        train_samples,
        "Entrenamiento",
    )
    print_dataset_summary(
        val_samples,
        "Validación",
    )
    print_dataset_summary(
        test_samples,
        "Prueba",
    )
    train_dataset = VideoDataset(
        train_samples,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
    )
    val_dataset = VideoDataset(
        val_samples,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
    )
    test_dataset = VideoDataset(
        test_samples,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    print("\nConstruyendo Video Swin 3D preentrenado...")
    model = VideoSwinClassifier(
        num_classes=len(CLASS_NAMES),
        pretrained=True,
        freeze_backbone=FREEZE_BACKBONE,
    )
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE_HEAD,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1": [],
        "val_f1": [],
    }
    best_val_f1 = -np.inf
    best_epoch = -1
    best_model_path = (
        OUTPUT_DIR / "best_video_swin3d.pt"
    )
    for epoch in range(NUM_EPOCHS):
        print("\n" + "=" * 70)
        print(f"ÉPOCA {epoch + 1}/{NUM_EPOCHS}")
        print("=" * 70)
        # Descongelamos el backbone después de algunas épocas.
        if (
            FREEZE_BACKBONE
            and epoch == FREEZE_EPOCHS
        ):
            print("Descongelando backbone Video Swin 3D...")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": model.backbone.features.parameters(),
                        "lr": LEARNING_RATE_BACKBONE,
                    },
                    {
                        "params": model.backbone.head.parameters(),
                        "lr": LEARNING_RATE_HEAD,
                    },
                ],
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
            )
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )
        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            description="Validación",
        )
        scheduler.step(
            val_metrics["f1_macro"]
        )
        history["train_loss"].append(
            train_metrics["loss"]
        )
        history["val_loss"].append(
            val_metrics["loss"]
        )
        history["train_accuracy"].append(
            train_metrics["accuracy"]
        )
        history["val_accuracy"].append(
            val_metrics["accuracy"]
        )
        history["train_f1"].append(
            train_metrics["f1_macro"]
        )
        history["val_f1"].append(
            val_metrics["f1_macro"]
        )
        print("\nResultados de la época ",epoch + 1,":")
        print(
            f"Train loss: {train_metrics['loss']:.4f}"
        )
        print(
            f"Train accuracy: {train_metrics['accuracy']:.4f}"
        )
        print(
            f"Train F1: {train_metrics['f1_macro']:.4f}"
        )
        print(
            f"Val loss: {val_metrics['loss']:.4f}"
        )
        print(
            f"Val accuracy: {val_metrics['accuracy']:.4f}"
        )
        print(
            f"Val F1: {val_metrics['f1_macro']:.4f}"
        )
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "num_frames": NUM_FRAMES,
                    "image_size": IMAGE_SIZE,
                    "best_val_f1": best_val_f1,
                },
                best_model_path,
            )
            print(
                f"\nMejor modelo guardado en: {best_model_path}"
            )
    save_training_curves(history)
    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=4,
        )
    print("\nCargando el mejor modelo...")
    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    test_metrics, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
        description="Prueba",
    )
    print("\n" + "=" * 70)
    print("RESULTADOS FINALES EN TEST")
    print("=" * 70)
    print(
        f"Accuracy: {test_metrics['accuracy']:.4f}"
    )
    print(
        f"Precision macro: {test_metrics['precision_macro']:.4f}"
    )
    print(
        f"Recall macro: {test_metrics['recall_macro']:.4f}"
    )
    print(
        f"F1 macro: {test_metrics['f1_macro']:.4f}"
    )
    print("\nReporte por clase:")
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    print(report)
    save_confusion_matrix(
        y_true,
        y_pred,
    )
    final_results = {
        "best_epoch": best_epoch,
        "best_validation_f1": best_val_f1,
        "test_metrics": test_metrics,
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ).tolist(),
    }
    with open(
        OUTPUT_DIR / "test_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_results,
            file,
            indent=4,
        )
    print("\nArchivos generados:")
    print(f"- {best_model_path}")
    print(f"- {OUTPUT_DIR / 'training_history.json'}")
    print(f"- {OUTPUT_DIR / 'test_results.json'}")
    print(f"- {OUTPUT_DIR / 'loss_curve.png'}")
    print(f"- {OUTPUT_DIR / 'accuracy_curve.png'}")
    print(f"- {OUTPUT_DIR / 'f1_curve.png'}")
    print(f"- {OUTPUT_DIR / 'confusion_matrix_test.png'}")
    
if __name__ == "__main__":
    main()