# ============================================================
# AVT - Anticipación de desenfundado
#
# Adaptación del Anticipative Video Transformer (AVT)
# para un problema ONE-CLASS:
#
#     VIDEO PREVIO -> PREDECIR REPRESENTACIÓN DEL DESENFUNDE
#
# Los videos utilizados pertenecen únicamente a:
#
#     dataset/Des/
#
# El desenfundado ocurre SIEMPRE en el último frame.
#
# NO existe la clase Nodes en este experimento.
#
# ============================================================
import os
import cv2
import json
import math
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
DATASET_DIR = "dataset/Des"
VIDEO_EXTENSIONS = [
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
]
# ------------------------------------------------------------
# Video
# ------------------------------------------------------------
IMAGE_SIZE = 224
# Número de frames OBSERVADOS
NUM_OBSERVED_FRAMES = 16
# Número de frames utilizados para representar
# el futuro/desenfundado.
#
# Como el desenfundado ocurre en el último frame,
# utilizamos los últimos frames del video como
# representación del futuro.
NUM_FUTURE_FRAMES = 4
FPS_EXPECTED = 30.0
# ------------------------------------------------------------
# Anticipación
# ------------------------------------------------------------
# El modelo se entrenará con diferentes horizontes.
#
# Por ejemplo:
#
#   0.5 s
#   1.0 s
#   1.5 s
#   2.0 s
#
ANTICIPATION_TIMES = [
    0.5,
    1.0,
    1.5,
    2.0
]
# ------------------------------------------------------------
# ViT
# ------------------------------------------------------------
PATCH_SIZE = 16
EMBED_DIM = 128
NUM_HEADS = 8
NUM_SPATIAL_LAYERS = 4
SPATIAL_MLP_DIM = 512
DROPOUT = 0.4
# ------------------------------------------------------------
# Transformer temporal
# ------------------------------------------------------------
NUM_TEMPORAL_LAYERS = 4
TEMPORAL_MLP_DIM = 512
# ------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------
BATCH_SIZE = 8
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
# Peso de la pérdida de características
LAMBDA_FEATURE = 1.0
# Peso de cosine loss
LAMBDA_COSINE = 0.5
# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------
fecha = datetime.now()
dir_salida = str(fecha.year) + str(fecha.month) +str(fecha.day) + str("-") + str(fecha.hour) + str(fecha.minute) + str(fecha.second) + str("_AVT")
OUTPUT_DIR = Path(dir_salida)
BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "best_avt_anticipation.pth"
)
HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "training_history.json"
)
CURVE_PATH = os.path.join(
    OUTPUT_DIR,
    "training_curves.png"
)
RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "test_results.json"
)
CONFUSION_MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)
# ============================================================
# 2. DEVICE
# ============================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("=" * 70)
print("AVT - ANTICIPACIÓN DE DESENFUNDADO")
print("=" * 70)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
    torch.backends.cudnn.benchmark = True
# ============================================================
# 3. SEED
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(RANDOM_SEED)
# ============================================================
# 4. CREAR DIRECTORIO
# ============================================================
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)
# ============================================================
# 5. OBTENER VIDEOS
# ============================================================
def get_video_files(directory):
    directory = Path(directory)
    videos = []
    for extension in VIDEO_EXTENSIONS:
        videos.extend(
            directory.glob(f"*{extension}")
        )
    return sorted(videos)
video_files = get_video_files(
    DATASET_DIR
)
print()
print(
    f"Videos encontrados: {len(video_files)}"
)
if len(video_files) == 0:
    raise RuntimeError(
        f"No se encontraron videos en: {DATASET_DIR}"
    )
# ============================================================
# 6. SPLIT POR VIDEO
#
# MUY IMPORTANTE:
#
# No hacemos split de ventanas.
#
# Primero dividimos los videos.
#
# Esto evita que diferentes ventanas del mismo video
# aparezcan simultáneamente en train y test.
# ============================================================
train_videos, temp_videos = train_test_split(
    video_files,
    test_size=(1.0 - TRAIN_RATIO),
    random_state=RANDOM_SEED
)
relative_test = TEST_RATIO / (
    VAL_RATIO + TEST_RATIO
)
val_videos, test_videos = train_test_split(
    temp_videos,
    test_size=relative_test,
    random_state=RANDOM_SEED
)
print()
print("Split:")
print(
    f"Train: {len(train_videos)} videos"
)
print(
    f"Validation: {len(val_videos)} videos"
)
print(
    f"Test: {len(test_videos)} videos"
)
# ============================================================
# 7. FUNCIONES DE VIDEO
# ============================================================
def get_video_info(video_path):
    cap = cv2.VideoCapture(
        str(video_path)
    )
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir: {video_path}"
        )
    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )
    fps = cap.get(
        cv2.CAP_PROP_FPS
    )
    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )
    cap.release()
    return total_frames, fps, duration
# ============================================================
# 8. LEER FRAMES ESPECÍFICOS
# ============================================================
def read_frames(
    video_path,
    frame_indices
):
    cap = cv2.VideoCapture(
        str(video_path)
    )
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir {video_path}"
        )
    frames = []
    for idx in frame_indices:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(idx)
        )
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(
                f"No se pudo leer frame "
                f"{idx} de {video_path}"
            )
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
        frame = cv2.resize(
            frame,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )
        frame = (
            frame.astype(np.float32)
            / 255.0
        )
        frames.append(frame)
    cap.release()
    frames = np.stack(
        frames,
        axis=0
    )
    return frames
# ============================================================
# 9. NORMALIZACIÓN
# ============================================================
def normalize_frames(frames):
    mean = np.array(
        [
            0.485,
            0.456,
            0.406
        ],
        dtype=np.float32
    )
    std = np.array(
        [
            0.229,
            0.224,
            0.225
        ],
        dtype=np.float32
    )
    frames = (
        frames - mean
    ) / std
    return frames
# ============================================================
# 10. DATASET
# ============================================================
def apply_video_augmentation(observed, future):
    """
    observed: Tensor [T_obs, C, H, W]
    future: Tensor [T_fut, C, H, W]
    """
    # 1. Flip Horizontal aleatorio
    if random.random() > 0.5:
        observed = TF.hflip(observed)
        future = TF.hflip(future)
        
    # 2. Crop Aleatorio Coherente
    i, j, h, w = transforms.RandomCrop.get_params(
        observed, output_size=(192, 192)
    )
    observed = TF.crop(observed, i, j, h, w)
    future = TF.crop(future, i, j, h, w)
    
    observed = TF.resize(observed, [IMAGE_SIZE, IMAGE_SIZE])
    future = TF.resize(future, [IMAGE_SIZE, IMAGE_SIZE])
    
    # 3. Color Jitter (aplicado por igual)
    if random.random() > 0.5:
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        
        observed = TF.adjust_brightness(observed, brightness_factor)
        future = TF.adjust_brightness(future, brightness_factor)
        
        observed = TF.adjust_contrast(observed, contrast_factor)
        future = TF.adjust_contrast(future, contrast_factor)
        
    return observed, future

class DesAnticipationDataset(
    Dataset
):
    def __init__(
        self,
        video_files,
        anticipation_times,
        training=False
    ):
        self.video_files = video_files
        self.anticipation_times = (
            anticipation_times
        )
        self.training = training
        self.samples = []
        self._build_samples()
    # --------------------------------------------------------
    # Construcción de muestras
    # --------------------------------------------------------
    def _build_samples(self):
        for video_path in self.video_files:
            try:
                total_frames, fps, duration = (
                    get_video_info(video_path)
                )
                if fps <= 0:
                    fps = FPS_EXPECTED
                # Último frame
                draw_frame = (
                    total_frames - 1
                )
                for anticipation in (
                    self.anticipation_times
                ):
                    anticipation_frames = int(
                        anticipation * fps
                    )
                    observed_end = (
                        draw_frame
                        - anticipation_frames
                    )
                    # Necesitamos suficientes frames
                    if observed_end < (
                        NUM_OBSERVED_FRAMES
                    ):
                        continue
                    # El futuro debe contener
                    # suficientes frames.
                    future_start = max(
                        0,
                        draw_frame
                        - NUM_FUTURE_FRAMES
                        + 1
                    )
                    self.samples.append(
                        {
                            "video": video_path,
                            "fps": fps,
                            "total_frames": total_frames,
                            "draw_frame": draw_frame,
                            "observed_end": observed_end,
                            "future_start": future_start,
                            "anticipation": anticipation
                        }
                    )
            except Exception as e:
                print(
                    f"[WARNING] "
                    f"{video_path}: {e}"
                )
    def __len__(self):
        return len(self.samples)
    # --------------------------------------------------------
    # Muestreo uniforme de frames
    # --------------------------------------------------------
    def _uniform_indices(
        self,
        start,
        end,
        num_frames
    ):
        if end <= start:
            indices = [
                start
            ] * num_frames
        else:
            indices = np.linspace(
                start,
                end,
                num_frames
            )
        return indices.astype(int)
    # --------------------------------------------------------
    # Obtener muestra
    # --------------------------------------------------------
    def __getitem__(
        self,
        index
    ):
        sample = self.samples[index]
        video_path = sample["video"]
        draw_frame = sample["draw_frame"]
        observed_end = sample["observed_end"]
        future_start = sample["future_start"]
        # ----------------------------------------------------
        # Frames observados
        #
        # SOLO hasta anticipation_time antes
        # del desenfundado.
        # ----------------------------------------------------
        observed_indices = (
            self._uniform_indices(
                0,
                observed_end,
                NUM_OBSERVED_FRAMES
            )
        )
        # ----------------------------------------------------
        # Frames futuros
        #
        # Los últimos frames representan el desenfundado.
        # ----------------------------------------------------
        future_indices = (
            self._uniform_indices(
                future_start,
                draw_frame,
                NUM_FUTURE_FRAMES
            )
        )
        # ----------------------------------------------------
        # Leer frames
        # ----------------------------------------------------
        observed = read_frames(
            video_path,
            observed_indices
        )
        future = read_frames(
            video_path,
            future_indices
        )
        # ----------------------------------------------------
        # Normalización
        # ----------------------------------------------------
        observed = normalize_frames(
            observed
        )
        future = normalize_frames(
            future
        )
        # ----------------------------------------------------
        # Convertir: T,H,W,C -> T,C,H,W
        # ----------------------------------------------------
        observed = torch.from_numpy(observed).permute(0, 3, 1, 2).float()
        future = torch.from_numpy(future).permute(0, 3, 1, 2).float()

        # Aplicar aumento en entrenamiento
        if self.training:
            observed, future = apply_video_augmentation(observed, future)

        return {
            "observed": observed,
            "future": future,
            "anticipation": torch.tensor(sample["anticipation"], dtype=torch.float32),
            "video": str(video_path)
        }
# ============================================================
# 11. DATASETS
# ============================================================
train_dataset = (
    DesAnticipationDataset(
        train_videos,
        ANTCIPATION_TIMES
        if False else ANTICIPATION_TIMES,
        training=True
    )
)
val_dataset = (
    DesAnticipationDataset(
        val_videos,
        ANTICIPATION_TIMES,
        training=False
    )
)
test_dataset = (
    DesAnticipationDataset(
        test_videos,
        ANTICIPATION_TIMES,
        training=False
    )
)
print()
print(
    f"Muestras train: {len(train_dataset)}"
)
print(
    f"Muestras val:   {len(val_dataset)}"
)
print(
    f"Muestras test:  {len(test_dataset)}"
)
# ============================================================
# 12. DATALOADERS
# ============================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)
# ============================================================
# 13. PATCH EMBEDDING
# ============================================================
class PatchEmbedding(
    nn.Module
):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=256
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        )
        self.num_patches = (
            image_size // patch_size
        ) ** 2
        # Cada patch es proyectado mediante
        # una convolución.
        #
        # kernel = patch_size
        # stride = patch_size
        #
        # Esto equivale a dividir la imagen
        # en patches no solapados.
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    def forward(self, x):
        # x:
        #
        # B,C,H,W
        #
        x = self.projection(x)
        # B,D,H/P,W/P
        x = x.flatten(
            2
        )
        # B,D,N
        x = x.transpose(
            1,
            2
        )
        # B,N,D
        return x
# ============================================================
# 14. SPATIAL VISION TRANSFORMER
# ============================================================
class SpatialViT(
    nn.Module
):
    def __init__(self):
        super().__init__()
        self.patch_embedding = (
            PatchEmbedding(
                image_size=IMAGE_SIZE,
                patch_size=PATCH_SIZE,
                embed_dim=EMBED_DIM
            )
        )
        num_patches = (
            self.patch_embedding.num_patches
        )
        # CLS token
        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                EMBED_DIM
            )
        )
        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                EMBED_DIM
            )
        )
        # Transformer Encoder
        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=EMBED_DIM,
                nhead=NUM_HEADS,
                dim_feedforward=SPATIAL_MLP_DIM,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )
        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=NUM_SPATIAL_LAYERS
            )
        )
        self.norm = nn.LayerNorm(
            EMBED_DIM
        )
        self._initialize()
    def _initialize(self):
        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02
        )
        nn.init.trunc_normal_(
            self.pos_embedding,
            std=0.02
        )
    def forward(self, x):
        # x:
        #
        # B,C,H,W
        #
        B = x.size(0)
        x = self.patch_embedding(
            x
        )
        # B,N,D
        cls = self.cls_token.expand(
            B,
            -1,
            -1
        )
        x = torch.cat(
            [
                cls,
                x
            ],
            dim=1
        )
        x = (
            x
            + self.pos_embedding
        )
        x = self.encoder(
            x
        )
        x = self.norm(
            x
        )
        # Extraemos CLS
        cls_feature = x[:, 0]
        return cls_feature
# ============================================================
# 15. BACKBONE AVT
# ============================================================
class AVTBackbone(
    nn.Module
):
    def __init__(self):
        super().__init__()
        self.spatial_encoder = (
            SpatialViT()
        )
    def forward(self, video):
        # video:
        #
        # B,T,C,H,W
        #
        B, T, C, H, W = (
            video.shape
        )
        # Convertimos:
        #
        # B,T,C,H,W
        #
        # a
        #
        # B*T,C,H,W
        frames = video.reshape(
            B * T,
            C,
            H,
            W
        )
        # Cada frame pasa por el mismo
        # encoder espacial.
        features = (
            self.spatial_encoder(
                frames
            )
        )
        # B*T,D
        #
        # -> B,T,D
        features = features.reshape(
            B,
            T,
            EMBED_DIM
        )
        return features
# ============================================================
# 16. TEMPORAL CAUSAL TRANSFORMER
# ============================================================
class CausalTemporalTransformer(
    nn.Module
):
    def __init__(self):
        super().__init__()
        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=EMBED_DIM,
                nhead=NUM_HEADS,
                dim_feedforward=TEMPORAL_MLP_DIM,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )
        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=NUM_TEMPORAL_LAYERS
            )
        )
        self.norm = nn.LayerNorm(
            EMBED_DIM
        )
    def _causal_mask(
        self,
        T,
        device
    ):
        # Máscara triangular superior.
        #
        # Los valores -inf impiden que un frame
        # atienda a frames futuros.
        mask = torch.triu(
            torch.ones(
                T,
                T,
                device=device
            ),
            diagonal=1
        )
        mask = mask.masked_fill(
            mask == 1,
            float("-inf")
        )
        return mask
    def forward(self, x):
        # x:
        #
        # B,T,D
        #
        T = x.size(1)
        mask = self._causal_mask(
            T,
            x.device
        )
        x = self.transformer(
            x,
            mask=mask
        )
        x = self.norm(
            x
        )
        return x
# ============================================================
# 17. FUTURE FEATURE PREDICTOR
# ============================================================
class FutureFeaturePredictor(
    nn.Module
):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(
                EMBED_DIM,
                EMBED_DIM
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                EMBED_DIM,
                EMBED_DIM
            )
        )
    def forward(self, x):
        return self.network(x)
# ============================================================
# 18. AVT COMPLETO
# ============================================================
class AnticipativeVideoTransformer(
    nn.Module
):
    def __init__(self):
        super().__init__()
        # ----------------------------------------------------
        # Backbone espacial
        # ----------------------------------------------------
        self.backbone = (
            AVTBackbone()
        )
        # ----------------------------------------------------
        # Transformer temporal causal
        # ----------------------------------------------------
        self.temporal = (
            CausalTemporalTransformer()
        )
        # ----------------------------------------------------
        # Predictor futuro
        # ----------------------------------------------------
        self.future_predictor = (
            FutureFeaturePredictor()
        )
    def encode_video(
        self,
        video
    ):
        # Frame features
        frame_features = (
            self.backbone(
                video
            )
        )
        # Temporal causal features
        temporal_features = (
            self.temporal(
                frame_features
            )
        )
        return (
            frame_features,
            temporal_features
        )
    def predict_future(
        self,
        video
    ):
        (
            frame_features,
            temporal_features
        ) = self.encode_video(
            video
        )
        # Utilizamos solamente la representación
        # correspondiente al último frame OBSERVADO.
        current_feature = (
            temporal_features[:, -1]
        )
        # Predicción del futuro
        predicted_future = (
            self.future_predictor(
                current_feature
            )
        )
        return (
            predicted_future,
            frame_features,
            temporal_features
        )
    def forward(
        self,
        video
    ):
        return self.predict_future(
            video
        )
# ============================================================
# 19. MODELO
# ============================================================
model = (
    AnticipativeVideoTransformer()
    .to(DEVICE)
)
# ============================================================
# 20. INFORMACIÓN DEL MODELO
# ============================================================
num_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
print()
print(
    f"Parámetros entrenables: "
    f"{num_parameters:,}"
)
# ============================================================
# 21. OPTIMIZADOR
# ============================================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )
)
# ============================================================
# 22. FUNCIÓN DE PÉRDIDA
# ============================================================
def future_loss(
    predicted,
    target
):
    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------
    mse = F.mse_loss(
        predicted,
        target
    )
    # --------------------------------------------------------
    # Cosine distance
    # --------------------------------------------------------
    predicted_norm = F.normalize(
        predicted,
        dim=-1
    )
    target_norm = F.normalize(
        target,
        dim=-1
    )
    cosine_similarity = (
        predicted_norm
        * target_norm
    ).sum(
        dim=-1
    )
    cosine_loss = (
        1.0
        - cosine_similarity.mean()
    )
    loss = (
        LAMBDA_FEATURE * mse
        +
        LAMBDA_COSINE * cosine_loss
    )
    return (
        loss,
        mse,
        cosine_loss
    )
# ============================================================
# 23. OBTENER REPRESENTACIÓN FUTURA
# ============================================================
@torch.no_grad()
def compute_future_target(
    future_video
):
    B, T, C, H, W = (
        future_video.shape
    )
    # Pasamos los frames futuros por el
    # mismo encoder espacial.
    frames = future_video.reshape(
        B * T,
        C,
        H,
        W
    )
    features = (
        model.backbone.spatial_encoder(
            frames
        )
    )
    features = features.reshape(
        B,
        T,
        EMBED_DIM
    )
    # Promedio de las representaciones
    # de los últimos frames.
    target = features.mean(
        dim=1
    )
    return target
# ============================================================
# 24. TRAIN EPOCH
# ============================================================
def train_one_epoch():
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    num_batches = 0
    for batch in train_loader:
        observed = (
            batch["observed"]
            .to(
                DEVICE,
                non_blocking=True
            )
        )
        future = (
            batch["future"]
            .to(
                DEVICE,
                non_blocking=True
            )
        )
        optimizer.zero_grad(
            set_to_none=True
        )
        # ----------------------------------------------------
        # Predicción
        # ----------------------------------------------------
        predicted_future, _, _ = (
            model(
                observed
            )
        )
        # ----------------------------------------------------
        # Target
        #
        # IMPORTANTE:
        #
        # El target futuro se calcula sin gradiente.
        # El modelo NO puede utilizarlo como entrada.
        # ----------------------------------------------------
        with torch.no_grad():
            target_future = (
                compute_future_target(
                    future
                )
            )
        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------
        loss, mse, cosine = (
            future_loss(
                predicted_future,
                target_future
            )
        )
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        optimizer.step()
        total_loss += (
            loss.item()
        )
        total_mse += (
            mse.item()
        )
        total_cosine += (
            cosine.item()
        )
        num_batches += 1
    return (
        total_loss / num_batches,
        total_mse / num_batches,
        total_cosine / num_batches
    )
# ============================================================
# 25. VALIDACIÓN
# ============================================================
@torch.no_grad()
def validate():
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    num_batches = 0
    for batch in val_loader:
        observed = (
            batch["observed"]
            .to(DEVICE)
        )
        future = (
            batch["future"]
            .to(DEVICE)
        )
        predicted_future, _, _ = (
            model(
                observed
            )
        )
        target_future = (
            compute_future_target(
                future
            )
        )
        loss, mse, cosine = (
            future_loss(
                predicted_future,
                target_future
            )
        )
        total_loss += (
            loss.item()
        )
        total_mse += (
            mse.item()
        )
        total_cosine += (
            cosine.item()
        )
        num_batches += 1
    return (
        total_loss / num_batches,
        total_mse / num_batches,
        total_cosine / num_batches
    )
# ============================================================
# 26. ENTRENAMIENTO
# ============================================================
history = {
    "train_loss": [],
    "train_mse": [],
    "train_cosine": [],
    "val_loss": [],
    "val_mse": [],
    "val_cosine": []
}
best_val_loss = float(
    "inf"
)
print()
print("=" * 70)
print("INICIANDO ENTRENAMIENTO")
print("=" * 70)
for epoch in range(
    NUM_EPOCHS
):
    train_loss, train_mse, train_cosine = (
        train_one_epoch()
    )
    val_loss, val_mse, val_cosine = (
        validate()
    )
    scheduler.step()
    # Guardar historia
    history[
        "train_loss"
    ].append(
        train_loss
    )
    history[
        "train_mse"
    ].append(
        train_mse
    )
    history[
        "train_cosine"
    ].append(
        train_cosine
    )
    history[
        "val_loss"
    ].append(
        val_loss
    )
    history[
        "val_mse"
    ].append(
        val_mse
    )
    history[
        "val_cosine"
    ].append(
        val_cosine
    )
    # --------------------------------------------------------
    # Guardar mejor modelo
    # --------------------------------------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "epoch":
                    epoch + 1,
                "val_loss":
                    val_loss,
                "config":
                    {
                        "num_observed_frames":
                            NUM_OBSERVED_FRAMES,
                        "num_future_frames":
                            NUM_FUTURE_FRAMES,
                        "anticipation_times":
                            ANTICIPATION_TIMES,
                        "image_size":
                            IMAGE_SIZE,
                        "embed_dim":
                            EMBED_DIM
                    }
            },
            BEST_MODEL_PATH
        )
    current_lr = (
        optimizer.param_groups[0]["lr"]
    )
    print(
        f"Epoch "
        f"{epoch + 1:03d}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Val Loss: {val_loss:.5f} | "
        f"MSE: {val_mse:.5f} | "
        f"Cos: {val_cosine:.5f} | "
        f"LR: {current_lr:.7f}"
    )
# ============================================================
# 27. GUARDAR HISTORIA
# ============================================================
with open(
    HISTORY_PATH,
    "w"
) as f:
    json.dump(
        history,
        f,
        indent=4
    )
# ============================================================
# 28. CURVAS
# ============================================================
plt.figure(
    figsize=(10, 6)
)
plt.plot(
    history["train_loss"],
    label="Train Loss"
)
plt.plot(
    history["val_loss"],
    label="Validation Loss"
)
plt.xlabel(
    "Epoch"
)
plt.ylabel(
    "Loss"
)
plt.title(
    "AVT - Future Prediction Loss"
)
plt.legend()
plt.grid(
    True
)
plt.tight_layout()
plt.savefig(
    CURVE_PATH,
    dpi=200
)
plt.close()
# ============================================================
# 28. CURVAS DE PÉRDIDA
# ============================================================
print("Generando curvas de pérdida...")
plt.figure(figsize=(10, 6))
plt.plot(history["train_loss"], label="Train Loss", color='blue', linewidth=2)
plt.plot(history["val_loss"], label="Validation Loss", color='orange', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("AVT - Curva de Pérdida en Entrenamiento y Validación")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(CURVE_PATH, dpi=200)
plt.close()
# ============================================================
# 29. CARGAR MEJOR MODELO
# ============================================================
checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE
)
model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)
model.eval()
print()
print("=" * 70)
print("MEJOR MODELO CARGADO")
print("=" * 70)
print(
    f"Epoch: {checkpoint['epoch']}"
)
print(
    f"Validation Loss: "
    f"{checkpoint['val_loss']:.6f}"
)
# ============================================================
# 30. EVALUACIÓN DEL TEST
# ============================================================
def evaluate_test():
    model.eval()
    results = {}
    for anticipation in (
        ANTICIPATION_TIMES
    ):
        similarities = []
        mse_values = []
        for sample in test_dataset.samples:
            if abs(
                sample["anticipation"]
                - anticipation
            ) > 1e-6:
                continue
            # ----------------------------------------------
            # Índices observados
            # ----------------------------------------------
            draw_frame = (
                sample["draw_frame"]
            )
            observed_end = (
                sample["observed_end"]
            )
            future_start = (
                sample["future_start"]
            )
            observed_indices = (
                np.linspace(
                    0,
                    observed_end,
                    NUM_OBSERVED_FRAMES
                ).astype(int)
            )
            future_indices = (
                np.linspace(
                    future_start,
                    draw_frame,
                    NUM_FUTURE_FRAMES
                ).astype(int)
            )
            # ----------------------------------------------
            # Leer
            # ----------------------------------------------
            observed = read_frames(
                sample["video"],
                observed_indices
            )
            future = read_frames(
                sample["video"],
                future_indices
            )
            observed = normalize_frames(
                observed
            )
            future = normalize_frames(
                future
            )
            observed = torch.from_numpy(
                observed
            ).permute(
                0,
                3,
                1,
                2
            ).float().unsqueeze(0).to(
                DEVICE
            )
            future = torch.from_numpy(
                future
            ).permute(
                0,
                3,
                1,
                2
            ).float().unsqueeze(0).to(
                DEVICE
            )
            # ----------------------------------------------
            # Predicción
            # ----------------------------------------------
            predicted, _, _ = (
                model(
                    observed
                )
            )
            # ----------------------------------------------
            # Target
            # ----------------------------------------------
            target = (
                compute_future_target(
                    future
                )
            )
            # ----------------------------------------------
            # Cosine similarity
            # ----------------------------------------------
            similarity = F.cosine_similarity(
                predicted,
                target,
                dim=-1
            ).item()
            mse = F.mse_loss(
                predicted,
                target
            ).item()
            similarities.append(
                similarity
            )
            mse_values.append(
                mse
            )
        if len(similarities) > 0:
            results[
                str(anticipation)
            ] = {
                "mean_cosine_similarity":
                    float(
                        np.mean(
                            similarities
                        )
                    ),
                "std_cosine_similarity":
                    float(
                        np.std(
                            similarities
                        )
                    ),
                "mean_mse":
                    float(
                        np.mean(
                            mse_values
                        )
                    ),
                "num_samples":
                    len(
                        similarities
                    )
            }
    return results
test_results = evaluate_test()
# ============================================================
# 31. MOSTRAR RESULTADOS Y MATRIZ DE CONFUSIÓN
# ============================================================
print()
print("=" * 70)
print("RESULTADOS DE TEST Y MATRIZ DE CONFUSIÓN")
print("=" * 70)
# Umbral para decidir si el modelo anticipó correctamente la acción (One-class)
THRESHOLD = 0.80
y_true_all = []
y_pred_all = []
for anticipation, result in test_results.items():
    print(f"\nAnticipación: {anticipation} segundos")
    print(f"Cosine similarity: {result['mean_cosine_similarity']:.4f}")
    print(f"Std cosine: {result['std_cosine_similarity']:.4f}")
    print(f"MSE: {result['mean_mse']:.6f}")
    print(f"Muestras: {result['num_samples']}")
    # Extraemos todas las similitudes de esta anticipación
    # (Para esto, necesitas que evaluate_test devuelva la lista cruda de 'similarities'.
    # Si modificas evaluate_test para guardar los crudos en test_results, úsalos aquí.
    # Como atajo usando la media, simularemos el comportamiento global para la matriz)
# Nota: Para generar una matriz exacta, reconstruimos la evaluación global
y_true = []
y_pred = []
model.eval()
with torch.no_grad():
    for sample in test_dataset.samples:
        observed_indices = np.linspace(0, sample["observed_end"], NUM_OBSERVED_FRAMES).astype(int)
        future_indices = np.linspace(sample["future_start"], sample["draw_frame"], NUM_FUTURE_FRAMES).astype(int)
        observed = normalize_frames(read_frames(sample["video"], observed_indices))
        future = normalize_frames(read_frames(sample["video"], future_indices))
        observed = torch.from_numpy(observed).permute(0, 3, 1, 2).float().unsqueeze(0).to(DEVICE)
        future = torch.from_numpy(future).permute(0, 3, 1, 2).float().unsqueeze(0).to(DEVICE)
        predicted, _, _ = model(observed)
        target = compute_future_target(future)
        similarity = F.cosine_similarity(predicted, target, dim=-1).item()
        y_true.append(1) # Siempre es 1 porque es One-Class (siempre hay desenfunde)
        y_pred.append(1 if similarity >= THRESHOLD else 0)
# Generar y guardar la matriz
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Anticipado (0)', 'Anticipado (1)'],
            yticklabels=['No Anticipado (0)', 'Anticipado (1)'])
plt.title(f"Matriz de Confusión (Umbral Similitud >= {THRESHOLD})")
plt.xlabel("Predicción del Modelo")
plt.ylabel("Etiqueta Real")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
plt.close()
# ============================================================
# 32. GUARDAR RESULTADOS
# ============================================================
with open(RESULTS_PATH, "w") as f:
    json.dump(test_results, f, indent=4)
# ============================================================
# 33. FINAL
# ============================================================
print()
print("=" * 70)
print("ENTRENAMIENTO FINALIZADO")
print("=" * 70)
print(f"Modelo: {BEST_MODEL_PATH}")
print(f"Historia: {HISTORY_PATH}")
print(f"Curvas (Pérdida): {CURVE_PATH}")
print(f"Matriz de Confusión: {CONFUSION_MATRIX_PATH}")
print(f"Resultados: {RESULTS_PATH}")