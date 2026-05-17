"""
Bird-MAE frozen encoder with prototypical probing (Bird-MAE paper).

This module implements prototypical probing as described in the Bird-MAE paper:
- Preserves spatial structure (no global pooling)
- Learns class prototypes that match local spectrogram patches
- Uses max-pooled cosine similarity for classification
- Parameter-efficient and interpretable
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import librosa
from typing import Dict, List, Tuple, Optional, Union
import warnings
import os
import tempfile
import soundfile as sf
from sklearn.utils import shuffle as shuffle_func
import math

# DirectML/CUDA/CPU-aware helpers
from model_pipelines.utils.device_utils import get_device, should_pin_memory

# Suppress the symlink warning on Windows
warnings.filterwarnings("ignore", message=".*symlinks.*")

try:
    from transformers import AutoFeatureExtractor, AutoModel
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


class BirdMAEExtractor:
    """
    Frozen Bird-MAE feature extractor returning spatial feature maps.
    
    Extracts spatial feature map [H, W, D] that preserves the structure
    of the spectrogram, critical for prototypical probing.
    """
    
    def __init__(self, cfg):
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        self.cfg = cfg
        self.device = cfg.device()
        
        print(f"\n  Loading Bird-MAE on {self.device}...")
        
        # Load with specific configuration to avoid compatibility issues
        try:
            # First try normal loading
            self.model = AutoModel.from_pretrained(
                "DBD-research-group/Bird-MAE-Base",
                trust_remote_code=True
            ).to(self.device)
        except AttributeError as e:
            if "all_tied_weights_keys" in str(e):
                # Compatibility fix for older transformers version
                print("  Applying compatibility fix for transformers version...")
                self.model = AutoModel.from_pretrained(
                    "DBD-research-group/Bird-MAE-Base",
                    trust_remote_code=True,
                    ignore_mismatched_sizes=True
                ).to(self.device)
            else:
                raise
        
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "DBD-research-group/Bird-MAE-Base",
            trust_remote_code=True
        )
        
        # Configure model to return full sequence, not pooled
        # Set output_hidden_states to True to get all patch embeddings
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = False
        
        if hasattr(self.model.config, 'output_patches'):
            self.model.config.output_patches = True
        
        # Ensure we're not using global pooling
        if hasattr(self.model.config, 'pooling_mode'):
            self.model.config.pooling_mode = 'none'
        
        self.model.eval()
        
        # Cache for spatial features
        self.spatial_cache = {}
        
        # Spatial dimensions (for 128x512 spectrogram with 16x16 patches)
        self.embed_dim = 768
        self.num_patches_h = 8   # 128 / 16
        self.num_patches_w = 32  # 512 / 16
        
        print(f"  Bird-MAE loaded. Embedding dimension: {self.embed_dim}")
        print(f"  Spatial feature map shape: {self.num_patches_h}×{self.num_patches_w}×{self.embed_dim}")
        
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio."""
        try:
            audio, _ = librosa.load(audio_path, sr=self.cfg.sample_rate,
                                     duration=self.cfg.duration, mono=True)
        except Exception as e:
            print(f"    Warning: Could not load {audio_path}: {e}")
            audio = np.zeros(int(self.cfg.sample_rate * self.cfg.duration))
        
        target_len = int(self.cfg.sample_rate * self.cfg.duration)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        return audio[:target_len]

    @torch.no_grad()
    def _extract_spatial_from_audio(self, audio: np.ndarray, 
                                  target_duration: float = None) -> np.ndarray:
        """
        Core extraction logic: audio array -> spatial feature map [H, W, D].
        All public extraction methods delegate here.
        """
        # Ensure correct length
        target_len = int(self.cfg.sample_rate * (target_duration or self.cfg.duration))
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        # Feature extraction
        inputs = self.feature_extractor(audio, return_tensors="pt")
        if isinstance(inputs, torch.Tensor):
            inputs = {"input_values": inputs.to(self.device)}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        
        # Get per-patch embeddings from hidden_states if available,
        # falling back to last_hidden_state
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            patch_embeddings = outputs.hidden_states[-1]
        else:
            patch_embeddings = outputs.last_hidden_state

        expected_patches = self.num_patches_h * self.num_patches_w

        if patch_embeddings.shape[1] == expected_patches + 1:
            patch_embeddings = patch_embeddings[:, 1:, :]  # remove CLS token
        elif patch_embeddings.shape[1] != expected_patches:
            raise ValueError(
                f"Unexpected patch count: {patch_embeddings.shape[1]}. "
                f"Expected {expected_patches} or {expected_patches + 1}."
            )

        return patch_embeddings[0].reshape(
            self.num_patches_h, self.num_patches_w, self.embed_dim
        ).cpu().numpy()
    
    @torch.no_grad()
    def get_spatial_features(self, audio_path: str) -> np.ndarray:
        if audio_path in self.spatial_cache:
            return self.spatial_cache[audio_path]
        audio = self.load_audio(audio_path)
        features = self._extract_spatial_from_audio(audio)
        self.spatial_cache[audio_path] = features
        return features
    
    def extract_batch(self, audio_paths: List[str]) -> List[np.ndarray]:
        """
        Extract spatial features for multiple audio files.
        
        RETURNS:
            List of numpy arrays each of shape [H, W, D]
        """
        return [self.get_spatial_features(path) for path in audio_paths]
    
    def clear_cache(self):
        """Clear cache to free memory."""
        self.spatial_cache.clear()
    

class PrototypicalProbe(nn.Module):
    """
    Prototypical probing head as described in Bird-MAE paper (Section 3.3).
    
    This is a parameter-efficient probing method that:
    1. Preserves spatial structure (no global pooling)
    2. Learns class prototypes that match local spectrogram patches
    3. Uses max-pooled cosine similarity for classification
    4. Provides interpretability via prototype activation heatmaps
    
    ARCHITECTURE (from Bird-MAE paper):
        Input: Spatial feature map [B, D, H, W] from frozen encoder
        ↓
        For each class c, J prototypes p_{c,j} ∈ ℝ^D (learnable)
        ↓
        Cosine similarity between each prototype and all spatial patches
        ↓
        Max-pool across space: s_{c,j} = max_{h,w} similarity(p_{c,j}, h_{h,w})
        ↓
        Class logit = Σ_j (w_{c,j} * s_{c,j}) + b_c (with w_{c,j} ≥ 0)
        ↓
        Output: [B, C] logits for multi-label classification
    
    PARAMETER COUNT:
        For C classes, J prototypes, D dimension:
        - Prototypes: C × J × D
        - Final layer: C × J (weights) + C (biases)
        Total: C × J × (D + 1) + C ≈ C × J × D (dominant term)
    
    ARGUMENTS:
        num_classes: Number of bird species (C)
        num_prototypes: Prototypes per class (J), default 20 (from paper ablation)
        feature_dim: Bird-MAE embedding dimension (D), default 768
        use_orthogonality_loss: Encourage diverse prototypes within each class
    """
    
    def __init__(self, 
                 num_classes: int,
                 num_prototypes: int = 20,
                 feature_dim: int = 768,
                 use_orthogonality_loss: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.feature_dim = feature_dim
        self.use_orthogonality_loss = use_orthogonality_loss
        
        # Learnable prototypes: shape [C, J, D]
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, num_prototypes, feature_dim)
        )
        
        # Initialize prototypes on unit sphere
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)
        
        # Final layer: maps J prototype similarities to class logit
        self.final_weights = nn.Parameter(
            torch.ones(num_classes, num_prototypes)
        )
        self.final_biases = nn.Parameter(
            torch.full((num_classes,), -2.0)
        )
        
        print(f"  PrototypicalProbe initialized:")
        print(f"    - Classes: {num_classes}")
        print(f"    - Prototypes per class: {num_prototypes}")
        print(f"    - Total prototypes: {num_classes * num_prototypes}")
        print(f"    - Trainable params: {self._count_params():,}")
    
    def _count_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through prototypical probe.
        
        ARGUMENTS:
            x: Spatial feature map [B, D, H, W]
        
        RETURNS:
            logits: [B, C] logits for multi-label classification
        """
        B, D, H, W = x.shape
        
        # Reshape to [B, H*W, D] for efficient computation
        features = x.permute(0, 2, 3, 1).reshape(B, H * W, D)
        
        # Normalize to unit sphere (for cosine similarity)
        features_norm = F.normalize(features, dim=-1)
        
        # Normalize prototypes
        prototypes_norm = F.normalize(self.prototypes, dim=-1)  # [C, J, D]
        
        # Compute cosine similarities
        similarities = torch.einsum('bnd,cjd->bncj', features_norm, prototypes_norm)
        
        # Max-pool across spatial dimensions
        max_similarities, _ = torch.max(similarities, dim=1)  # [B, C, J]
        
        # Apply non-negative weights
        positive_weights = F.relu(self.final_weights)  # [C, J]
        
        # Compute class logits
        logits = torch.einsum('bcj,cj->bc', max_similarities, positive_weights) + self.final_biases
        
        return logits
    
    def get_prototype_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get spatial activation maps for interpretability.
        
        ARGUMENTS:
            x: Spatial feature map [B, D, H, W]
        
        RETURNS:
            activations: [B, C, J, H, W] - similarity heatmaps
        """
        B, D, H, W = x.shape
        
        # Reshape to [B, H, W, D]
        features = x.permute(0, 2, 3, 1)
        features_norm = F.normalize(features, dim=-1)
        
        # Normalize prototypes
        prototypes_norm = F.normalize(self.prototypes, dim=-1)  # [C, J, D]
        
        # Compute similarity at each spatial location
        # Result: [B, H, W, C, J]
        activations = torch.einsum('bhwd,cjd->bhwcj', features_norm, prototypes_norm)
        
        # Permute to [B, C, J, H, W] for easier visualization
        # Use contiguous() after permute to ensure memory layout is optimized
        activations = activations.permute(0, 3, 4, 1, 2).contiguous()
        
        return activations
    
    def get_orthogonality_loss(self) -> torch.Tensor:
        """
        Encourage prototypes within the same class to be orthogonal.
        """
        if not self.use_orthogonality_loss:
            return torch.tensor(0.0, device=self.prototypes.device)
        
        # Normalize prototypes
        prototypes_norm = F.normalize(self.prototypes, dim=-1)  # [C, J, D]
        
        # For each class, compute pairwise cosine similarity between prototypes
        class_losses = []
        for c in range(self.num_classes):
            proto_c = prototypes_norm[c]  # [J, D]
            
            # Cosine similarity matrix [J, J]
            sim_matrix = torch.mm(proto_c, proto_c.T)
            
            # Remove diagonal (self-similarity)
            off_diag = sim_matrix - torch.diag(sim_matrix.diag())
            
            # L2 norm of off-diagonal elements
            loss = torch.norm(off_diag, p=2) / (self.num_prototypes * (self.num_prototypes - 1))
            class_losses.append(loss)
        
        return torch.mean(torch.stack(class_losses))
    
    def get_interpretable_prediction(self, x: torch.Tensor) -> Dict:
        """
        Get prediction with interpretable prototype activations.
        
        ARGUMENTS:
            x: Spatial feature map [B, D, H, W]
        
        RETURNS:
            Dictionary with predictions and prototype explanations
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            
            # Get prototype activations (already contiguous from get_prototype_activations)
            activations = self.get_prototype_activations(x)  # [B, C, J, H, W]
            
            # Use reshape instead of view for safety (handles any memory layout)
            B, C, J, H, W = activations.shape
            
            # Flatten spatial dimensions: [B, C, J, H*W]
            flattened = activations.reshape(B, C, J, H * W)
            
            # Find max activation across spatial dimensions for each class-prototype
            # Result: [B, C, J] - best match score for each prototype
            max_per_prototype, _ = torch.max(flattened, dim=3)
            
            # Find which prototype is most active for each class
            # Result: [B, C] - max activation value, and [B, C] - which prototype index
            max_activations_per_class, best_prototype_idx = torch.max(max_per_prototype, dim=2)
            
        return {
            'logits': logits,
            'probabilities': probabilities,
            'prototype_activations': activations,
            'max_activations': max_activations_per_class,
            'best_prototypes': best_prototype_idx
        }
    

class BirdMAEModel(nn.Module):
    """
    Complete Bird-MAE model with prototypical probing.
    
    This is the main model class that other scripts should import.
    It provides a clean interface for training and inference.
    """
    
    def __init__(self, cfg, num_classes: int):
        """
        ARGUMENTS:
            cfg: Configuration object with model parameters
            num_classes: Number of bird species (C)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.probe = PrototypicalProbe(
            num_classes=num_classes,
            num_prototypes=getattr(cfg, 'num_prototypes', 20),
            feature_dim=cfg.birdmae_input_dim,
            use_orthogonality_loss=getattr(cfg, 'use_orthogonality_loss', True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        ARGUMENTS:
            x: Spatial feature map [B, D, H, W]
        
        RETURNS:
            logits: [B, C]
        """
        return self.probe(x)
    
    def get_orthogonality_loss(self) -> torch.Tensor:
        """Get orthogonality regularization loss."""
        return self.probe.get_orthogonality_loss()
    
    def get_prototype_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get prototype activation maps for interpretability."""
        return self.probe.get_prototype_activations(x)


def precompute_window_features(samples: List[Tuple[str, int]],
                               cfg) -> Tuple[Dict[str, np.ndarray], Dict, Dict]:
    """
    Precompute spatial features for sliding windows from each audio file.

    Features are cached to disk so extraction only happens once.

    RETURNS:
        - window_features: lazy-loading dict of window_key -> feature map
        - window_to_label: dict of window_key -> class label
        - window_to_path: dict of window_key -> original audio path
    """

    print("\n" + "=" * 70)
    print("Precomputing Sliding Window Features (memory-efficient)")
    print("=" * 70)

    class _LazyFeatureDict:
        def __init__(self, file_map):
            self._files = file_map

        def __getitem__(self, key):
            return np.load(self._files[key])

        def __contains__(self, key):
            return key in self._files

        def __len__(self):
            return len(self._files)

        def keys(self):
            return self._files.keys()

        def values(self):
            for k in self._files:
                yield self[k]

        def items(self):
            for k in self._files:
                yield k, self[k]

    out_dir = os.path.join(cfg.checkpoint_directory, "window_features_disk")

    label_path = os.path.join(
        cfg.checkpoint_directory,
        "window_to_label.npy"
    )

    path_map_path = os.path.join(
        cfg.checkpoint_directory,
        "window_to_path.npy"
    )

    # ============================================================
    # CACHE LOAD
    # ============================================================

    if (
        os.path.exists(out_dir)
        and len([f for f in os.listdir(out_dir) if f.endswith(".npy")]) > 0
        and os.path.exists(label_path)
        and os.path.exists(path_map_path)
    ):

        print(f"  Found cached features at {out_dir}")

        window_to_label = np.load(
            label_path,
            allow_pickle=True
        ).item()

        window_to_path = np.load(
            path_map_path,
            allow_pickle=True
        ).item()

        window_to_file = {}

        for fname in os.listdir(out_dir):
            if fname.endswith(".npy"):
                window_key = fname[:-4]
                window_to_file[window_key] = os.path.join(out_dir, fname)

        print(f"  Loaded {len(window_to_file)} cached windows")

        if window_to_file:
            sample_key = next(iter(window_to_file))
            sample_features = np.load(window_to_file[sample_key])

            print(f"  Feature map shape: {sample_features.shape}")

        return (
            _LazyFeatureDict(window_to_file),
            window_to_label,
            window_to_path
        )

    # ============================================================
    # FEATURE EXTRACTION
    # ============================================================

    window_duration = getattr(cfg, 'window_duration', 5.0)
    step_duration = getattr(cfg, 'step_duration', 2.5)

    unique_paths = list(set([path for path, _ in samples]))

    print(f"  Unique audio files: {len(unique_paths)}")
    print(f"  Window duration: {window_duration}s")
    print(f"  Step duration: {step_duration}s")

    os.makedirs(out_dir, exist_ok=True)

    extractor = BirdMAEExtractor(cfg)

    window_to_label = {}
    window_to_file = {}
    window_to_path = {}

    stats = {
        'total_windows': 0,
        'filtered_windows': 0,
        'files_processed': 0
    }

    path_to_label = {
        path: label for path, label in samples
    }

    for path in tqdm(
        unique_paths,
        desc="  Processing files",
        unit="file"
    ):

        try:
            window_result = extractor.precompute_window_features_for_file(
                path,
                window_duration=window_duration,
                step_duration=step_duration
            )

            label = path_to_label[path]

            for window_key, data in window_result.items():

                stats['total_windows'] += 1

                feature_file = os.path.join(
                    out_dir,
                    f"{window_key}.npy"
                )

                np.save(
                    feature_file,
                    data['features'].astype(np.float32)
                )

                window_to_label[window_key] = label
                window_to_file[window_key] = feature_file
                window_to_path[window_key] = path

                stats['filtered_windows'] += 1

            stats['files_processed'] += 1

            extractor.clear_cache()

        except Exception as e:

            print(f"    Warning: Failed to process {path}: {e}")

            try:
                features = extractor.get_spatial_features(path)

                window_key = os.path.basename(path)

                feature_file = os.path.join(
                    out_dir,
                    f"{window_key}.npy"
                )

                np.save(
                    feature_file,
                    features.astype(np.float32)
                )

                window_to_label[window_key] = path_to_label[path]
                window_to_file[window_key] = feature_file
                window_to_path[window_key] = path

                stats['filtered_windows'] += 1

            except Exception as inner_e:
                print(f"      Fallback extraction failed: {inner_e}")

    # ============================================================
    # SAVE CACHE METADATA
    # ============================================================

    np.save(
        label_path,
        window_to_label,
        allow_pickle=True
    )

    np.save(
        path_map_path,
        window_to_path,
        allow_pickle=True
    )

    print(f"\n  Processed {stats['files_processed']}/{len(unique_paths)} files")
    print(f"  Generated {stats['filtered_windows']}/{stats['total_windows']} windows")
    print(f"  Total cached windows: {len(window_to_file)}")

    if len(window_to_file) > 0:

        sample_key = next(iter(window_to_file))

        sample_features = np.load(
            window_to_file[sample_key]
        )

        print(f"  Feature map shape: {sample_features.shape}")

    return (
        _LazyFeatureDict(window_to_file),
        window_to_label,
        window_to_path
    )

def create_window_dataloader(samples: List[Tuple[str, int]],
                              window_features: Dict[str, np.ndarray],
                              window_to_label: Dict[str, int],
                              window_to_path: Dict[str, str],
                              cfg,
                              shuffle: bool = False,
                              include_outliers: bool = False) -> torch.utils.data.DataLoader:
    """
    Create a PyTorch DataLoader for window-based features.

    The cache can contain windows from known and outlier species. This function
    only selects windows whose original audio path belongs to the provided
    samples split. By default, negative labels are skipped because the
    supervised prototypical probe has outputs only for known classes.
    """
    from torch.utils.data import DataLoader

    allowed_paths = {path for path, _ in samples}
    window_list = []
    skipped_outliers = 0
    skipped_missing_path = 0

    for key in window_features.keys():
        original_path = window_to_path.get(key)
        if original_path is None:
            skipped_missing_path += 1
            continue

        if original_path not in allowed_paths:
            continue

        label = window_to_label[key]
        if label < 0 and not include_outliers:
            skipped_outliers += 1
            continue

        window_list.append((key, label))

    if shuffle:
        window_list = shuffle_func(window_list, random_state=42)

    if len(window_list) == 0:
        raise ValueError(
            "No windows matched the provided samples after filtering. "
            "Check that window_to_path.npy matches the current cache and split."
        )

    labels = [label for _, label in window_list]
    print(f"  Matched {len(window_list)} windows")
    print(f"  Labels range: {min(labels)} to {max(labels)}")
    if skipped_outliers:
        print(f"  Kept {skipped_outliers} outlier windows separate from supervised training")
    if skipped_missing_path:
        print(f"  Warning: Skipped {skipped_missing_path} windows without path metadata")

    dataset = WindowDataset(window_list, window_features)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, windows, features):
        self.windows = windows
        self.features = features
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window_key, label = self.windows[idx]
        feature_map = self.features[window_key]
        feature_tensor = torch.tensor(feature_map, dtype=torch.float32).permute(2, 0, 1)
        return feature_tensor, label


class SpatialAudioDataset(torch.utils.data.Dataset):
    """ Dataset for spatial feature maps (prototypical probing). """
    def __init__(self, samples: List[Tuple[str, int]], spatial_features: Dict[str, np.ndarray]):
        """ ARGUMENTS: samples: List of (audio_path, label) tuples spatial_features:
        Dictionary mapping path -> spatial features [H, W, D] """
        self.samples = samples
        self.spatial_features = spatial_features
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path, label = self.samples[idx]
        feature_map = self.spatial_features[audio_path]  # [H, W, D]
        feature_tensor = torch.tensor(feature_map, dtype=torch.float32).permute(2, 0, 1)
        return feature_tensor, label

