"""
Audio dataset loading from Xeno-Canto JSON files.

This module handles:
1. Loading metadata.json files from Xeno-Canto downloads
2. Parsing different JSON structures (array, object with recordings key)
3. Constructing audio file paths matching download script naming
4. Creating PyTorch Dataset from precomputed embeddings
5. Building train/val/test splits

JSON STRUCTURES HANDLED:
    Format 1 (Array):           Format 2 (Object):
    [                           {
      { "id": "123", ... },       "recordings": [
      { "id": "456", ... }          { "id": "123", ... },
    ]                             ]
                                }
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Union
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle

from model_pipelines.utils.device_utils import should_pin_memory


def load_audio_from_json(json_path: Union[str, Path], label: int) -> List[Tuple[str, int]]:
    """
    Load audio file paths from Xeno-Canto metadata.json.
    
    Reads the JSON metadata file and constructs paths to the actual MP3 audio files.
    
    ARGUMENTS:
        json_path: Path to metadata.json file
        label: Integer label for this species (0,1,2... for native, -1,-2... for outliers)
    
    RETURNS:
        List of (audio_path, label) tuples for files that exist on disk
    
    EXAMPLE:
        samples = load_audio_from_json("xenocanto_data/native_Blackbird/metadata.json", 0)
        # Returns: [("/path/to/1102367_A.mp3", 0), ...]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        recordings = data
    elif isinstance(data, dict) and 'recordings' in data:
        recordings = data['recordings']
    else:
        recordings = [data] if isinstance(data, dict) else []
    
    audio_dir = Path(json_path).parent
    samples = []
    
    for rec in recordings:
        recording_id = rec.get('id')
        quality = rec.get('q', '?')
        
        if not recording_id:
            continue
        
        # Construct filename matching download script: {id}_{quality}.mp3
        audio_filename = f"{recording_id}_{quality}.mp3"
        audio_path = audio_dir / audio_filename
        
        # Fallback to original filename if our pattern doesn't match
        if not audio_path.exists():
            original_filename = rec.get('file-name')
            if original_filename:
                audio_path = audio_dir / original_filename
        
        if audio_path.exists():
            samples.append((str(audio_path), label))
    
    return samples


def build_audio_splits(cfg) -> Tuple[List, List, List, List]:
    """
    Build train/validation/test splits from JSON files.
    
    PROCESS:
    1. Iterates through native_folders to load all training species
    2. For each species, loads all audio files from metadata.json
    3. Splits files into train/val/test using configured ratios
    4. Loads outlier species files (all go to test set)
    
    ARGUMENTS:
        cfg: AudioConfig object with data_root, native_folders, etc.
    
    RETURNS:
        train_samples: List of (path, label) for training
        val_samples: List of (path, label) for validation
        test_known: List of (path, label) for known species test
        test_outlier: List of (path, label) for outlier species test
    """
    train_all, val_all, test_known_all = [], [], []
    outlier_samples = []
    
    print("\n" + "="*70)
    print("Loading Audio Data from JSON Files")
    print("="*70)
    print(f"{'Species':<30} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 70)
    
    # Load known species (native birds for training)
    for label, folder_name in enumerate(cfg.native_folders):
        species_name = cfg.folder_to_species.get(folder_name, folder_name)
        json_path = os.path.join(cfg.data_root, folder_name, "metadata.json")
        
        if not os.path.exists(json_path):
            continue
        
        samples = load_audio_from_json(json_path, label)
        if not samples:
            continue
        
        # Shuffle and split
        samples = shuffle(samples, random_state=42)
        n_total = len(samples)
        n_train = int(n_total * cfg.training_split_ratio)
        n_val = int(n_total * cfg.validation_split_ratio)
        
        train = samples[:n_train]
        val = samples[n_train:n_train + n_val]
        test = samples[n_train + n_val:]
        
        train_all.extend(train)
        val_all.extend(val)
        test_known_all.extend(test)
        
        print(f"  {species_name:<30} {n_total:>6} {len(train):>6} {len(val):>6} {len(test):>6}")
    
    # Load outlier species (test only)
    label = -1
    for folder_name in cfg.outlier_folders:
        species_name = cfg.folder_to_species.get(folder_name, folder_name)
        json_path = os.path.join(cfg.data_root, folder_name, "metadata.json")
        
        if not os.path.exists(json_path):
            continue
        
        samples = load_audio_from_json(json_path, label)
        label -= 1
        
        if samples:
            outlier_samples.extend(samples)
            print(f"  {species_name:<30} {len(samples):>6} {'(outlier)':>20}")
    
    print("-" * 70)
    print(f"\n  TRAIN TOTAL:     {len(train_all):>6}")
    print(f"  VAL TOTAL:       {len(val_all):>6}")
    print(f"  TEST KNOWN:      {len(test_known_all):>6}")
    print(f"  TEST OUTLIER:    {len(outlier_samples):>6}")
    
    return train_all, val_all, test_known_all, outlier_samples

def build_audio_splits_with_windows(cfg) -> Tuple[List, List, List, List]:
    """
    Build train/validation/test splits with sliding window support.
    
    This function works exactly like build_audio_splits but returns the same
    (path, label) pairs. The actual window expansion happens during feature
    extraction, not during splitting.
    
    ARGUMENTS:
        cfg: AudioConfig object with data_root, native_folders, etc.
    
    RETURNS:
        train_samples: List of (path, label) for training
        val_samples: List of (path, label) for validation
        test_known: List of (path, label) for known species test
        test_outlier: List of (path, label) for outlier species test
    """
    train_all, val_all, test_known_all = [], [], []
    outlier_samples = []
    
    print("\n" + "="*70)
    print("Loading Audio Data from JSON Files (with Sliding Window Support)")
    print("="*70)
    print(f"  Window duration: {getattr(cfg, 'window_duration', 5.0)}s")
    print(f"  Step duration: {getattr(cfg, 'step_duration', 2.5)}s")
    print(f"{'Species':<30} {'Files':>6} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 70)
    
    # Load known species (native birds for training)
    for label, folder_name in enumerate(cfg.native_folders):
        species_name = cfg.folder_to_species.get(folder_name, folder_name)
        json_path = os.path.join(cfg.data_root, folder_name, "metadata.json")
        
        if not os.path.exists(json_path):
            continue
        
        # Use existing load_audio_from_json - returns (path, label) per file
        samples = load_audio_from_json(json_path, label)
        if not samples:
            continue
        
        # Shuffle and split at the file level
        samples = shuffle(samples, random_state=42)
        n_total = len(samples)
        n_train = int(n_total * cfg.training_split_ratio)
        n_val = int(n_total * cfg.validation_split_ratio)
        
        train = samples[:n_train]
        val = samples[n_train:n_train + n_val]
        test = samples[n_train + n_val:]
        
        train_all.extend(train)
        val_all.extend(val)
        test_known_all.extend(test)
        
        # Estimate number of windows for display (rough estimate)
        est_windows_per_file = max(1, int(getattr(cfg, 'window_duration', 5.0) / getattr(cfg, 'step_duration', 2.5)))
        print(f"  {species_name:<30} {n_total:>6} {len(train):>6} {len(val):>6} {len(test):>6} (~{len(train)*est_windows_per_file} train windows)")
    
    # Load outlier species (test only)
    label = -1
    for folder_name in cfg.outlier_folders:
        species_name = cfg.folder_to_species.get(folder_name, folder_name)
        json_path = os.path.join(cfg.data_root, folder_name, "metadata.json")
        
        if not os.path.exists(json_path):
            continue
        
        samples = load_audio_from_json(json_path, label)
        label -= 1
        
        if samples:
            outlier_samples.extend(samples)
            print(f"  {species_name:<30} {len(samples):>6} {'(outlier)':>20}")
    
    print("-" * 70)
    print(f"\n  TRAIN TOTAL:     {len(train_all)} files")
    print(f"  VAL TOTAL:       {len(val_all)} files")
    print(f"  TEST KNOWN:      {len(test_known_all)} files")
    print(f"  TEST OUTLIER:    {len(outlier_samples)} files")
    print(f"\n  Note: Each file will be split into windows during feature extraction")
    
    return train_all, val_all, test_known_all, outlier_samples


class AudioEmbeddingDataset(Dataset):
    """
    PyTorch Dataset for precomputed audio embeddings.
    
    This dataset DOES NOT process audio directly. Instead, it expects
    precomputed Bird-MAE embeddings. This design:
    - Speeds up training (no repeated audio processing)
    - Reduces memory usage (embeddings are smaller than raw audio)
    - Allows multiple training runs without reprocessing
    """
    
    def __init__(self, samples: List[Tuple[str, int]], embeddings: Dict[str, np.ndarray]):
        """
        Initialize dataset.
        
        ARGUMENTS:
            samples: List of (audio_path, label) tuples
            embeddings: Dictionary mapping audio_path to embedding vector
        """
        self.samples = samples
        self.embeddings = embeddings
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return (embedding_tensor, label)."""
        audio_path, label = self.samples[idx]
        embedding = self.embeddings[audio_path]
        return torch.tensor(embedding, dtype=torch.float32), label


def create_audio_loader(samples: List[Tuple[str, int]], 
                        embeddings: Dict[str, np.ndarray],
                        cfg,
                        shuffle: bool = False) -> DataLoader:
    """
    Create a PyTorch DataLoader from samples and precomputed embeddings.
    
    ARGUMENTS:
        samples: List of (audio_path, label) tuples
        embeddings: Dictionary mapping path -> embedding
        cfg: Configuration object with batch_size
        shuffle: Whether to shuffle the data
    
    RETURNS:
        DataLoader that yields (embedding_batch, label_batch)
    """
    dataset = AudioEmbeddingDataset(samples, embeddings)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=should_pin_memory(),
        drop_last=False
    )