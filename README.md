# Bird Novelty Detection

This project contains three pipelines:

- **Visual**: EfficientNet-B0 embeddings + ArcFace + centroid/Mahalanobis outlier detection
- **Audio**: Bird-MAE features + prototypical probe + confidence-based outlier detection
- **Multimodal**: late fusion of saved visual, audio, and metadata scores

All commands below are run from the project root.

## 1. Environment

Use Python 3.10.

### DirectML setup

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install torch-directml==0.2.0.dev230426
pip install -r model_pipelines\requirements.txt
```

### CPU setup

For CPU-only installs, remove the `torch-directml` line from `model_pipelines/requirements.txt`, then install a CPU PyTorch build before the rest of the requirements.

### CUDA setup

For NVIDIA CUDA installs, install the CUDA PyTorch build first, remove the `torch-directml` line from `model_pipelines/requirements.txt`, then install the remaining requirements.

## 2. Device Selection

By default, the code chooses:

```text
DirectML -> CUDA -> CPU
```

You can force a device for the current terminal session:

```cmd
set MMEB_DEVICE=cpu
set MMEB_DEVICE=cuda
set MMEB_DEVICE=directml
```

If `MMEB_DEVICE` is not set and DirectML is unavailable, the code falls back automatically to CUDA or CPU.

## 3. Required Project Layout

Expected folders:

```text
data/
  clean_csv/
  processed/
metadata/
model_pipelines/
```

Visual CSV outputs are written under:

```text
data/processed/
```

Visual checkpoints and results are written under:

```text
checkpoints/
results/
```

Audio checkpoints and results are written under:

```text
audio_checkpoints/
audio_results/
```

Multimodal outputs are written under:

```text
results/multimodal_results/
```

## 4. Download iNaturalist Images

Before running the visual pipeline, place the cleaned source CSVs in:

```text
data/clean_csv/
```

The downloader expects these files:

```text
blackbird_clean.csv
blue_tit_clean.csv
crow_clean.csv
flamingo_clean.csv
great_tit_clean.csv
mallard_clean.csv
robin_clean.csv
sparrow_clean.csv
starling_clean.csv
thrush_clean.csv
toucan_clean.csv
```

Each input CSV must contain:

```text
image_url
```

Run:

```cmd
python -m model_pipelines.data.download_inaturalist_images
```

The script:

- downloads image files into species folders under `data/processed/`
- writes updated CSVs with `image_path` instead of `image_url`
- writes failures to `data/processed/download_failures.log`
- skips already downloaded files on reruns

If an expected input CSV is missing, it raises a `FileNotFoundError` with the exact missing path.

## 5. Configure The Visual Pipeline

Visual settings live in:

```text
model_pipelines/config.yaml
```

Important fields:

```yaml
data:
  data_root: "data/processed"
```

The path is project-relative. It can also be overridden per run:

```cmd
python -m model_pipelines.main train visual --data-root path\to\data
```

## 6. Run The Visual Pipeline

### Full visual training + evaluation

```cmd
python -m model_pipelines.main train visual
```

or directly:

```cmd
python -m model_pipelines.pipelines.run_visual
```

### Recompute calibration without retraining

```cmd
python -m model_pipelines.main train visual --skip-training
```

### Evaluate an existing visual checkpoint

```cmd
python -m model_pipelines.main evaluate visual
```

### Visual outputs

```text
checkpoints/best_model.pt
checkpoints/train_embeddings.npy
checkpoints/centroids.npy
checkpoints/covariances.npy
checkpoints/centroid_threshold.npy
checkpoints/classes.json
results/metrics.json
results/visual_scores_known.npy
results/visual_scores_outlier.npy
results/visual_threshold.npy
results/centroid_roc_pr.png
results/centroid_distribution.png
results/per_species_distributions.png
results/embedding_space.png
```

New visual checkpoints are saved in a CPU-portable format even if training runs on DirectML.

## 7. Run The Audio Pipeline

The audio pipeline expects Xeno-Canto-style folders under:

```text
data/processed/xenocanto_data/
```

Each species folder must contain:

```text
metadata.json
audio files referenced by metadata.json
```

### Full audio training + evaluation

```cmd
python -m model_pipelines.pipelines.run_audio
```

or:

```cmd
python -m model_pipelines.main train audio
```

### Evaluate from an existing audio checkpoint

```cmd
python -m model_pipelines.main evaluate audio
```

### Audio outputs

```text
audio_checkpoints/classes.json
audio_checkpoints/best_prototypical_probe.pt
audio_checkpoints/threshold.npy
audio_checkpoints/window_features_disk/
audio_checkpoints/window_to_label.npy
audio_checkpoints/window_to_path.npy
audio_results/results.json
audio_results/audio_scores_known.npy
audio_results/audio_scores_outlier.npy
audio_results/audio_threshold.npy
```

## 8. Run Multimodal Late Fusion

The multimodal pipeline requires:

```text
results/visual_scores_known.npy
results/visual_scores_outlier.npy
audio_results/audio_scores_known.npy
audio_results/audio_scores_outlier.npy
metadata/multimodal_dataset.csv
```

Run:

```cmd
python -m model_pipelines.main evaluate multimodal
```

The CSV at `metadata/multimodal_dataset.csv` must contain valid project-relative paths for its metadata columns, for example:

```text
image_metadata_path
audio_metadata_path
```

If a required score file, dataset CSV, or metadata CSV is missing, the pipeline raises a `FileNotFoundError` with the exact expected path.

Multimodal outputs:

```text
results/multimodal_results/fusion_results.json
results/multimodal_results/fusion_results.csv
results/multimodal_results/confusion_table.csv
results/multimodal_results/per_species_results.csv
results/multimodal_results/roc_curves.png
```

## 9. Prediction Commands

### Visual prediction

Single image:

```cmd
python -m model_pipelines.main predict visual path\to\image.jpg
```

Folder:

```cmd
python -m model_pipelines.main predict visual path\to\folder --save
```

### Audio prediction

Single audio file:

```cmd
python -m model_pipelines.main predict audio path\to\clip.wav
```

Folder:

```cmd
python -m model_pipelines.main predict audio path\to\folder --save
```

## 10. Recommended Run Order

```cmd
python -m model_pipelines.data.download_inaturalist_images
python -m model_pipelines.main train visual
python -m model_pipelines.pipelines.run_audio
python -m model_pipelines.main evaluate multimodal
```

## 11. Common Errors

### `PrivateUse1` / DirectML checkpoint error

An old visual checkpoint may have been saved with DirectML tensors inside it. Retrain once with the current code so `best_model.pt` is saved in CPU-portable form.

### Missing visual CSV

The visual loader now reports the exact missing CSV path. Run the image downloader first or correct `data_root`.

### Missing multimodal metadata file

The multimodal CSV may reference metadata files that do not exist in the repository. Fix the path inside `metadata/multimodal_dataset.csv` or add the missing files.

### Running from another folder

Core project paths are resolved from the repository root. Use the commands above from the root directory for the cleanest workflow.
