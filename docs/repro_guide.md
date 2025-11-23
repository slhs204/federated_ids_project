# Reproducibility Guide

## 🎯 Goal
This guide ensures anyone can reproduce the results of this Federated Learning-based Network Intrusion Detection System from scratch.

---

## 📋 Prerequisites

### Hardware Requirements
- **Minimum:** 
  - CPU: 4 cores
  - RAM: 16GB
  - GPU: 8GB VRAM (GTX 1080 Ti or better)
  - Storage: 50GB free space

- **Recommended (Used in this project):**
  - CPU: Intel i9-13xxx (16 threads)
  - RAM: 32GB
  - GPU: RTX 5070 Ti (12GB VRAM)
  - Storage: 100GB SSD

### Software Requirements
- OS: Ubuntu 20.04+ / Windows 10+ with WSL2 / macOS 12+
- Python: 3.10+
- CUDA: 12.4 (for GPU support)
- Conda or venv for environment management

---

## 🚀 Step-by-Step Reproduction

### Step 1: Environment Setup (15 minutes)

```bash
# 1. Clone or extract project
cd /path/to/federated_ids_project

# 2. Create Conda environment
conda create -n fl_ids python=3.10 -y
conda activate fl_ids

# 3. Install PyTorch with CUDA 12.4
pip install torch==2.3.0+cu124 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import flwr; print(f'Flower: {flwr.__version__}')"
```

**Expected Output:**
```
PyTorch: 2.3.0+cu124
CUDA: True
Flower: 1.8.0
```

---

### Step 2: Data Preparation (1-2 hours)

#### Option A: Download Original Datasets

```bash
# Create data directories
mkdir -p data/raw data/processed

# Download CICIDS2017 (10GB)
# Visit: https://www.unb.ca/cic/datasets/ids-2017.html
# Download all CSV files to data/raw/cicids2017/

# Download UNSW-NB15 (2GB)
# Visit: https://research.unsw.edu.au/projects/unsw-nb15-dataset
# Download to data/raw/unsw-nb15/
```

#### Option B: Use Preprocessed Data (if provided)

```bash
# If submission includes preprocessed data
tar -xzf data_processed.tar.gz -C data/
```

#### Preprocess Data

```bash
# Run preprocessing pipeline
python src/data_processing/preprocessor.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --seed 42

# Create client partitions (Non-IID)
python src/data_processing/client_partitioner.py \
    --data_path data/processed/cicids2017_train.pkl \
    --num_clients 8 \
    --alpha 0.5 \
    --output_dir data/clients \
    --seed 42
```

**Expected Output:**
```
✓ Processed CICIDS2017: 2,830,540 samples
  - Train: 1,981,378 (70%)
  - Val: 424,581 (15%)
  - Test: 424,581 (15%)

✓ Processed UNSW-NB15: 257,673 samples
  - Test: 257,673 (100%)

✓ Created 8 client partitions
  - Client 0: 247,672 samples (High DDoS)
  - Client 1: 247,672 samples (High Botnet)
  - ...
```

**Verification:**
```bash
# Check data files exist
ls -lh data/processed/
# Should see: cicids2017_train.pkl, cicids2017_val.pkl, cicids2017_test.pkl, unsw_nb15_test.pkl

ls -lh data/clients/
# Should see: client_0_train.pkl ... client_7_train.pkl
```

---

### Step 3: Baseline Training (2-3 hours)

Train centralized baseline models for comparison:

```bash
# Train Temporal CNN (centralized)
python experiments/train_centralized.py \
    --model temporal_cnn \
    --data_path data/processed \
    --epochs 30 \
    --batch_size 128 \
    --lr 0.001 \
    --device cuda \
    --output_dir results/baseline/temporal_cnn \
    --seed 42

# Train LSTM (centralized)
python experiments/train_centralized.py \
    --model lstm \
    --data_path data/processed \
    --epochs 30 \
    --batch_size 128 \
    --lr 0.001 \
    --device cuda \
    --output_dir results/baseline/lstm \
    --seed 42
```

**Expected Results (CICIDS2017 Test Set):**
```
Temporal CNN:
  - Accuracy: 94.2% ± 0.3%
  - F1-Macro: 92.8% ± 0.4%
  - Training time: ~2.5 hours

LSTM:
  - Accuracy: 92.7% ± 0.4%
  - F1-Macro: 90.5% ± 0.5%
  - Training time: ~3.2 hours
```

---

### Step 4: Federated Learning Training (6-8 hours)

Train with Federated Learning:

```bash
# FedAvg strategy (8 clients, 50 rounds)
python experiments/train_federated.py \
    --model temporal_cnn \
    --strategy fedavg \
    --num_clients 8 \
    --num_rounds 50 \
    --local_epochs 3 \
    --batch_size 128 \
    --fraction_fit 0.8 \
    --device cuda \
    --num_cpus 8 \
    --output_dir results/federated/fedavg \
    --data_path data/processed

# Monitor progress with TensorBoard (optional, in another terminal)
tensorboard --logdir results/federated/fedavg/logs
```

**Expected Output:**
```
[Round 1/50] Server Evaluation:
  Loss: 1.234
  Accuracy: 0.456 (45.6%)
  F1-Macro: 0.432

[Round 10/50] Server Evaluation:
  Loss: 0.543
  Accuracy: 0.812 (81.2%)
  F1-Macro: 0.789

...

[Round 50/50] Server Evaluation:
  Loss: 0.178
  Accuracy: 0.938 (93.8%)
  F1-Macro: 0.921

✅ Training completed!
📁 Results saved to: results/federated/fedavg
```

**Training Time Estimate:**
- RTX 5070 Ti: ~6-7 hours
- RTX 4070: ~8-10 hours
- RTX 3070: ~12-15 hours

---

### Step 5: Cross-Dataset Evaluation (30 minutes)

Test generalization on UNSW-NB15:

```bash
# Evaluate on secondary dataset
python src/evaluation/cross_dataset.py \
    --model_path results/federated/fedavg/final_model.pt \
    --primary_test data/processed/cicids2017_test.pkl \
    --secondary_test data/processed/unsw_nb15_test.pkl \
    --output_dir results/evaluation

# Generate visualizations
python src/evaluation/visualizer.py \
    --results_dir results/evaluation \
    --output_dir results/figures
```

**Expected Results:**

| Metric | CICIDS2017 (Primary) | UNSW-NB15 (Secondary) | Degradation |
|--------|----------------------|-----------------------|-------------|
| Accuracy | 93.8% | 86.3% | -7.5% |
| F1-Macro | 92.1% | 84.7% | -7.4% |
| F1-Weighted | 93.5% | 86.8% | -6.7% |

**Generated Figures:**
- `confusion_matrix_primary.png`
- `confusion_matrix_secondary.png`
- `roc_curves.png`
- `pr_curves.png`
- `training_curves.png`

---

### Step 6: Deployment (1 hour)

#### Local Deployment (Gradio)

```bash
# Launch Gradio interface
python src/deployment/app.py

# Access at: http://localhost:7860
# Public URL will be printed (e.g., https://xxxx.gradio.live)
```

#### Docker Deployment

```bash
# Build Docker image
cd docker/
docker build -t fl-ids:latest .

# Run container
docker run -p 7860:7860 --gpus all fl-ids:latest

# Access at: http://localhost:7860
```

#### Hugging Face Spaces Deployment

```bash
# 1. Create Hugging Face account and Space
# 2. Clone your Space locally
git clone https://huggingface.co/spaces/YOUR_USERNAME/fl-ids
cd fl-ids

# 3. Copy deployment files
cp ../src/deployment/app.py .
cp ../results/models/best_model.pt .
cp ../requirements.txt .

# 4. Create README.md with configuration
cat > README.md << EOF
---
title: FL-IDS
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
---
EOF

# 5. Push to Hugging Face
git add .
git commit -m "Deploy FL-IDS"
git push

# Your Space will be available at:
# https://huggingface.co/spaces/YOUR_USERNAME/fl-ids
```

---

### Step 7: Generate Report (3-4 hours)

```bash
# Run all analysis notebooks
jupyter notebook notebooks/

# Export key results
python scripts/generate_report.py \
    --results_dir results/ \
    --output docs/report/main.md

# Convert to PDF (requires pandoc)
cd docs/report/
pandoc main.md -o main.pdf --pdf-engine=xelatex
```

---

## 🔍 Verification Checklist

### Data Processing
- [ ] CICIDS2017 train/val/test splits are stratified (70/15/15)
- [ ] UNSW-NB15 features aligned to 78 dimensions
- [ ] Client partitions show Non-IID distribution
- [ ] No data leakage between train/test sets

### Training
- [ ] Random seeds fixed (seed=42)
- [ ] Baseline accuracy within ±1% of reported
- [ ] FL accuracy within ±2% of baseline
- [ ] Training loss converges smoothly

### Evaluation
- [ ] Confusion matrices show clear diagonal pattern
- [ ] ROC-AUC > 0.95 for primary dataset
- [ ] Secondary dataset F1 > 85%
- [ ] No class has 0% recall

### Deployment
- [ ] Gradio app loads successfully
- [ ] Inference latency < 100ms
- [ ] Predictions are deterministic (same input → same output)
- [ ] UI displays top-5 predictions correctly

---

## 🐛 Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:**
```bash
# Reduce batch size
--batch_size 64  # Instead of 128

# Or reduce number of parallel clients
--num_clients 4  # Instead of 8
```

### Issue 2: Slow Training
**Solution:**
```bash
# Enable mixed precision
# Add to train_federated.py:
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Issue 3: Data Download Fails
**Solution:**
```bash
# Use mirror sites or torrent downloads
# Check docs/data_sources.md for alternative links
```

### Issue 4: Port Already in Use (Gradio)
```
OSError: [Errno 48] Address already in use
```
**Solution:**
```bash
# Use different port
python src/deployment/app.py --port 7861
```

---

## 📊 Expected Resource Usage

| Phase | Time | GPU VRAM | Storage |
|-------|------|----------|---------|
| Data Download | 1-2h | - | 12GB |
| Data Processing | 30min | - | 8GB |
| Baseline Training | 2-3h | 4-6GB | 500MB |
| FL Training | 6-8h | 10-12GB | 2GB |
| Evaluation | 30min | 2GB | 100MB |
| Deployment | - | 2GB | 500MB |
| **Total** | **~12h** | **12GB** | **23GB** |

---

## 📝 Results Validation

To validate your results match the published ones:

```bash
# Run validation script
python scripts/validate_results.py \
    --baseline_dir results/baseline \
    --federated_dir results/federated \
    --reference_file results_reference.json

# Expected output:
✓ Baseline accuracy match: 94.2% vs 94.1% (Δ=0.1%)
✓ FL accuracy match: 93.8% vs 93.7% (Δ=0.1%)
✓ Cross-dataset F1 match: 86.3% vs 86.5% (Δ=0.2%)
✓ All metrics within tolerance (±2%)
```

---

## 🆘 Getting Help

If you encounter issues:

1. Check this guide first
2. Review error logs in `results/logs/`
3. Verify environment with `python scripts/check_environment.py`
4. Contact: [your-email@example.com]

---

## 📚 References

- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Flower FL: https://flower.dev/docs/
- PyTorch: https://pytorch.org/docs/stable/index.html

---

**Last Updated:** October 2025  
**Version:** 1.0  
**Hardware:** MSI Vector 16 HX (RTX 5070 Ti)
