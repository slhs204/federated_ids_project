# 21-Day Sprint Plan: Federated IDS Project

## 📅 Timeline Overview

**Total Duration:** 21 days (3 weeks)  
**Hardware:** MSI Vector 16 HX (RTX 5070 Ti, 12GB VRAM)  
**Daily Commitment:** 4-6 hours  
**Submission Deadline:** Day 21 (Final Class)

---

## Week 1: Foundation (Days 1-7)

### Day 1: Environment Setup & Data Download ⏱️ 4-5 hours

**Morning (2-3 hours)**
- [ ] Install Conda/Miniconda
- [ ] Create Python 3.10 environment
- [ ] Install PyTorch 2.3 + CUDA 12.4
- [ ] Install Flower, Gradio, and dependencies
- [ ] Test GPU availability (`nvidia-smi`, `torch.cuda.is_available()`)

**Afternoon (2 hours)**
- [ ] Download CICIDS2017 dataset (~10GB)
  - Source: https://www.unb.ca/cic/datasets/ids-2017.html
  - Files: All CSV files (Monday to Friday)
- [ ] Download UNSW-NB15 dataset (~2GB)
  - Source: https://research.unsw.edu.au/projects/unsw-nb15-dataset
  - Files: Training and testing sets

**Deliverables:**
```
✓ environment.yml or requirements.txt
✓ data/raw/cicids2017/ (10GB)
✓ data/raw/unsw-nb15/ (2GB)
✓ screenshots/gpu_test.png
```

**Troubleshooting Time:** 30 minutes

---

### Day 2: Data Exploration & Understanding ⏱️ 5-6 hours

**Morning (3 hours)**
- [ ] Create EDA notebook: `notebooks/01_eda.ipynb`
- [ ] Load CICIDS2017 CSV files
- [ ] Analyze:
  - [ ] Dataset shape (rows, columns)
  - [ ] Feature distributions (histograms)
  - [ ] Class balance (bar chart)
  - [ ] Missing values (heatmap)
  - [ ] Correlation matrix (select top 20 features)

**Afternoon (2-3 hours)**
- [ ] Repeat for UNSW-NB15
- [ ] Compare feature sets:
  - [ ] Common features
  - [ ] Unique features
  - [ ] Feature name mapping table
- [ ] Document findings in `docs/data_analysis_report.md`

**Deliverables:**
```
✓ notebooks/01_eda.ipynb (with visualizations)
✓ figures/class_distribution_cicids.png
✓ figures/class_distribution_unsw.png
✓ figures/correlation_heatmap.png
✓ docs/data_analysis_report.md (2-3 pages)
```

**Key Insights to Document:**
- Class imbalance ratio
- Most important features (from correlation)
- Outliers and anomalies
- Data quality issues

---

### Day 3-4: Data Preprocessing (Part 1) ⏱️ 8-10 hours

**Day 3 Morning (3 hours)**
- [ ] Create `src/data_processing/preprocessor.py`
- [ ] Implement:
  - [ ] Missing value handling (drop or impute)
  - [ ] Infinite value handling (replace with max/min)
  - [ ] Label encoding (string → integer)
  - [ ] Feature selection (remove highly correlated features)

**Day 3 Afternoon (2-3 hours)**
- [ ] Implement stratified split (70/15/15)
  - [ ] Set random seed = 42
  - [ ] Verify class distribution in each split
- [ ] Apply StandardScaler (fit on train, transform all)
- [ ] Save processed data:
  ```
  data/processed/cicids2017_train.pkl
  data/processed/cicids2017_val.pkl
  data/processed/cicids2017_test.pkl
  data/processed/scaler.pkl
  ```

**Day 4 Morning (3 hours)**
- [ ] Create `src/data_processing/feature_alignment.py`
- [ ] Implement UNSW-NB15 → CICIDS2017 feature mapping:
  - [ ] Semantic mapping (e.g., 'srcip' → 'Source IP')
  - [ ] Zero-padding for missing features
  - [ ] Apply same scaler from CICIDS2017
- [ ] Save: `data/processed/unsw_nb15_test.pkl`

**Day 4 Afternoon (2 hours)**
- [ ] Verification tests:
  - [ ] Check no NaN/Inf in processed data
  - [ ] Verify feature dimensions (all 78 features)
  - [ ] Check class distribution preservation
  - [ ] Test train/val/test have no overlap
- [ ] Document preprocessing steps in `data/metadata/preprocessing_log.txt`

**Deliverables:**
```
✓ src/data_processing/preprocessor.py
✓ src/data_processing/feature_alignment.py
✓ data/processed/ (all .pkl files)
✓ data/metadata/feature_mapping.json
✓ tests/test_data_processing.py (unit tests)
```

---

### Day 5: Non-IID Client Partitioning ⏱️ 4-5 hours

**Morning (3 hours)**
- [ ] Create `src/data_processing/client_partitioner.py`
- [ ] Implement Dirichlet distribution partitioning (alpha=0.5)
  - [ ] 8 clients total
  - [ ] Client 0-2: High DDoS concentration (70%)
  - [ ] Client 3-5: High Botnet concentration (60%)
  - [ ] Client 6-7: Balanced distribution
- [ ] Visualize client distributions (stacked bar chart)

**Afternoon (2 hours)**
- [ ] Save client partitions:
  ```
  data/clients/client_0_train.pkl
  ...
  data/clients/client_7_train.pkl
  data/clients/distribution_stats.json
  ```
- [ ] Create visualization: `figures/client_distribution.png`
- [ ] Verify total samples = train set size

**Deliverables:**
```
✓ src/data_processing/client_partitioner.py
✓ data/clients/ (8 client files)
✓ figures/client_distribution.png
✓ data/metadata/non_iid_stats.json
```

---

### Day 6-7: Baseline Models ⏱️ 10-12 hours

**Day 6: Temporal CNN Implementation (5-6 hours)**
- [ ] Create `src/models/temporal_cnn.py`
- [ ] Implement architecture:
  - [ ] 3 Conv1D layers (64, 128, 256 channels)
  - [ ] Batch normalization after each conv
  - [ ] Multi-head attention (8 heads)
  - [ ] 2 FC layers (256→128→16)
  - [ ] Dropout (0.3)
- [ ] Test forward pass with dummy data
- [ ] Count parameters (~2.1M)

**Day 6 Afternoon: LSTM Implementation (2-3 hours)**
- [ ] Create `src/models/lstm_baseline.py`
- [ ] Implement:
  - [ ] Bidirectional LSTM (2 layers, hidden=128)
  - [ ] FC layer to 16 classes
- [ ] Test forward pass

**Day 7: Centralized Training (5-6 hours)**
- [ ] Create `experiments/train_centralized.py`
- [ ] Implement training loop:
  - [ ] Adam optimizer (lr=0.001)
  - [ ] CrossEntropyLoss
  - [ ] Learning rate scheduler (ReduceLROnPlateau)
  - [ ] Early stopping (patience=5)
- [ ] Train both models:
  ```bash
  python experiments/train_centralized.py --model temporal_cnn --epochs 30
  python experiments/train_centralized.py --model lstm --epochs 30
  ```
- [ ] Log metrics every epoch (TensorBoard)
- [ ] Save best model checkpoints

**Expected Results:**
- Temporal CNN: 94%+ accuracy (should take ~2-3 hours)
- LSTM: 92%+ accuracy (should take ~3-4 hours)

**Deliverables:**
```
✓ src/models/temporal_cnn.py
✓ src/models/lstm_baseline.py
✓ experiments/train_centralized.py
✓ results/baseline/temporal_cnn/best_model.pt
✓ results/baseline/lstm/best_model.pt
✓ results/baseline/*/training_curves.png
✓ results/baseline/*/metrics.json
```

---

## Week 2: Federated Learning (Days 8-14)

### Day 8-9: FL Implementation ⏱️ 10-12 hours

**Day 8 Morning: FL Client (3-4 hours)**
- [ ] Create `src/federated/fl_client.py`
- [ ] Implement `FlowerClient` class:
  - [ ] `get_parameters()` - extract model weights
  - [ ] `set_parameters()` - load model weights
  - [ ] `fit()` - local training (3 epochs)
  - [ ] `evaluate()` - local evaluation
- [ ] Test with dummy data

**Day 8 Afternoon: FL Server (3-4 hours)**
- [ ] Create `src/federated/fl_server.py`
- [ ] Implement:
  - [ ] `get_evaluate_fn()` - server-side evaluation
  - [ ] `create_strategy()` - FedAvg setup
  - [ ] `weighted_average()` - metric aggregation
- [ ] Test server initialization

**Day 9: Integration & Testing (4 hours)**
- [ ] Create `experiments/train_federated.py`
- [ ] Implement simulation mode:
  - [ ] Load 8 client data partitions
  - [ ] Create client_fn factory
  - [ ] Configure Ray for parallel execution
- [ ] Test with 3 rounds (quick validation)
- [ ] Debug any communication issues

**Deliverables:**
```
✓ src/federated/fl_client.py
✓ src/federated/fl_server.py
✓ experiments/train_federated.py
✓ tests/test_federated.py (integration tests)
```

---

### Day 10-11: FL Training (50 Rounds) ⏱️ 8-10 hours

**Day 10 Evening: Start Training**
- [ ] Launch FL training (will run overnight):
  ```bash
  nohup python experiments/train_federated.py \
      --model temporal_cnn \
      --strategy fedavg \
      --num_clients 8 \
      --num_rounds 50 \
      --local_epochs 3 \
      --batch_size 128 \
      --device cuda \
      > logs/fl_training.log 2>&1 &
  ```
- [ ] Monitor first 5 rounds to ensure stability
- [ ] Set up monitoring dashboard (TensorBoard)

**Day 11 Morning: Monitor & Checkpoint**
- [ ] Check training progress (should be ~Round 30-35)
- [ ] Verify no errors in log
- [ ] Analyze intermediate results

**Day 11 Afternoon: Complete & Evaluate**
- [ ] Training should complete (~Round 50)
- [ ] Extract final model
- [ ] Run evaluation on CICIDS test set
- [ ] Document results

**Expected Timeline:**
- Round 1-10: ~1-2 hours
- Round 11-30: ~3-4 hours
- Round 31-50: ~2-3 hours
- Total: ~6-8 hours

**Expected Results:**
- Final accuracy: 93-94%
- F1-Score: 92-93%
- Convergence: Smooth curve (no oscillations)

**Deliverables:**
```
✓ results/federated/fedavg/final_model.pt
✓ results/federated/fedavg/training_history.json
✓ results/federated/fedavg/round_*.pt (checkpoints)
✓ logs/fl_training.log
```

---

### Day 12: Comparison Experiments ⏱️ 6-8 hours

**Morning: FedProx Training (3-4 hours)**
- [ ] Modify `src/federated/strategies.py` to add FedProx
- [ ] Train with FedProx (proximal_mu=0.01):
  ```bash
  python experiments/train_federated.py --strategy fedprox
  ```
- [ ] Compare convergence with FedAvg

**Afternoon: Ablation Studies (3-4 hours)**
- [ ] Create `experiments/ablation_study.py`
- [ ] Experiment 1: Vary number of clients (4, 8, 12, 16)
- [ ] Experiment 2: Vary client sampling rate (0.5, 0.8, 1.0)
- [ ] Experiment 3: Vary local epochs (1, 3, 5)
- [ ] Record results in table

**Deliverables:**
```
✓ results/federated/fedprox/
✓ results/ablation/
✓ tables/ablation_results.csv
✓ figures/ablation_comparison.png
```

---

### Day 13-14: Cross-Dataset Evaluation ⏱️ 8 hours

**Day 13: UNSW-NB15 Testing (4 hours)**
- [ ] Create `src/evaluation/cross_dataset.py`
- [ ] Load trained FL model
- [ ] Evaluate on UNSW-NB15 test set
- [ ] Calculate metrics:
  - [ ] Accuracy, Precision, Recall, F1
  - [ ] Per-class metrics
  - [ ] Confusion matrix
- [ ] Analyze performance degradation

**Day 14: Visualization & Analysis (4 hours)**
- [ ] Create `src/evaluation/visualizer.py`
- [ ] Generate figures:
  - [ ] Confusion matrices (16×16 heatmaps)
  - [ ] ROC curves (One-vs-Rest)
  - [ ] PR curves
  - [ ] Training curves (loss, accuracy over rounds)
  - [ ] Bar chart: Primary vs Secondary performance
- [ ] Write analysis section in report

**Expected Results:**
- UNSW-NB15 F1: 85-87%
- Degradation: 6-8% from primary dataset
- Some attack types: 0% recall (document why)

**Deliverables:**
```
✓ src/evaluation/cross_dataset.py
✓ src/evaluation/visualizer.py
✓ results/evaluation/unsw_metrics.json
✓ figures/confusion_matrix_primary.png
✓ figures/confusion_matrix_secondary.png
✓ figures/roc_curves.png
✓ figures/pr_curves.png
✓ docs/cross_dataset_analysis.md
```

---

## Week 3: Deployment & Presentation (Days 15-21)

### Day 15-16: Gradio Deployment ⏱️ 8-10 hours

**Day 15: Backend Development (4-5 hours)**
- [ ] Create `src/deployment/inference.py`
- [ ] Implement:
  - [ ] Model loading (compressed version)
  - [ ] Feature preprocessing
  - [ ] Batch inference
  - [ ] Confidence thresholding
- [ ] Optimize inference speed (<50ms target)
- [ ] Test with sample data

**Day 15 Afternoon: PCAP Parser (2 hours)**
- [ ] Create `src/deployment/pcap_parser.py`
- [ ] Implement:
  - [ ] PCAP file reading (scapy)
  - [ ] Feature extraction (flow statistics)
  - [ ] Conversion to model input format
- [ ] Test with real PCAP files

**Day 16: Frontend Development (4-5 hours)**
- [ ] Complete `src/deployment/app.py`
- [ ] Design Gradio interface:
  - [ ] Tab 1: CSV upload
  - [ ] Tab 2: Manual feature input
  - [ ] Tab 3: About/documentation
- [ ] Add visualizations:
  - [ ] Top-5 predictions (bar chart)
  - [ ] Confidence meter
  - [ ] Risk level indicator
- [ ] Test all features
- [ ] Launch local demo

**Deliverables:**
```
✓ src/deployment/inference.py
✓ src/deployment/pcap_parser.py
✓ src/deployment/app.py
✓ screenshots/gradio_demo.png
✓ Demo video: demo.mp4 (2-3 minutes)
```

---

### Day 17: Docker & Cloud Deployment ⏱️ 4-6 hours

**Morning: Docker (2-3 hours)**
- [ ] Create `docker/Dockerfile`
- [ ] Create `docker/docker-compose.yml`
- [ ] Build image:
  ```bash
  docker build -t fl-ids:latest .
  ```
- [ ] Test container locally:
  ```bash
  docker run -p 7860:7860 --gpus all fl-ids:latest
  ```
- [ ] Verify GPU access inside container

**Afternoon: Hugging Face Spaces (2-3 hours)**
- [ ] Create Hugging Face account
- [ ] Create new Space (Gradio SDK)
- [ ] Upload:
  - [ ] app.py
  - [ ] Model checkpoint (compressed <50MB)
  - [ ] requirements.txt
  - [ ] README.md
- [ ] Configure Space settings
- [ ] Test public URL
- [ ] Update README with deployment link

**Deliverables:**
```
✓ docker/Dockerfile
✓ docker/docker-compose.yml
✓ Hugging Face Space URL
✓ docs/deployment_guide.md
```

---

### Day 18-19: Report Writing ⏱️ 12-14 hours

**Day 18: Report Structure & Content (6-7 hours)**

**Section 1: Introduction (1 hour)**
- [ ] Background & motivation
- [ ] Problem statement
- [ ] Research questions (RQ1, RQ2, RQ3)
- [ ] Contributions

**Section 2: Related Work (1 hour)**
- [ ] Review 5-8 papers on FL-IDS
- [ ] Comparison table (methods, datasets, results)
- [ ] Gap analysis

**Section 3: Methodology (2 hours)**
- [ ] System architecture diagram
- [ ] Data preprocessing pipeline
- [ ] FL protocol description
- [ ] Model architecture details
- [ ] Training procedure

**Section 4: Experimental Setup (1 hour)**
- [ ] Datasets description
- [ ] Hardware/software specs
- [ ] Hyperparameters table
- [ ] Evaluation metrics

**Section 5: Results (1-2 hours)**
- [ ] Primary dataset results (tables & figures)
- [ ] Cross-dataset results
- [ ] Comparison with baseline
- [ ] Ablation study results
- [ ] Statistical significance tests

**Day 19: Finalize Report (6-7 hours)**

**Section 6: Discussion (2 hours)**
- [ ] Answer RQs explicitly
- [ ] Analyze performance degradation
- [ ] Discuss limitations:
  - [ ] Dataset biases
  - [ ] Computational constraints
  - [ ] Deployment challenges
- [ ] Threats to validity

**Section 7: Conclusion (1 hour)**
- [ ] Summarize contributions
- [ ] Key findings
- [ ] Future work (specific ideas)

**Polish & References (3-4 hours)**
- [ ] Proofread entire document
- [ ] Check figure/table captions
- [ ] Format references (APA style)
- [ ] Add appendices (hyperparameters, etc.)
- [ ] Generate table of contents
- [ ] Convert to PDF

**Deliverables:**
```
✓ docs/report/main.md (15-20 pages)
✓ docs/report/main.pdf
✓ docs/report/references.bib (20-30 entries)
✓ All figures in high resolution (300 DPI)
```

---

### Day 20: Presentation Preparation ⏱️ 6-8 hours

**Morning: Slides Creation (4 hours)**

Create 15 slides (C-level briefing format):

1. **Title Slide** (1 slide)
   - [ ] Project title, name, date

2. **Problem Overview** (2 slides)
   - [ ] "Why" this matters (business impact)
   - [ ] Current limitations of IDS
   - [ ] Market opportunity

3. **Solution Approach** (2 slides)
   - [ ] Federated Learning concept (simple diagram)
   - [ ] Technical architecture (high-level)

4. **Data & Methodology** (2 slides)
   - [ ] Datasets overview
   - [ ] Training pipeline

5. **Key Results** (4 slides)
   - [ ] Primary metrics (big numbers!)
   - [ ] Cross-dataset generalization
   - [ ] Comparison chart
   - [ ] Live demo screenshot

6. **Business Value** (2 slides)
   - [ ] ROI calculation (cost savings)
   - [ ] Privacy compliance (GDPR/HIPAA)
   - [ ] Scalability benefits

7. **Deployment** (1 slide)
   - [ ] Architecture diagram
   - [ ] Real-time capabilities

8. **Q&A** (1 slide)

**Afternoon: Demo Prep (2-3 hours)**
- [ ] Prepare demo script
- [ ] Create sample test cases (benign & malicious)
- [ ] Record backup demo video (in case live fails)
- [ ] Practice transitions

**Evening: Rehearsal (2 hours)**
- [ ] Time presentation (target: 13-14 minutes)
- [ ] Practice Q&A responses
- [ ] Get feedback from friend/colleague
- [ ] Refine based on feedback

**Deliverables:**
```
✓ docs/presentation/slides.pptx
✓ docs/presentation/demo_script.md
✓ docs/presentation/backup_demo.mp4
✓ docs/presentation/qa_prep.md
```

---

### Day 21: Final Checks & Submission ⏱️ 4-6 hours

**Morning: Reproducibility Test (2-3 hours)**
- [ ] Fresh clone/extract of project
- [ ] Follow repro_guide.md step-by-step
- [ ] Verify all commands work
- [ ] Check all file paths
- [ ] Run quick training (5 rounds) to verify
- [ ] Fix any issues

**Midday: Package Submission (2 hours)**
- [ ] Create submission structure:
  ```
  submission/
  ├── code/                 # All source code
  ├── models/               # Trained model checkpoints
  ├── results/              # Figures, tables, logs
  ├── report/               # Final PDF report
  ├── presentation/         # Slides
  ├── deployment/           # Gradio app + Docker
  ├── repro_guide.md
  └── README.md
  ```
- [ ] Compress: `tar -czf submission.tar.gz submission/`
- [ ] Verify archive size < 500MB (exclude raw data)
- [ ] Create checksum: `sha256sum submission.tar.gz`

**Final Checklist:**
- [ ] All code runs without errors
- [ ] Report has no TODOs or placeholders
- [ ] All figures have captions
- [ ] References are complete and formatted
- [ ] Gradio app is deployed and accessible
- [ ] Presentation timing is 14-15 minutes
- [ ] Demo backup video prepared
- [ ] Submission file uploaded to course platform

**Afternoon: Class Presentation ⏱️ 20 minutes**
- [ ] Arrive early, test equipment
- [ ] Present for 15 minutes
- [ ] Demo for 2-3 minutes (within presentation)
- [ ] Q&A for 5 minutes

---

## 📊 Progress Tracking

Use this table to track daily progress:

| Day | Date | Tasks Completed | Hours Spent | Blockers | Status |
|-----|------|-----------------|-------------|----------|--------|
| 1 | ___ | ___ | ___ | ___ | ⬜ |
| 2 | ___ | ___ | ___ | ___ | ⬜ |
| ... | ___ | ___ | ___ | ___ | ⬜ |
| 21 | ___ | ___ | ___ | ___ | ⬜ |

**Status Legend:**
- ⬜ Not Started
- 🟡 In Progress
- ✅ Completed
- 🔴 Blocked

---

## 🎯 Success Metrics

**Minimum Viable Project:**
- [ ] FL training converges (accuracy > 90%)
- [ ] Cross-dataset F1 > 80%
- [ ] Gradio demo works
- [ ] Report is complete (15+ pages)
- [ ] Presentation is polished

**Target Excellence:**
- [ ] FL accuracy within 2% of baseline
- [ ] Cross-dataset F1 > 85%
- [ ] Inference latency < 50ms
- [ ] Published on Hugging Face Spaces
- [ ] Report has novel insights
- [ ] Presentation impresses C-level audience

---

## 🚨 Risk Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Data download slow | High | Medium | Use torrents, start early |
| GPU out of memory | High | Low | Reduce batch size, use mixed precision |
| Training doesn't converge | High | Low | Check data preprocessing, tune hyperparameters |
| Deployment fails | Medium | Low | Test early, have backup demo video |
| Time overrun | High | Medium | Follow schedule strictly, ask for help |

---

**Good luck! 🚀**

Remember:
- **Start early**, especially data download
- **Test frequently**, don't wait until the end
- **Ask for help** when stuck > 1 hour
- **Document everything** as you go
- **Backup your work** daily (Git, cloud storage)
