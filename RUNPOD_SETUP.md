# RunPod GPU Setup Guide for DP-PEFT Experiments

Complete step-by-step guide to run DP-PEFT experiments on RunPod GPU servers (L40S, A100, etc.)

---

## Part 1: Initial Setup on RunPod

### Step 1: Create RunPod Account & Deploy GPU Pod

1. **Go to RunPod.io** and create an account
2. **Add credits** to your account (recommended: $50-100 for full experiments)
3. **Deploy a GPU Pod:**
   - Click "Deploy" → "GPU Pods"
   - **Recommended GPU:** L40S (48GB VRAM, ~$0.79/hr) or A100 (40GB, ~$1.14/hr)
   - **Template:** PyTorch 2.0+ (or Ubuntu 22.04 with CUDA)
   - **Container Disk:** 50GB minimum
   - **Volume Storage:** 100GB (optional, for persistent storage)
   - Click "Deploy On-Demand"

4. **Wait for pod to start** (~1-2 minutes)
5. **Connect via SSH or Web Terminal:**
   - Click "Connect" → Copy SSH command or use "Start Web Terminal"

### Step 2: Verify GPU Setup

```bash
# Check GPU
nvidia-smi

# Expected output: L40S with 48GB VRAM or similar
```

---

## Part 2: Clone and Setup Repository

### Step 3: Clone Repository

```bash
# Navigate to workspace
cd /workspace  # or ~/

# Clone repository
git clone https://github.com/YOUR_USERNAME/dp-peft-research.git
cd dp-peft-research

# Verify files
ls -la
```

### Step 4: Install Dependencies

```bash
# Update system
apt-get update && apt-get install -y git tmux htop

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies (this takes 5-10 minutes)
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import transformers, peft, opacus; print('All packages imported successfully!')"
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA: True
All packages imported successfully!
```

---

## Part 3: Quick Validation Test

### Step 5: Run Quick Test (5 minutes)

```bash
# Test basic functionality
python scripts/run_experiment.py \
    --model bert \
    --dataset agnews \
    --placement no_dp \
    --epsilon 8.0 \
    --epochs 2 \
    --batch_size 32 \
    --lr 1e-3 \
    --device cuda

# Should see:
# - Dataset downloading
# - Model loading
# - Training progress bars
# - Final accuracy printed
```

**Success criteria:** No errors, accuracy >50% after 2 epochs

---

## Part 4: Hyperparameter Tuning (Critical!)

### Step 6: Fix DP Hyperparameters

The current DP placements need hyperparameter tuning. Test improved settings:

```bash
# Create tmux session (so it keeps running if disconnected)
tmux new -s dp_tuning

# Test improved hyperparameters for Adapter-Only DP
python scripts/run_experiment.py \
    --model bert \
    --dataset agnews \
    --placement adapter_only \
    --epsilon 8.0 \
    --epochs 10 \
    --batch_size 32 \
    --lr 5e-4 \
    --device cuda \
    2>&1 | tee results/tuning_adapter_only.log

# Detach from tmux: Ctrl+B, then D
# Reattach: tmux attach -s dp_tuning
```

**Success criteria:** Test accuracy should reach >60% by epoch 10

### Step 7: Validate All Placements

Once hyperparameters work for adapter_only, test all placements:

```bash
# In tmux session
tmux new -s dp_validation

# Quick validation (10K samples, 10 epochs)
python scripts/run_all_placements.py \
    --model bert \
    --dataset agnews \
    --epsilon 8.0 \
    --epochs 10 \
    --batch_size 32 \
    --lr 5e-4 \
    2>&1 | tee results/validation_log.txt

# This takes ~3-4 hours
```

**Expected results:**
- No DP: ~90% accuracy
- Adapter-Only DP: ~70-80% accuracy
- Head+Adapter DP: ~65-75% accuracy
- Last-Layer DP: ~60-70% accuracy
- Full-Model DP: ~55-65% accuracy
- Partial Backbone DP: ~60-70% accuracy

---

## Part 5: Full Production Experiments

### Step 8: Run Complete Text Experiments

Once validation passes, run full experiments with complete dataset:

```bash
# Create persistent tmux session
tmux new -s dp_full_experiments

# Activate environment
cd /workspace/dp-peft-research  # or your path
source venv/bin/activate

# Run full experiments (120K samples, 20 epochs)
# This will take ~48-60 hours for all placements
python scripts/run_all_placements.py \
    --model bert \
    --dataset agnews \
    --epsilon 8.0 \
    --epochs 20 \
    --batch_size 64 \
    --lr 5e-4 \
    2>&1 | tee results/full_experiment_eps8_log.txt

# Detach: Ctrl+B, then D
```

### Step 9: Run Epsilon=1 Experiments

```bash
# In same or new tmux session
python scripts/run_all_placements.py \
    --model bert \
    --dataset agnews \
    --epsilon 1.0 \
    --epochs 20 \
    --batch_size 64 \
    --lr 5e-4 \
    2>&1 | tee results/full_experiment_eps1_log.txt
```

### Step 10: Generate Privacy-Utility Curves

```bash
# Sweep epsilon values for each placement
for placement in no_dp adapter_only head_adapter last_layer full_dp partial_backbone; do
    echo "Running privacy curve for $placement..."
    python scripts/run_privacy_curve.py \
        --model bert \
        --dataset agnews \
        --placement $placement \
        --epochs 20 \
        --batch_size 64 \
        --lr 5e-4 \
        2>&1 | tee results/privacy_curve_${placement}.log
done
```

**Time estimate:** ~5-7 days continuous GPU time for complete text experiments

---

## Part 6: Membership Inference Attacks

### Step 11: Run MIA Evaluation

```bash
# After training completes, run MIA on all checkpoints
for checkpoint in checkpoints/*.pt; do
    echo "Running MIA on $checkpoint..."
    python scripts/run_mia.py \
        --checkpoint $checkpoint \
        --model bert \
        --dataset agnews
done
```

---

## Part 7: Vision Experiments (Optional)

### Step 12: Run ViT + CIFAR-10 Experiments

```bash
# Similar to text experiments but with vision model
python scripts/run_all_placements.py \
    --model vit \
    --dataset cifar10 \
    --epsilon 8.0 \
    --epochs 30 \
    --batch_size 32 \
    --lr 5e-4 \
    2>&1 | tee results/vision_experiment_log.txt
```

---

## Part 8: Analysis and Visualization

### Step 13: Download Results and Generate Figures

```bash
# Option 1: Use Jupyter on RunPod
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Then access via RunPod's HTTP service or port forwarding

# Option 2: Download results to local machine
# On your local machine:
scp -r runpod:/workspace/dp-peft-research/results ./local_results
scp -r runpod:/workspace/dp-peft-research/checkpoints ./local_checkpoints
```

### Step 14: Generate Paper Figures

```bash
# Open and run the analysis notebook
jupyter notebook notebooks/results_analysis.ipynb

# Or run as script (if converted)
python notebooks/results_analysis.py
```

---

## Monitoring and Management

### Monitor GPU Usage

```bash
# In a separate terminal/tmux pane
watch -n 5 nvidia-smi
```

### Monitor Experiment Progress

```bash
# Tail the log file
tail -f results/full_experiment_eps8_log.txt

# Or check specific metrics
grep "Test Acc" results/full_experiment_eps8_log.txt
```

### Check Disk Space

```bash
df -h
du -sh results/ checkpoints/
```

### Manage Tmux Sessions

```bash
# List sessions
tmux ls

# Attach to session
tmux attach -s dp_full_experiments

# Kill session
tmux kill-session -s dp_full_experiments

# Create new pane (inside tmux)
Ctrl+B, then %  # vertical split
Ctrl+B, then "  # horizontal split
```

---

## Cost Estimation

### GPU Costs (RunPod)

| GPU | VRAM | Cost/Hour | Full Experiments | Total Cost |
|-----|------|-----------|------------------|------------|
| **L40S** | 48GB | $0.79 | ~7 days (168 hrs) | ~$133 |
| **A100** | 40GB | $1.14 | ~7 days (168 hrs) | ~$191 |
| **RTX 4090** | 24GB | $0.44 | ~7 days (168 hrs) | ~$74 |
| **A6000** | 48GB | $0.79 | ~7 days (168 hrs) | ~$133 |

**Recommendations:**
- **Best value:** RTX 4090 (24GB is sufficient for batch_size=32-64)
- **Best performance:** L40S or A100 (can use larger batches)
- **Budget option:** RTX 3090 (~$0.30/hr, 24GB)

### Cost Optimization Tips

1. **Use Spot Instances:** 50-70% cheaper but can be interrupted
2. **Run overnight:** Some providers have off-peak pricing
3. **Start small:** Validate with 10K samples first (costs ~$5)
4. **Use early stopping:** Implemented in code, saves time
5. **Pause between experiments:** Stop pod when not actively training

---

## Troubleshooting

### Issue: Out of Memory (OOM)

```bash
# Reduce batch size
python scripts/run_experiment.py --batch_size 16 ...

# Or use gradient accumulation (already implemented)
# The code automatically handles this
```

### Issue: Slow Downloads

```bash
# Set HuggingFace cache to fast storage
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# Pre-download datasets
python -c "from datasets import load_dataset; load_dataset('ag_news')"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"
```

### Issue: Connection Lost

```bash
# If using tmux, your experiments continue running
# Just reconnect and reattach:
ssh runpod
tmux attach -s dp_full_experiments
```

### Issue: Disk Full

```bash
# Clean up old results
rm -rf results/*.log
rm -rf checkpoints/old_*.pt

# Or request larger volume storage
```

---

## Quick Reference Commands

### Essential Commands

```bash
# Activate environment
source venv/bin/activate

# Check GPU
nvidia-smi

# Start experiment in background
tmux new -s experiment_name
# ... run command ...
# Ctrl+B, D to detach

# Reattach to experiment
tmux attach -s experiment_name

# Monitor progress
tail -f results/experiment_log.txt

# Check results
cat results/bert_agnews_adapter_only_eps8.0.json | python -m json.tool
```

### File Locations

- **Code:** `/workspace/dp-peft-research/`
- **Results:** `/workspace/dp-peft-research/results/`
- **Checkpoints:** `/workspace/dp-peft-research/checkpoints/`
- **Logs:** `/workspace/dp-peft-research/results/*.log`
- **Configs:** `/workspace/dp-peft-research/configs/`

---

## Expected Timeline

### Minimal Viable Experiments (Class Project)
- **Setup:** 30 minutes
- **Validation:** 3-4 hours
- **Core experiments:** 2-3 days
- **Analysis:** 4-6 hours
- **Total:** ~4 days, ~$75-100

### Full Paper Quality
- **Setup:** 30 minutes
- **Hyperparameter tuning:** 1 day
- **Text experiments:** 5-7 days
- **Vision experiments:** 3-4 days
- **MIA evaluation:** 1 day
- **Analysis:** 2-3 days
- **Total:** ~2-3 weeks, ~$200-300

---

## Data Backup

### Save Results Regularly

```bash
# Compress results
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/ checkpoints/

# Download to local machine
scp runpod:/workspace/dp-peft-research/results_backup_*.tar.gz ./

# Or use RunPod's volume storage for persistence
```

---

## Next Steps After Experiments Complete

1. **Download all results** to local machine
2. **Run analysis notebook** to generate figures
3. **Verify reproducibility** by checking saved configs
4. **Write paper** using results
5. **Stop RunPod instance** to avoid charges

---

## Support

- **RunPod Docs:** https://docs.runpod.io/
- **Opacus Issues:** https://github.com/pytorch/opacus/issues
- **Project Issues:** Create issue on GitHub repo

---

## Checklist

Before starting full experiments:

- [ ] RunPod account created with credits
- [ ] GPU pod deployed (L40S or A100 recommended)
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Quick test passed (Step 5)
- [ ] Hyperparameters validated (Step 6)
- [ ] All placements working (Step 7)
- [ ] Tmux session created
- [ ] Monitoring setup (nvidia-smi, logs)

Ready to start full experiments! 🚀
