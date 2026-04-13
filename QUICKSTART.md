# Quick Start Guide: DP-PEFT on RunPod

**Goal:** Get from zero to running experiments in 30 minutes.

---

## 🚀 Super Quick Start (TL;DR)

```bash
# On RunPod GPU server:
cd /workspace
git clone https://github.com/YOUR_USERNAME/dp-peft-research.git
cd dp-peft-research
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Quick test (5 minutes)
bash scripts/runpod_quick_test.sh

# Full experiments (5-7 days)
tmux new -s experiments
bash scripts/runpod_full_experiments.sh
# Ctrl+B, D to detach
```

---

## 📋 Detailed Steps

### 1. Deploy RunPod GPU (5 minutes)

1. Go to **RunPod.io** → Deploy → GPU Pods
2. Select **L40S** (48GB, ~$0.79/hr) or **A100** (40GB, ~$1.14/hr)
3. Template: **PyTorch 2.0+** or **Ubuntu 22.04**
4. Container Disk: **50GB minimum**
5. Click **Deploy On-Demand**
6. Wait for pod to start (~1-2 min)
7. Click **Connect** → Use Web Terminal or copy SSH command

### 2. Clone Repository (2 minutes)

```bash
# Navigate to workspace
cd /workspace

# Clone repository (replace with your GitHub URL)
git clone https://github.com/YOUR_USERNAME/dp-peft-research.git
cd dp-peft-research

# Verify files
ls -la
```

### 3. Install Dependencies (10 minutes)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 4. Run Quick Test (5 minutes)

```bash
# Run validation test
bash scripts/runpod_quick_test.sh

# Should see:
# ✓ GPU detected
# ✓ Packages imported
# ✓ No-DP test passed
# ✓ DP test passed
```

### 5. Start Full Experiments (5-7 days)

```bash
# Create tmux session (keeps running if disconnected)
tmux new -s experiments

# Run full experiment suite
bash scripts/runpod_full_experiments.sh

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -s experiments
```

---

## 📊 What Gets Run

### Full Experiment Suite Includes:

1. **Phase 1:** All 6 placements at ε=8.0
   - No DP, Adapter-Only, Head+Adapter, Last-Layer, Full-Model, Partial Backbone
   
2. **Phase 2:** All 6 placements at ε=1.0
   
3. **Phase 3:** Privacy-utility curves (sweep ε values)
   
4. **Phase 4:** Membership inference attacks

**Total time:** ~5-7 days continuous GPU time  
**Total cost:** ~$95-135 on L40S

---

## 🔍 Monitor Progress

### Check GPU Usage
```bash
watch -n 5 nvidia-smi
```

### View Logs
```bash
# Tail latest log
tail -f results/*_full_*.log

# Check accuracy progress
grep "Test Acc" results/*_full_*.log
```

### Reattach to Experiment
```bash
tmux attach -s experiments
```

---

## 💾 Download Results

### Option 1: SCP (from local machine)
```bash
scp -r runpod:/workspace/dp-peft-research/results ./local_results
scp -r runpod:/workspace/dp-peft-research/checkpoints ./local_checkpoints
```

### Option 2: Compress and Download
```bash
# On RunPod
cd /workspace/dp-peft-research
tar -czf results_$(date +%Y%m%d).tar.gz results/ checkpoints/

# Then download via RunPod web interface or SCP
```

---

## 🛠️ Troubleshooting

### Out of Memory?
```bash
# Edit scripts/runpod_full_experiments.sh
# Change: BATCH_SIZE=64 → BATCH_SIZE=32
```

### Slow Training?
```bash
# Check GPU utilization
nvidia-smi

# Should see ~90%+ GPU usage
# If low, check batch size or data loading
```

### Connection Lost?
```bash
# Your experiments keep running in tmux!
# Just reconnect and reattach:
ssh runpod
tmux attach -s experiments
```

---

## 📈 Expected Results

After full experiments complete, you should have:

- ✅ 12 trained models (6 placements × 2 ε values)
- ✅ Privacy-utility curves data
- ✅ MIA evaluation results
- ✅ Training logs and metrics
- ✅ Model checkpoints

**Accuracy targets:**
- No DP: ~90%
- Adapter-Only DP (ε=8): ~75-85%
- Adapter-Only DP (ε=1): ~65-75%
- Full-Model DP (ε=8): ~60-70%

---

## 🎯 Next Steps After Experiments

1. **Download results** to local machine
2. **Run analysis notebook:** `jupyter notebook notebooks/results_analysis.ipynb`
3. **Generate figures** for paper
4. **Write up results**
5. **Stop RunPod instance** to save costs

---

## 💰 Cost Estimates

| GPU | Cost/Hour | Full Experiments | Total |
|-----|-----------|------------------|-------|
| L40S (48GB) | $0.79 | 168 hours | ~$133 |
| A100 (40GB) | $1.14 | 168 hours | ~$191 |
| RTX 4090 (24GB) | $0.44 | 168 hours | ~$74 |

**Tip:** Use spot instances for 50-70% savings (but can be interrupted)

---

## 📞 Need Help?

- **RunPod Issues:** https://docs.runpod.io/
- **Opacus Issues:** https://github.com/pytorch/opacus
- **Project Issues:** Create issue on GitHub repo

---

## ✅ Pre-Flight Checklist

Before starting full experiments:

- [ ] RunPod GPU deployed (L40S or A100)
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Quick test passed
- [ ] Tmux session created
- [ ] Monitoring setup (nvidia-smi, logs)
- [ ] Understand costs (~$100-200 for full suite)

**Ready to go! 🚀**

---

## 🔗 Useful Links

- **Full Setup Guide:** See `RUNPOD_SETUP.md`
- **GitHub Setup:** See `GITHUB_SETUP.md`
- **Project Status:** See `PROJECT_STATUS_SUMMARY.md`
- **Immediate Actions:** See `IMMEDIATE_ACTION_PLAN.md`
