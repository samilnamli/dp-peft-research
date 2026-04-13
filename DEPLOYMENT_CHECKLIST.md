# DP-PEFT Deployment Checklist

Complete checklist for deploying to GitHub and running on RunPod GPU servers.

---

## 📦 Part 1: Push to GitHub (10 minutes)

### Step 1: Create GitHub Repository

- [ ] Go to https://github.com/new
- [ ] Repository name: `dp-peft-research` (or your choice)
- [ ] Description: "Differential Privacy Placement in Parameter-Efficient Fine-Tuning"
- [ ] Visibility: Public or Private
- [ ] **DO NOT** initialize with README (we have one)
- [ ] Click "Create repository"
- [ ] Copy the repository URL

### Step 2: Push Code to GitHub

```bash
cd /home/asami/privacy/dp_peft

# Repository is already initialized and committed
# Just add remote and push:

# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/dp-peft-research.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

**If prompted for credentials:**
- Username: Your GitHub username
- Password: Use Personal Access Token from https://github.com/settings/tokens

### Step 3: Verify Upload

- [ ] Go to your GitHub repository in browser
- [ ] Verify all files are present (53 files)
- [ ] Check that `venv/` and `results/*.json` are NOT uploaded (gitignored)
- [ ] Verify README displays correctly

---

## 🚀 Part 2: Deploy on RunPod (30 minutes)

### Step 4: Set Up RunPod Account

- [ ] Go to https://runpod.io
- [ ] Create account / Sign in
- [ ] Add credits ($50-100 recommended for full experiments)
- [ ] Verify payment method

### Step 5: Deploy GPU Pod

**Recommended Configuration:**

- [ ] Click "Deploy" → "GPU Pods"
- [ ] **GPU:** L40S (48GB VRAM, ~$0.79/hr) or A100 (40GB, ~$1.14/hr)
- [ ] **Template:** PyTorch 2.0+ or Ubuntu 22.04 with CUDA
- [ ] **Container Disk:** 50GB minimum
- [ ] **Volume Storage:** 100GB (optional, for persistence)
- [ ] **Deployment Type:** On-Demand (or Spot for 50% savings)
- [ ] Click "Deploy"
- [ ] Wait for pod to start (~1-2 minutes)

### Step 6: Connect to Pod

- [ ] Click "Connect" on your pod
- [ ] Choose "Start Web Terminal" or copy SSH command
- [ ] Verify you're connected: `nvidia-smi` should show GPU

---

## 🔧 Part 3: Setup Environment (15 minutes)

### Step 7: Clone Repository

```bash
# Navigate to workspace
cd /workspace

# Clone your repository (replace with your URL)
git clone https://github.com/YOUR_USERNAME/dp-peft-research.git
cd dp-peft-research

# Verify files
ls -la
```

**Checklist:**
- [ ] Repository cloned successfully
- [ ] All directories present (dp_peft/, scripts/, configs/, etc.)
- [ ] README.md visible

### Step 8: Install Dependencies

```bash
# Update system (optional but recommended)
apt-get update && apt-get install -y git tmux htop

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies (takes 5-10 minutes)
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

**Checklist:**
- [ ] Virtual environment created
- [ ] All packages installed without errors
- [ ] No version conflicts

### Step 9: Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Check all packages
python -c "import transformers, peft, opacus, datasets; print('All packages OK!')"

# Check GPU
nvidia-smi
```

**Expected Output:**
- [ ] PyTorch version 2.x.x
- [ ] CUDA: True
- [ ] GPU visible in nvidia-smi
- [ ] All packages import successfully

---

## ✅ Part 4: Validation (10 minutes)

### Step 10: Run Quick Test

```bash
# Run validation script
bash scripts/runpod_quick_test.sh
```

**Expected Results:**
- [ ] Test 1: GPU Check - PASSED
- [ ] Test 2: Package Imports - PASSED
- [ ] Test 3: No-DP Baseline - PASSED (accuracy >50%)
- [ ] Test 4: DP Test - PASSED (runs without errors)

**If any test fails, STOP and debug before proceeding.**

---

## 🎯 Part 5: Start Experiments (5-7 days)

### Step 11: Create Tmux Session

```bash
# Create persistent session
tmux new -s dp_experiments

# Verify you're in tmux (should see green bar at bottom)
```

**Checklist:**
- [ ] Tmux session created
- [ ] Green status bar visible at bottom
- [ ] Can detach (Ctrl+B, D) and reattach (tmux attach -s dp_experiments)

### Step 12: Launch Full Experiments

```bash
# Inside tmux session
cd /workspace/dp-peft-research
source venv/bin/activate

# Start full experiment suite
bash scripts/runpod_full_experiments.sh

# Detach from tmux: Ctrl+B, then D
```

**Checklist:**
- [ ] Script started successfully
- [ ] First experiment (no_dp, ε=8.0) running
- [ ] Progress bars visible
- [ ] No immediate errors
- [ ] Detached from tmux (experiments continue in background)

---

## 📊 Part 6: Monitoring (Ongoing)

### Step 13: Set Up Monitoring

**In a separate terminal/SSH session:**

```bash
# Monitor GPU usage
watch -n 5 nvidia-smi

# Or in a new tmux pane:
tmux attach -s dp_experiments
# Ctrl+B, % (split vertically)
# In new pane: watch -n 5 nvidia-smi
```

**Checklist:**
- [ ] GPU utilization >80% (good)
- [ ] Memory usage visible
- [ ] Temperature reasonable (<85°C)

### Step 14: Check Progress Regularly

```bash
# View latest logs
tail -f /workspace/dp-peft-research/results/*_full_*.log

# Check accuracy progress
grep "Test Acc" /workspace/dp-peft-research/results/*_full_*.log

# Reattach to main experiment
tmux attach -s dp_experiments
```

**Daily Checklist:**
- [ ] Experiments still running (check tmux)
- [ ] No errors in logs
- [ ] Accuracy improving over epochs
- [ ] Disk space sufficient (`df -h`)
- [ ] RunPod credits sufficient

---

## 💾 Part 7: Results Collection (After Completion)

### Step 15: Download Results

**Option A: Direct SCP (from local machine)**

```bash
# Replace with your RunPod SSH details
scp -r runpod:/workspace/dp-peft-research/results ./local_results
scp -r runpod:/workspace/dp-peft-research/checkpoints ./local_checkpoints
```

**Option B: Compress First**

```bash
# On RunPod
cd /workspace/dp-peft-research
tar -czf results_$(date +%Y%m%d).tar.gz results/ checkpoints/

# Download via RunPod web interface or SCP
```

**Checklist:**
- [ ] All result JSON files downloaded
- [ ] All checkpoint .pt files downloaded
- [ ] Log files downloaded
- [ ] Summary file downloaded

---

## 📈 Part 8: Analysis (Local Machine)

### Step 16: Generate Figures

```bash
# On local machine with downloaded results
cd local_results

# Open Jupyter notebook
jupyter notebook ../notebooks/results_analysis.ipynb

# Run all cells to generate figures
```

**Checklist:**
- [ ] Privacy-utility curves generated
- [ ] Training convergence plots created
- [ ] MIA comparison charts made
- [ ] Summary table created
- [ ] Figures saved as PDF and PNG

---

## 🧹 Part 9: Cleanup

### Step 17: Stop RunPod Instance

**IMPORTANT:** Stop your pod to avoid ongoing charges!

- [ ] Verify all results downloaded
- [ ] Verify all checkpoints downloaded
- [ ] Go to RunPod dashboard
- [ ] Click "Stop" on your pod
- [ ] Confirm pod is stopped
- [ ] Check billing to ensure charges stopped

---

## 📋 Final Verification

### Experiment Completeness

- [ ] All 6 placements tested at ε=8.0
- [ ] All 6 placements tested at ε=1.0
- [ ] Privacy-utility curves generated
- [ ] MIA evaluation completed
- [ ] All results saved and downloaded

### Expected Results Files

```
results/
├── no_dp_eps8.0_full_*.log
├── adapter_only_eps8.0_full_*.log
├── head_adapter_eps8.0_full_*.log
├── last_layer_eps8.0_full_*.log
├── full_dp_eps8.0_full_*.log
├── partial_backbone_eps8.0_full_*.log
├── [same for eps1.0]
├── privacy_curve_adapter_only_*.log
├── privacy_curve_full_dp_*.log
├── mia_*.log
└── experiment_summary_*.txt

checkpoints/
├── bert_agnews_no_dp_eps8.0.pt
├── bert_agnews_adapter_only_eps8.0.pt
├── [etc for all placements]
```

### Quality Checks

- [ ] No-DP baseline: ~90% accuracy
- [ ] Adapter-Only DP (ε=8): >70% accuracy
- [ ] Adapter-Only DP (ε=1): >60% accuracy
- [ ] Full-Model DP shows lower accuracy than Adapter-Only (validates hypothesis)
- [ ] Privacy accounting correct (ε values match targets)
- [ ] MIA results reasonable (lower AUC for DP models)

---

## 💰 Cost Tracking

### Estimated Costs

| Phase | Duration | L40S Cost | A100 Cost |
|-------|----------|-----------|-----------|
| Setup & Validation | 1 hour | $0.79 | $1.14 |
| Phase 1 (ε=8) | 48 hours | $38 | $55 |
| Phase 2 (ε=1) | 48 hours | $38 | $55 |
| Privacy Curves | 48 hours | $38 | $55 |
| MIA | 12 hours | $9.50 | $14 |
| **Total** | **~157 hours** | **~$124** | **~$179** |

**Actual Cost Checklist:**
- [ ] Initial credits: $______
- [ ] Final credits: $______
- [ ] Total spent: $______
- [ ] Within budget: Yes / No

---

## 🎓 Next Steps

After successful deployment and experiments:

- [ ] Analyze results
- [ ] Generate paper figures
- [ ] Write methodology section
- [ ] Write results section
- [ ] Write discussion
- [ ] Prepare presentation
- [ ] Submit paper/project

---

## 🆘 Troubleshooting Reference

### Common Issues

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce batch size in scripts/runpod_full_experiments.sh |
| Slow training | Check GPU utilization with nvidia-smi |
| Connection lost | Experiments continue in tmux, just reconnect |
| Disk full | Clean up old logs: `rm results/*.log` |
| Poor DP accuracy | Verify hyperparameters (LR=5e-4, not 2e-5) |

---

## ✨ Success Criteria

You've successfully completed deployment when:

- [x] Code pushed to GitHub
- [x] RunPod GPU deployed
- [x] Environment setup complete
- [x] Quick tests passed
- [x] Full experiments running
- [x] Monitoring in place
- [x] Results downloading plan ready

**You're ready to go! 🚀**

---

## 📞 Support Resources

- **RunPod Docs:** https://docs.runpod.io/
- **Opacus GitHub:** https://github.com/pytorch/opacus
- **PEFT Docs:** https://huggingface.co/docs/peft
- **Project Issues:** GitHub repository issues tab

---

**Estimated Total Time:** 
- Setup: 1 hour
- Experiments: 5-7 days GPU time
- Analysis: 1-2 days
- **Total: ~1-2 weeks calendar time**

**Good luck with your experiments! 🎉**
