# 🚀 START HERE - DP-PEFT Project

**Everything you need to deploy and run experiments on RunPod GPU.**

---

## ⚡ Quick Start (30 minutes to running)

### 1. Push to GitHub (5 min)

```bash
cd /home/asami/privacy/dp_peft

# Create repo at: https://github.com/new
# Name: dp-peft-research

# Push code (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/dp-peft-research.git
git push -u origin main
```

### 2. Deploy RunPod (10 min)

1. Go to https://runpod.io
2. Deploy → GPU Pods → Select **L40S** (48GB, $0.79/hr)
3. Template: PyTorch 2.1
4. Deploy On-Demand
5. Connect → Web Terminal

### 3. Setup Environment (10 min)

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/dp-peft-research.git
cd dp-peft-research
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

### 4. Run Quick Test (5 min)

```bash
bash scripts/runpod_quick_test.sh
```

### 5. Start Full Experiments (5-7 days)

```bash
tmux new -s experiments
bash scripts/runpod_full_experiments.sh
# Ctrl+B, D to detach
```

**Done! Experiments running in background.**

---

## 📚 Documentation

Choose your guide based on detail level:

### 🏃 Quick & Simple
- **`QUICKSTART.md`** - TL;DR version (1 page)
- **`START_HERE.md`** - This file

### 📖 Detailed & Complete
- **`COMPLETE_EXECUTION_GUIDE.md`** - Full walkthrough (GitHub → Results)
- **`STEP_BY_STEP_EXECUTION.md`** - Detailed step-by-step
- **`RUNPOD_SETUP.md`** - Comprehensive RunPod guide

### ✅ Checklists & References
- **`DEPLOYMENT_CHECKLIST.md`** - Tick-box checklist
- **`GITHUB_SETUP.md`** - Git commands reference
- **`README_GITHUB_RUNPOD.md`** - Overview & file structure

### 📊 Project Status
- **`PROJECT_STATUS_SUMMARY.md`** - What's done, what's needed
- **`GPU_READINESS_ANALYSIS.md`** - Technical analysis
- **`IMMEDIATE_ACTION_PLAN.md`** - Next steps

---

## 🎯 What You'll Get

### After 7 days on RunPod:
- ✅ 12 trained models (6 placements × 2 ε values)
- ✅ Privacy-utility curves
- ✅ MIA evaluation results
- ✅ Publication-ready figures

### Expected Results:
- No-DP: ~90% accuracy
- Adapter-Only DP (ε=8): ~75-85%
- Full-Model DP (ε=8): ~60-70%

### Cost:
- L40S: ~$133 for full experiments
- RTX 4090: ~$74 (budget option)

---

## 🆘 Common Issues

### "Git push fails"
→ Use Personal Access Token from https://github.com/settings/tokens

### "Out of memory"
→ Edit `scripts/runpod_full_experiments.sh`, change BATCH_SIZE=64 to 32

### "Poor DP accuracy"
→ Verify LR=5e-4 in script (not 2e-5)

### "Can't reconnect"
→ `tmux attach -s experiments`

---

## 📞 Need Help?

**Read the appropriate guide:**
- Confused? → `COMPLETE_EXECUTION_GUIDE.md`
- Want checklist? → `DEPLOYMENT_CHECKLIST.md`
- GitHub issues? → `GITHUB_SETUP.md`
- RunPod problems? → `RUNPOD_SETUP.md`

**External resources:**
- RunPod: https://docs.runpod.io/
- Opacus: https://github.com/pytorch/opacus

---

## ✨ Your Next Action

**Right now, do this:**

```bash
# 1. Create GitHub repo at: https://github.com/new

# 2. Push code:
cd /home/asami/privacy/dp_peft
git remote add origin https://github.com/YOUR_USERNAME/dp-peft-research.git
git push -u origin main

# 3. Then follow: COMPLETE_EXECUTION_GUIDE.md
```

**That's it! You're ready to go! 🚀**

---

## 📋 Quick Reference

### Essential Commands

```bash
# Activate environment
source venv/bin/activate

# Check GPU
nvidia-smi

# Tmux
tmux new -s experiments      # Create
tmux attach -s experiments   # Reattach
Ctrl+B, D                    # Detach

# Monitor
tail -f results/*_full_*.log
grep "Test Acc" results/*_full_*.log

# Download results (from local machine)
scp root@RUNPOD_IP:/workspace/dp-peft-research/results_*.tar.gz ./
```

### File Locations on RunPod

```
/workspace/dp-peft-research/     # Project root
├── results/                      # Experiment results
├── checkpoints/                  # Model checkpoints
├── scripts/                      # Experiment scripts
└── configs/                      # Configuration files
```

---

**Everything is ready. Just push to GitHub and deploy! 🎉**
