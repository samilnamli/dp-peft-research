# Step-by-Step Execution Guide for RunPod L40S

**Complete walkthrough from zero to running experiments in 30 minutes.**

---

## 🎯 Overview

This guide will walk you through:
1. Pushing code to GitHub (5 min)
2. Setting up RunPod L40S GPU (10 min)
3. Installing dependencies (10 min)
4. Starting experiments (5 min)

**Total setup time: ~30 minutes**  
**Total experiment time: ~5-7 days GPU time**  
**Estimated cost: ~$124 on L40S**

---

## 📤 STEP 1: Push to GitHub (5 minutes)

### 1.1 Create GitHub Repository

1. Open browser → https://github.com/new
2. Fill in:
   - **Repository name:** `dp-peft-research`
   - **Description:** "Differential Privacy Placement in PEFT"
   - **Visibility:** Public (or Private if preferred)
   - **DO NOT check:** Initialize with README
3. Click **"Create repository"**
4. **Copy the repository URL** shown on the next page
   - Example: `https://github.com/yourusername/dp-peft-research.git`

### 1.2 Push Your Code

Open terminal on your local machine:

```bash
cd /home/asami/privacy/dp_peft

# Add your GitHub repository as remote (replace with YOUR URL)
git remote add origin https://github.com/YOUR_USERNAME/dp-peft-research.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

**If asked for credentials:**
- Username: Your GitHub username
- Password: Generate token at https://github.com/settings/tokens
  - Click "Generate new token (classic)"
  - Select scope: `repo`
  - Copy the token and use it as password

### 1.3 Verify Upload

1. Go to your repository: `https://github.com/YOUR_USERNAME/dp-peft-research`
2. You should see:
   - ✅ README.md
   - ✅ 53 files total
   - ✅ All directories (dp_peft/, scripts/, configs/)
   - ❌ No venv/ folder (correctly ignored)

**✅ GitHub setup complete!**

---

## 🚀 STEP 2: Deploy RunPod GPU (10 minutes)

### 2.1 Create RunPod Account

1. Go to https://runpod.io
2. Click **"Sign Up"** or **"Login"**
3. Complete registration
4. Add billing method
5. **Add credits:** $100-150 recommended

### 2.2 Deploy L40S GPU Pod

1. Click **"Deploy"** in top menu
2. Click **"GPU Pods"**
3. Configure pod:
   - **GPU Type:** Filter for "L40S" (48GB VRAM)
   - **Template:** Select "PyTorch 2.1" or "RunPod PyTorch"
   - **Container Disk:** 50 GB
   - **Volume Disk:** 100 GB (optional but recommended)
   - **Deployment:** On-Demand (or Spot for 50% savings)
4. Click **"Deploy On-Demand"**
5. Wait ~1-2 minutes for pod to start
6. Status should change to "Running"

### 2.3 Connect to Pod

**Option A: Web Terminal (Easiest)**
1. Click **"Connect"** button on your pod
2. Click **"Start Web Terminal"**
3. Terminal opens in browser

**Option B: SSH**
1. Click **"Connect"** → Copy SSH command
2. Paste in your local terminal
3. Example: `ssh root@123.456.789.012 -p 12345 -i ~/.ssh/id_rsa`

### 2.4 Verify GPU

In the terminal (web or SSH):

```bash
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA L40S         Off  | 00000000:00:05.0 Off |                    0 |
| N/A   35C    P0    72W / 350W |      0MiB / 46068MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

**✅ GPU verified!**

---

## 🔧 STEP 3: Setup Environment (10 minutes)

### 3.1 Clone Repository

```bash
# Navigate to workspace
cd /workspace

# Clone your repository (replace with YOUR GitHub URL)
git clone https://github.com/YOUR_USERNAME/dp-peft-research.git

# Enter directory
cd dp-peft-research

# Verify files
ls -la
```

**You should see:**
- README.md, requirements.txt, setup.py
- Directories: dp_peft/, scripts/, configs/, notebooks/

### 3.2 Install System Dependencies

```bash
# Update package manager
apt-get update

# Install useful tools
apt-get install -y git tmux htop vim

# Verify Python version
python3 --version  # Should be 3.10+
```

### 3.3 Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate

# You should see (venv) in your prompt
```

### 3.4 Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies (takes 5-10 minutes)
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

**Wait for installation to complete...**

### 3.5 Verify Installation

```bash
# Test PyTorch and CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA available: True
GPU: NVIDIA L40S
```

```bash
# Test all packages
python -c "import transformers, peft, opacus, datasets; print('✓ All packages imported successfully!')"
```

**Expected output:**
```
✓ All packages imported successfully!
```

**✅ Environment setup complete!**

---

## ✅ STEP 4: Run Quick Test (5 minutes)

### 4.1 Run Validation Script

```bash
# Make sure you're in the project directory
cd /workspace/dp-peft-research
source venv/bin/activate

# Run quick test
bash scripts/runpod_quick_test.sh
```

**This will:**
1. Check GPU (should pass)
2. Import packages (should pass)
3. Run 2-epoch No-DP test (~2 min)
4. Run 2-epoch DP test (~2 min)

**Expected output at end:**
```
==========================================
ALL TESTS PASSED! ✓
==========================================

Your RunPod environment is ready for full experiments!
```

**If any test fails, STOP and debug before proceeding.**

**✅ Validation complete!**

---

## 🎯 STEP 5: Start Full Experiments (5 minutes setup, then 5-7 days running)

### 5.1 Create Tmux Session

Tmux keeps your experiments running even if you disconnect.

```bash
# Create new tmux session
tmux new -s experiments

# You should see a green bar at the bottom
```

**Tmux basics:**
- **Detach:** `Ctrl+B`, then press `D`
- **Reattach:** `tmux attach -s experiments`
- **Kill session:** `tmux kill-session -s experiments`

### 5.2 Launch Full Experiment Suite

Inside the tmux session:

```bash
# Navigate to project
cd /workspace/dp-peft-research

# Activate environment
source venv/bin/activate

# Start full experiments
bash scripts/runpod_full_experiments.sh
```

**This script will run:**
1. **Phase 1:** All 6 placements at ε=8.0 (~48 hours)
2. **Phase 2:** All 6 placements at ε=1.0 (~48 hours)
3. **Phase 3:** Privacy-utility curves (~48 hours)
4. **Phase 4:** Membership inference attacks (~12 hours)

**Total time: ~5-7 days continuous GPU time**

### 5.3 Detach from Tmux

Once experiments start running:

1. Press `Ctrl+B`
2. Then press `D`
3. You'll see: `[detached (from session experiments)]`

**Your experiments continue running in the background!**

You can now:
- Close your terminal
- Shut down your computer
- Disconnect from RunPod
- **Experiments keep running!**

**✅ Experiments launched!**

---

## 📊 STEP 6: Monitor Progress (Ongoing)

### 6.1 Reconnect to RunPod

Anytime you want to check progress:

1. Go to RunPod dashboard
2. Click **"Connect"** on your pod
3. Open Web Terminal or SSH

### 6.2 Reattach to Tmux

```bash
tmux attach -s experiments
```

You'll see the live experiment output.

### 6.3 Check GPU Usage

In a new terminal or tmux pane:

```bash
# Watch GPU usage (updates every 5 seconds)
watch -n 5 nvidia-smi
```

**Healthy signs:**
- GPU Utilization: 80-100%
- Memory Usage: 20-40GB (out of 48GB)
- Temperature: <85°C

### 6.4 View Logs

```bash
# Tail the latest log file
tail -f /workspace/dp-peft-research/results/*_full_*.log

# Or check accuracy progress
grep "Test Acc" /workspace/dp-peft-research/results/*_full_*.log
```

### 6.5 Check Disk Space

```bash
df -h /workspace
```

Make sure you have >10GB free.

---

## 💾 STEP 7: Download Results (After ~7 days)

### 7.1 Check Completion

Reattach to tmux:

```bash
tmux attach -s experiments
```

Look for:
```
==========================================
ALL EXPERIMENTS COMPLETE!
==========================================
```

### 7.2 Compress Results

```bash
cd /workspace/dp-peft-research

# Create compressed archive
tar -czf results_$(date +%Y%m%d).tar.gz results/ checkpoints/

# Check size
ls -lh results_*.tar.gz
```

### 7.3 Download to Local Machine

**Option A: SCP (from your local machine)**

```bash
# Replace with your RunPod SSH details
scp root@YOUR_RUNPOD_IP:/workspace/dp-peft-research/results_*.tar.gz ./
```

**Option B: RunPod Web Interface**

1. In RunPod web terminal, navigate to file
2. Right-click → Download

### 7.4 Extract Locally

```bash
# On your local machine
tar -xzf results_*.tar.gz
```

**✅ Results downloaded!**

---

## 🧹 STEP 8: Stop RunPod Instance

**CRITICAL:** Stop your pod to avoid ongoing charges!

1. Go to RunPod dashboard
2. Find your pod
3. Click **"Stop"** (not "Terminate")
4. Confirm stop
5. Verify status changes to "Stopped"

**Stopping vs Terminating:**
- **Stop:** Keeps your data, can restart later
- **Terminate:** Deletes everything, cannot recover

**If you might need to restart:**
- Choose "Stop" (small storage fee)
- Can restart and continue later

**If completely done:**
- Choose "Terminate" (no further charges)
- Make sure you downloaded all results first!

**✅ Pod stopped - no more charges!**

---

## 📈 STEP 9: Analyze Results (Local Machine)

### 9.1 Open Analysis Notebook

```bash
# Navigate to your local copy
cd /path/to/local/dp-peft-research

# Activate local venv (or create one)
source venv/bin/activate
pip install jupyter matplotlib seaborn pandas

# Start Jupyter
jupyter notebook notebooks/results_analysis.ipynb
```

### 9.2 Generate Figures

Run all cells in the notebook to generate:
- Privacy-utility curves
- Training convergence plots
- MIA comparison charts
- Summary tables

Figures will be saved in `results/figures/`

**✅ Analysis complete!**

---

## 🎓 Expected Results

After successful completion, you should have:

### Accuracy Results

| Placement | ε=8.0 | ε=1.0 |
|-----------|-------|-------|
| No DP | ~90% | ~90% |
| Adapter-Only DP | ~75-85% | ~65-75% |
| Head+Adapter DP | ~70-80% | ~60-70% |
| Last-Layer DP | ~65-75% | ~55-65% |
| Full-Model DP | ~60-70% | ~50-60% |
| Partial Backbone DP | ~65-75% | ~55-65% |

### Key Findings

1. **Adapter-Only DP achieves best privacy-utility tradeoff**
2. **Applying DP to fewer parameters preserves more utility**
3. **All placements provide similar empirical privacy (MIA resistance)**
4. **Training stability varies by placement**

---

## 💰 Cost Summary

### L40S Costs

- **Hourly rate:** $0.79/hour
- **Total time:** ~168 hours (7 days)
- **Total cost:** ~$133

### Cost Breakdown

| Phase | Hours | Cost |
|-------|-------|------|
| Setup & Testing | 2 | $1.58 |
| Phase 1 (ε=8) | 48 | $38 |
| Phase 2 (ε=1) | 48 | $38 |
| Privacy Curves | 48 | $38 |
| MIA | 12 | $9.50 |
| Buffer | 10 | $7.90 |
| **Total** | **168** | **~$133** |

---

## 🆘 Troubleshooting

### Issue: Git push fails

**Solution:**
```bash
# Use Personal Access Token
# Generate at: https://github.com/settings/tokens
# Use token as password when prompted
```

### Issue: Out of Memory on GPU

**Solution:**
```bash
# Edit scripts/runpod_full_experiments.sh
# Change line: BATCH_SIZE=64
# To: BATCH_SIZE=32
```

### Issue: Experiments stopped unexpectedly

**Solution:**
```bash
# Check if tmux session still exists
tmux ls

# If exists, reattach
tmux attach -s experiments

# If not, check logs for errors
tail -100 /workspace/dp-peft-research/results/*_full_*.log
```

### Issue: Poor DP accuracy (<50%)

**Solution:**
```bash
# Verify hyperparameters in script
# Should be: LR=5e-4 (not 2e-5)
grep "LR=" scripts/runpod_full_experiments.sh
```

### Issue: Disk full

**Solution:**
```bash
# Clean up old logs
rm /workspace/dp-peft-research/results/*.log

# Or delete old checkpoints
rm /workspace/dp-peft-research/checkpoints/old_*.pt
```

---

## ✅ Success Checklist

- [x] Code pushed to GitHub
- [x] RunPod L40S deployed
- [x] Environment installed
- [x] Quick test passed
- [x] Full experiments running in tmux
- [x] Monitoring setup
- [x] Results downloaded
- [x] Pod stopped
- [x] Analysis complete

**Congratulations! You've successfully deployed and run the DP-PEFT experiments! 🎉**

---

## 📞 Need Help?

- **RunPod Support:** https://docs.runpod.io/
- **GitHub Issues:** Create issue in your repository
- **Opacus Docs:** https://opacus.ai/
- **PEFT Docs:** https://huggingface.co/docs/peft

---

## 🔗 Quick Reference

### Essential Commands

```bash
# Activate environment
source venv/bin/activate

# Check GPU
nvidia-smi

# Tmux commands
tmux new -s experiments      # Create session
tmux attach -s experiments   # Reattach
Ctrl+B, D                    # Detach
tmux kill-session -s experiments  # Kill

# Monitor logs
tail -f results/*_full_*.log

# Check progress
grep "Test Acc" results/*_full_*.log
```

### Important Paths

- **Project:** `/workspace/dp-peft-research/`
- **Results:** `/workspace/dp-peft-research/results/`
- **Checkpoints:** `/workspace/dp-peft-research/checkpoints/`
- **Scripts:** `/workspace/dp-peft-research/scripts/`

---

**You're all set! Good luck with your experiments! 🚀**
