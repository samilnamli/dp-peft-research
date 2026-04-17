# 🎯 Ready to Run: Resilient Text Experiments

**Your project is now configured for resilient, resumable text experiments on RunPod.**

---

## ✅ What's Ready

Your repository now has:

1. **✅ Resilient experiment script** - Handles interruptions gracefully
2. **✅ Progress tracking** - Never lose your work
3. **✅ Intermediate saving** - Results saved after each placement
4. **✅ Immediate MIA** - Privacy evaluation right after training
5. **✅ Text-only focus** - BERT + AG News experiments
6. **✅ Complete documentation** - Step-by-step guides

---

## 🚀 Quick Start (30 minutes to running)

### 1. Push to GitHub (5 min)

```bash
cd /home/asami/privacy/dp_peft

# If not already added remote:
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

### 5. Start Resilient Experiments

```bash
tmux new -s experiments
bash scripts/runpod_text_experiments_resilient.sh
# Ctrl+B, D to detach
```

**Done! Your experiments are running with automatic progress tracking.**

---

## 🎯 What Will Run

### Phase 1: ε = 8.0 (6 placements)
Each experiment:
1. Trains model
2. Saves results immediately
3. Generates individual report
4. Runs MIA evaluation
5. Updates progress tracker

**Placements:**
- No DP (baseline)
- Adapter-Only DP
- Head+Adapter DP
- Last-Layer DP
- Full-Model DP
- Partial Backbone DP

### Phase 2: ε = 1.0 (6 placements)
Same as Phase 1 with stricter privacy

**Total: 12 experiments + 12 MIA evaluations**

---

## 🔄 Key Feature: Resume Capability

If your GPU gets interrupted:

```bash
# Just run the script again
bash scripts/runpod_text_experiments_resilient.sh
```

**The script will:**
- ✅ Check `results/progress.txt`
- ✅ Skip completed experiments
- ✅ Resume from where it stopped
- ✅ Continue with pending experiments

**Example:**
```
⏭️  Skipping no_dp | ε=8.0 (already completed)
⏭️  Skipping adapter_only | ε=8.0 (already completed)
Running: head_adapter | ε=8.0
[continues...]
```

---

## 📊 Progress Tracking

### Check Status Anytime

```bash
# View progress file
cat results/progress.txt

# Output:
# no_dp,8.0,completed,20260417_193045
# adapter_only,8.0,completed,20260417_201523
# head_adapter,8.0,running,20260417_205012
# ...
```

### Count Completed

```bash
grep -c "completed" results/progress.txt
# Shows: 3 / 12
```

### View Latest Report

```bash
ls -t results/report_*.txt | head -1 | xargs cat
```

---

## 📁 Files Created

### After Each Experiment

```
results/
├── progress.txt                                    # Progress tracker
├── bert_agnews_no_dp_eps8.0.json                  # Results
├── report_no_dp_eps8.0_20260417_193045.txt       # Individual report
├── mia_no_dp_eps8.0_resilient_*.log              # MIA evaluation
└── no_dp_eps8.0_resilient_*.log                  # Training log

checkpoints/
└── bert_agnews_no_dp_eps8.0.pt                    # Model checkpoint
```

### After All Experiments

```
results/
├── progress.txt                    # All 12 marked as completed
├── final_summary_resilient_*.txt   # Final summary
├── 12 × result JSON files
├── 12 × individual reports
├── 12 × MIA logs
└── 12 × training logs

checkpoints/
└── 12 × model checkpoints
```

---

## ⏱️ Time & Cost

### L40S ($0.79/hr) - Recommended
- **Time:** ~140 hours (~5-6 days)
- **Cost:** ~$111

### A100 ($1.14/hr)
- **Time:** ~120 hours (~5 days)
- **Cost:** ~$137

### RTX 4090 ($0.44/hr) - Budget
- **Time:** ~150 hours (~6 days)
- **Cost:** ~$66

---

## 📖 Documentation

### Quick References
- **`UPDATED_INSTRUCTIONS.md`** - ⭐ **START HERE** for resilient mode
- **`RESILIENT_EXPERIMENTS_GUIDE.md`** - Complete resilient script guide
- **`START_HERE.md`** - Quick navigation
- **`QUICKSTART.md`** - TL;DR version

### Detailed Guides
- **`COMPLETE_EXECUTION_GUIDE.md`** - Full walkthrough (GitHub → Results)
- **`STEP_BY_STEP_EXECUTION.md`** - Detailed step-by-step
- **`RUNPOD_SETUP.md`** - Comprehensive RunPod guide
- **`DEPLOYMENT_CHECKLIST.md`** - Tick-box checklist

### Project Info
- **`PROJECT_STATUS_SUMMARY.md`** - What's done, what's needed
- **`GPU_READINESS_ANALYSIS.md`** - Technical analysis
- **`README.md`** - Main project README

---

## 🔍 Monitor Your Experiments

### Reattach to Tmux

```bash
tmux attach -s experiments
```

### Watch Progress Live

```bash
# In a new terminal
watch -n 10 'cat results/progress.txt'
```

### Check GPU Usage

```bash
watch -n 5 nvidia-smi
```

### View Latest Results

```bash
# Latest report
ls -t results/report_*.txt | head -1 | xargs cat

# Latest accuracy
python -c "
import json, glob
files = sorted(glob.glob('results/bert_agnews_*.json'))
if files:
    with open(files[-1]) as f:
        r = json.load(f)
    print(f\"Latest: {files[-1].split('/')[-1]}\")
    print(f\"Accuracy: {r.get('final_test_accuracy', 'N/A'):.4f}\")
"
```

---

## 💾 Download Results Incrementally

Don't wait for everything to finish!

### After Phase 1

```bash
cd /workspace/dp-peft-research
tar -czf results_phase1.tar.gz results/*_eps8.0* checkpoints/*_eps8.0*

# From local machine:
scp runpod:/workspace/dp-peft-research/results_phase1.tar.gz ./
```

### After Specific Placements

```bash
# Download adapter_only results
tar -czf results_adapter.tar.gz \
    results/*adapter_only* \
    checkpoints/*adapter_only*
```

---

## 🆘 Troubleshooting

### Script Stopped?
```bash
# Just run it again - it will resume
bash scripts/runpod_text_experiments_resilient.sh
```

### Out of Memory?
```bash
# Edit script
nano scripts/runpod_text_experiments_resilient.sh
# Change line 20: BATCH_SIZE=64 to BATCH_SIZE=32
```

### Experiment Stuck?
```bash
# Check if actually running
ps aux | grep run_experiment

# If not, mark as pending and restart
sed -i 's/,running,/,pending,/' results/progress.txt
bash scripts/runpod_text_experiments_resilient.sh
```

### Want to Skip an Experiment?
```bash
# Mark as completed manually
echo "placement_name,8.0,completed,$(date +%Y%m%d_%H%M%S)" >> results/progress.txt
```

---

## ✅ Success Indicators

### Healthy Run
- ✅ `progress.txt` updating regularly
- ✅ New report files appearing
- ✅ GPU utilization 80-100%
- ✅ Accuracy improving in reports
- ✅ No errors in logs

### Warning Signs
- ⚠️ GPU utilization <50%
- ⚠️ Same experiment running >24 hours
- ⚠️ Multiple failed experiments
- ⚠️ Disk space <10GB

---

## 📈 Expected Results

### Accuracy Targets

| Placement | ε=8.0 | ε=1.0 |
|-----------|-------|-------|
| No DP | ~90% | ~90% |
| Adapter-Only DP | ~75-85% | ~65-75% |
| Head+Adapter DP | ~70-80% | ~60-70% |
| Last-Layer DP | ~65-75% | ~55-65% |
| Full-Model DP | ~60-70% | ~50-60% |
| Partial Backbone DP | ~65-75% | ~55-65% |

### Key Finding
**Adapter-Only DP should achieve best privacy-utility tradeoff!**

---

## 🎯 After Completion

### 1. Verify All Done

```bash
grep -c "completed" results/progress.txt
# Should output: 12
```

### 2. Download Everything

```bash
cd /workspace/dp-peft-research
tar -czf results_complete_text.tar.gz results/ checkpoints/

# From local machine:
scp runpod:/workspace/dp-peft-research/results_complete_text.tar.gz ./
```

### 3. Stop RunPod

- Go to RunPod dashboard
- Click "Stop" on your pod
- Verify charges stopped

### 4. Analyze Results

```bash
# On local machine
tar -xzf results_complete_text.tar.gz
jupyter notebook notebooks/results_analysis.ipynb
```

---

## 🎉 You're All Set!

Everything is ready for resilient, resumable text experiments:

- ✅ Code committed and ready to push
- ✅ Resilient script configured
- ✅ Progress tracking implemented
- ✅ Intermediate saving enabled
- ✅ MIA runs after each experiment
- ✅ Complete documentation provided

**Next step: Push to GitHub and deploy on RunPod!**

```bash
# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/dp-peft-research.git
git push -u origin main

# Then follow: UPDATED_INSTRUCTIONS.md
```

**Good luck with your experiments! 🚀**
