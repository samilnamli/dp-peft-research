# ✅ UPDATED: Use Resilient Script for Text Experiments

## 🎯 What Changed

Instead of using `runpod_full_experiments.sh`, you should now use:

```bash
bash scripts/runpod_text_experiments_resilient.sh
```

## 🚀 Why This Is Better

### Old Script (`runpod_full_experiments.sh`)
- ❌ Runs everything (text + vision + privacy curves)
- ❌ No progress tracking
- ❌ If interrupted, starts from beginning
- ❌ No intermediate saving
- ❌ MIA runs at the end only

### New Script (`runpod_text_experiments_resilient.sh`)
- ✅ **Text experiments only** (BERT + AG News)
- ✅ **Progress tracking** (saves to `results/progress.txt`)
- ✅ **Resume capability** (skips completed experiments)
- ✅ **Intermediate saving** (after each placement)
- ✅ **Immediate MIA** (runs after each training)
- ✅ **Individual reports** (per experiment)
- ✅ **Error resilience** (continues on failure)

---

## 📋 Quick Start

### On RunPod GPU Server

```bash
# 1. Navigate to project
cd /workspace/dp-peft-research
source venv/bin/activate

# 2. Start resilient experiments in tmux
tmux new -s experiments

# 3. Run the new resilient script
bash scripts/runpod_text_experiments_resilient.sh

# 4. Detach from tmux
# Press: Ctrl+B, then D
```

**Done!** Your experiments will run with automatic progress tracking.

---

## 🔄 Resume After Interruption

If your GPU gets interrupted or you need to stop:

```bash
# Just run the same script again
bash scripts/runpod_text_experiments_resilient.sh
```

It will automatically:
- Skip completed experiments
- Resume from where it left off
- Show you the progress

---

## 📊 What Gets Run

### Phase 1: ε = 8.0
1. No DP → Save → Report → MIA ✅
2. Adapter-Only DP → Save → Report → MIA ✅
3. Head+Adapter DP → Save → Report → MIA ✅
4. Last-Layer DP → Save → Report → MIA ✅
5. Full-Model DP → Save → Report → MIA ✅
6. Partial Backbone DP → Save → Report → MIA ✅

### Phase 2: ε = 1.0
7-12. Same as above with ε=1.0

**Total: 12 experiments + 12 MIA evaluations**

**Time: ~5-6 days on L40S**

---

## 📁 Files Created

### Progress Tracking
```
results/progress.txt
```
Tracks completion status of each experiment.

### Individual Reports
```
results/report_no_dp_eps8.0_*.txt
results/report_adapter_only_eps8.0_*.txt
...
```
One report per experiment with results summary.

### Results & Checkpoints
```
results/bert_agnews_*.json
checkpoints/bert_agnews_*.pt
```
Saved after each experiment completes.

### MIA Results
```
results/mia_*.log
```
MIA evaluation for each placement.

---

## 🔍 Monitor Progress

### Check Status

```bash
# View progress
cat results/progress.txt

# Count completed
grep -c "completed" results/progress.txt

# View latest report
ls -t results/report_*.txt | head -1 | xargs cat
```

### Live Monitoring

```bash
# Reattach to tmux
tmux attach -s experiments

# Or watch progress file
watch -n 10 'cat results/progress.txt'
```

---

## 💾 Download Results Incrementally

You can download results as they complete:

```bash
# After Phase 1 completes
tar -czf results_phase1.tar.gz results/*_eps8.0*

# Download to local machine
scp runpod:/workspace/dp-peft-research/results_phase1.tar.gz ./
```

---

## ⏱️ Time & Cost Estimates

### L40S ($0.79/hr)
- **Time:** ~140 hours (~5-6 days)
- **Cost:** ~$111

### A100 ($1.14/hr)
- **Time:** ~120 hours (~5 days)
- **Cost:** ~$137

### RTX 4090 ($0.44/hr)
- **Time:** ~150 hours (~6 days)
- **Cost:** ~$66

---

## 📖 Documentation

For detailed information, see:

- **`RESILIENT_EXPERIMENTS_GUIDE.md`** - Complete guide for resilient script
- **`COMPLETE_EXECUTION_GUIDE.md`** - Full execution guide
- **`RUNPOD_SETUP.md`** - RunPod deployment guide

---

## 🆘 Quick Troubleshooting

### Script stopped?
```bash
# Just run it again - it will resume
bash scripts/runpod_text_experiments_resilient.sh
```

### Out of memory?
```bash
# Edit script, change BATCH_SIZE=64 to 32
nano scripts/runpod_text_experiments_resilient.sh
```

### Check if running?
```bash
# Reattach to tmux
tmux attach -s experiments
```

---

## ✅ Success Checklist

After completion, you should have:

- [ ] 12 result JSON files (6 placements × 2 ε values)
- [ ] 12 checkpoint .pt files
- [ ] 12 MIA evaluation logs
- [ ] 12 individual reports
- [ ] 1 progress.txt file showing all completed
- [ ] 1 final summary file

---

## 🎯 Next Steps

1. **Verify completion:**
   ```bash
   grep -c "completed" results/progress.txt  # Should be 12
   ```

2. **Download everything:**
   ```bash
   tar -czf results_complete.tar.gz results/ checkpoints/
   ```

3. **Stop RunPod instance** to avoid charges

4. **Analyze results** locally with Jupyter notebook

---

**Use the resilient script for better reliability! 🚀**
