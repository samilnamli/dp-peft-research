# Resilient Text Experiments Guide

**Run text experiments with automatic progress tracking and resume capability.**

---

## 🎯 What's Different?

This resilient script (`runpod_text_experiments_resilient.sh`) is designed for GPU environments that may be interrupted:

### Key Features

1. **✅ Automatic Progress Tracking**
   - Tracks which experiments are completed
   - Saves progress to `results/progress.txt`
   - Can resume from where it left off

2. **✅ Intermediate Saving & Reporting**
   - Saves results after EACH placement completes
   - Generates individual reports per experiment
   - Runs MIA immediately after each training

3. **✅ Error Resilience**
   - If one experiment fails, continues with next
   - Marks failed experiments for retry
   - Doesn't lose progress on interruption

4. **✅ Text Experiments Only**
   - Focuses on BERT + AG News
   - Skips vision experiments (ViT)
   - Skips privacy curves (can add later)

5. **✅ Real-time Progress Display**
   - Shows completion status after each experiment
   - Visual progress indicators (✅ ⏳ ❌)
   - Percentage completion

---

## 🚀 Quick Start

### On RunPod GPU Server

```bash
# Navigate to project
cd /workspace/dp-peft-research
source venv/bin/activate

# Start resilient experiments
tmux new -s experiments
bash scripts/runpod_text_experiments_resilient.sh

# Detach: Ctrl+B, D
```

**That's it!** The script will:
1. Run all 6 placements at ε=8.0
2. Run all 6 placements at ε=1.0
3. Run MIA after each placement
4. Save progress continuously
5. Generate reports after each completion

---

## 📊 What Gets Run

### Phase 1: ε = 8.0 (6 placements)
1. No DP → Save → Report → MIA
2. Adapter-Only DP → Save → Report → MIA
3. Head+Adapter DP → Save → Report → MIA
4. Last-Layer DP → Save → Report → MIA
5. Full-Model DP → Save → Report → MIA
6. Partial Backbone DP → Save → Report → MIA

### Phase 2: ε = 1.0 (6 placements)
7-12. Same as above with ε=1.0

**Total: 12 experiments + 12 MIA evaluations**

---

## 🔄 Resume After Interruption

If your GPU pod gets interrupted or you need to stop:

```bash
# Just run the same script again!
bash scripts/runpod_text_experiments_resilient.sh
```

**The script will:**
- ✅ Skip completed experiments automatically
- ✅ Resume from where it left off
- ✅ Show you what's already done
- ✅ Continue with pending experiments

**Example output:**
```
⏭️  Skipping no_dp | ε=8.0 (already completed)
⏭️  Skipping adapter_only | ε=8.0 (already completed)
Running: head_adapter | ε=8.0
[continues from here...]
```

---

## 📁 Files Created

### Progress Tracking
```
results/progress.txt
```
**Format:**
```
placement,epsilon,status,timestamp
no_dp,8.0,completed,20260417_193045
adapter_only,8.0,completed,20260417_201523
head_adapter,8.0,running,20260417_205012
...
```

### Individual Reports
```
results/report_no_dp_eps8.0_20260417_193045.txt
results/report_adapter_only_eps8.0_20260417_201523.txt
...
```

Each report contains:
- Experiment configuration
- Completion timestamp
- GPU info
- Results summary (accuracy, loss, epsilon)
- File locations

### Results & Checkpoints
```
results/bert_agnews_no_dp_eps8.0.json
results/bert_agnews_adapter_only_eps8.0.json
...

checkpoints/bert_agnews_no_dp_eps8.0.pt
checkpoints/bert_agnews_adapter_only_eps8.0.pt
...
```

### MIA Results
```
results/mia_no_dp_eps8.0_resilient_*.log
results/mia_adapter_only_eps8.0_resilient_*.log
...
```

### Final Summary
```
results/final_summary_resilient_20260417_193045.txt
```

---

## 📊 Monitor Progress

### Check Progress Anytime

```bash
# View progress file
cat results/progress.txt

# Count completed
grep -c "completed" results/progress.txt

# Count pending
grep -c "pending" results/progress.txt

# View latest report
ls -t results/report_*.txt | head -1 | xargs cat
```

### Live Monitoring

```bash
# In tmux session, split screen
Ctrl+B, %  # Split vertically

# In new pane, watch progress
watch -n 10 'grep "^[^#]" results/progress.txt | tail -20'

# Or watch GPU
watch -n 5 nvidia-smi
```

### Check Specific Experiment

```bash
# Check if specific experiment completed
grep "adapter_only,8.0" results/progress.txt

# View its report
cat results/report_adapter_only_eps8.0_*.txt

# View its results
cat results/bert_agnews_adapter_only_eps8.0.json | python -m json.tool
```

---

## 🛠️ Advanced Usage

### Modify Which Experiments Run

Edit the script to change which placements to run:

```bash
nano scripts/runpod_text_experiments_resilient.sh

# Find this line (around line 280):
for placement in no_dp adapter_only head_adapter last_layer full_dp partial_backbone; do

# Change to run only specific placements:
for placement in no_dp adapter_only full_dp; do
```

### Change Hyperparameters

```bash
# Edit at top of script (lines 18-23):
BATCH_SIZE=64    # Change to 32 if OOM
EPOCHS=20        # Change to 15 for faster runs
LR=5e-4          # Keep this for DP experiments
```

### Reset Progress

```bash
# To start fresh (deletes progress tracking)
rm results/progress.txt

# Then run script again
bash scripts/runpod_text_experiments_resilient.sh
```

### Retry Failed Experiments

```bash
# Find failed experiments
grep "failed" results/progress.txt

# Mark them as pending to retry
sed -i 's/,failed,/,pending,/' results/progress.txt

# Run script again
bash scripts/runpod_text_experiments_resilient.sh
```

---

## 💾 Download Results Incrementally

You don't have to wait for all experiments to finish!

### Download After Each Phase

```bash
# After Phase 1 (ε=8.0) completes
cd /workspace/dp-peft-research
tar -czf results_phase1.tar.gz \
    results/*_eps8.0*.json \
    checkpoints/*_eps8.0*.pt \
    results/report_*_eps8.0*.txt

# Download to local machine
scp runpod:/workspace/dp-peft-research/results_phase1.tar.gz ./
```

### Download Specific Placements

```bash
# Download just adapter_only results
tar -czf results_adapter_only.tar.gz \
    results/*adapter_only*.json \
    checkpoints/*adapter_only*.pt \
    results/report_*adapter_only*.txt

scp runpod:/workspace/dp-peft-research/results_adapter_only.tar.gz ./
```

---

## ⏱️ Time Estimates

### Per Experiment (BERT + AG News, 20 epochs)

| Placement | Trainable Params | Time (L40S) | Time (A100) |
|-----------|------------------|-------------|-------------|
| No DP | 888K | ~8-10 hours | ~6-8 hours |
| Adapter-Only DP | 295K | ~10-12 hours | ~8-10 hours |
| Head+Adapter DP | 296K | ~10-12 hours | ~8-10 hours |
| Last-Layer DP | 1K | ~8-10 hours | ~6-8 hours |
| Full-Model DP | 110M | ~14-16 hours | ~12-14 hours |
| Partial Backbone DP | 1M | ~12-14 hours | ~10-12 hours |

### Total Time

- **Phase 1 (ε=8.0):** ~60-70 hours
- **Phase 2 (ε=1.0):** ~60-70 hours
- **MIA (all):** ~6-8 hours
- **Total:** ~130-150 hours (~5-6 days)

---

## 💰 Cost Estimates

### L40S ($0.79/hr)
- **Total time:** ~140 hours
- **Total cost:** ~$111

### A100 ($1.14/hr)
- **Total time:** ~120 hours
- **Total cost:** ~$137

### RTX 4090 ($0.44/hr)
- **Total time:** ~150 hours
- **Total cost:** ~$66

---

## 🆘 Troubleshooting

### Script stops unexpectedly

```bash
# Check if tmux session still exists
tmux ls

# Reattach
tmux attach -s experiments

# Check last log
tail -100 results/*_resilient_*.log | grep -i error
```

### Experiment marked as running but not actually running

```bash
# Check if process exists
ps aux | grep run_experiment

# If not running, mark as pending and restart
sed -i 's/,running,/,pending,/' results/progress.txt
bash scripts/runpod_text_experiments_resilient.sh
```

### Out of memory

```bash
# Edit script to reduce batch size
nano scripts/runpod_text_experiments_resilient.sh
# Change: BATCH_SIZE=64 to BATCH_SIZE=32

# Mark failed experiments as pending
sed -i 's/,failed,/,pending,/' results/progress.txt

# Restart
bash scripts/runpod_text_experiments_resilient.sh
```

### Want to skip an experiment

```bash
# Mark as completed manually
echo "placement_name,8.0,completed,$(date +%Y%m%d_%H%M%S)" >> results/progress.txt
```

---

## ✅ Success Indicators

### Healthy Run
- ✅ Progress file updating regularly
- ✅ New report files appearing
- ✅ GPU utilization 80-100%
- ✅ Accuracy improving in reports
- ✅ No error messages in logs

### Warning Signs
- ⚠️ GPU utilization <50%
- ⚠️ Same experiment running >24 hours
- ⚠️ Multiple failed experiments
- ⚠️ Disk space <10GB

---

## 📈 Expected Results

After all experiments complete:

### Accuracy Targets

| Placement | ε=8.0 | ε=1.0 |
|-----------|-------|-------|
| No DP | ~90% | ~90% |
| Adapter-Only DP | ~75-85% | ~65-75% |
| Head+Adapter DP | ~70-80% | ~60-70% |
| Last-Layer DP | ~65-75% | ~55-65% |
| Full-Model DP | ~60-70% | ~50-60% |
| Partial Backbone DP | ~65-75% | ~55-65% |

### MIA Results
- All DP placements: AUC ~0.50-0.55 (good privacy)
- No DP: AUC ~0.60-0.70 (no privacy)

---

## 🎯 Next Steps After Completion

### 1. Verify All Completed

```bash
# Check progress
grep -c "completed" results/progress.txt
# Should be: 12

# Check for failures
grep "failed" results/progress.txt
# Should be: empty
```

### 2. Download Everything

```bash
cd /workspace/dp-peft-research
tar -czf results_complete_text.tar.gz results/ checkpoints/

# From local machine
scp runpod:/workspace/dp-peft-research/results_complete_text.tar.gz ./
```

### 3. Stop RunPod Instance

- Go to RunPod dashboard
- Click "Stop" on your pod
- Verify charges stopped

### 4. Analyze Results

```bash
# On local machine
tar -xzf results_complete_text.tar.gz
cd dp-peft-research
jupyter notebook notebooks/results_analysis.ipynb
```

---

## 🔗 Related Documentation

- **Quick Start:** `QUICKSTART.md`
- **Full Guide:** `COMPLETE_EXECUTION_GUIDE.md`
- **RunPod Setup:** `RUNPOD_SETUP.md`
- **Project Status:** `PROJECT_STATUS_SUMMARY.md`

---

## 💡 Pro Tips

1. **Start with small test:** Run 2-3 epochs first to verify everything works
2. **Monitor first experiment:** Stay attached to tmux for first placement
3. **Check progress daily:** Reattach and verify things are running
4. **Download incrementally:** Don't wait for everything to finish
5. **Keep progress.txt:** Backup this file, it's your resume point

---

**Your experiments are now resilient and resumable! 🚀**
