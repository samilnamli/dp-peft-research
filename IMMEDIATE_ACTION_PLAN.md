# Immediate Action Plan: GPU Text Experiments

## Priority 1: Fix DP Hyperparameters (TODAY)

### Problem
All DP placements stuck at ~24% accuracy (random guessing) while No-DP achieves 90%.

### Root Cause
DP-SGD requires different hyperparameters than standard fine-tuning:
- Current LR: 2e-5 (too low for noisy gradients)
- Current clipping: 1.0 (too aggressive)
- Current samples: 10K (insufficient for DP)

### Solution: Test Improved Hyperparameters

#### Quick Test Script
```bash
cd /home/asami/privacy/dp_peft
source venv/bin/activate

# Test 1: Higher learning rate
python scripts/run_experiment.py \
    --model bert \
    --dataset agnews \
    --placement adapter_only \
    --epsilon 8.0 \
    --epochs 10 \
    --batch_size 32 \
    --lr 5e-4 \
    --device cuda

# Expected: Should see >50% accuracy by epoch 5
```

#### Hyperparameter Grid to Test
| Parameter | Current | Test Values |
|-----------|---------|-------------|
| Learning Rate | 2e-5 | **5e-4**, 1e-3, 2e-3 |
| Max Grad Norm | 1.0 | **5.0**, 10.0 |
| Batch Size | 16 | **32**, 64 (with grad accum) |
| Samples | 10K | **20K**, 120K (full) |

**Start with bold values** - most likely to work.

---

## Priority 2: Validate All Placements (DAY 2)

Once hyperparameters are fixed, run quick validation:

```bash
# Run all placements with fixed hyperparameters
python scripts/run_all_placements.py \
    --model bert \
    --dataset agnews \
    --epsilon 8.0 \
    --epochs 10 \
    --batch_size 32 \
    --lr 5e-4 \
    --max_samples 20000
```

**Success Criteria:**
- Adapter-Only DP: >70% accuracy
- Head+Adapter DP: >65% accuracy
- Last-Layer DP: >60% accuracy
- Full-Model DP: >55% accuracy

---

## Priority 3: Full Production Runs (DAYS 3-7)

### Setup Long-Running Experiments

```bash
# Create tmux session
tmux new -s dp_full_experiments
cd /home/asami/privacy/dp_peft
source venv/bin/activate

# Run full sweep (will take ~48 hours)
python scripts/run_all_placements.py \
    --model bert \
    --dataset agnews \
    --epsilon 8.0 \
    --epochs 20 \
    --batch_size 32 \
    --lr 5e-4 \
    --max_samples 120000  # Full AG News dataset

# Detach: Ctrl+B, then D
# Check progress: tmux attach -s dp_full_experiments
```

### Monitor Progress
```bash
# In another terminal
watch -n 60 'tail -50 results/experiment_log.txt'

# Check GPU usage
watch -n 5 nvidia-smi
```

---

## Priority 4: Privacy-Utility Curves (DAYS 8-10)

```bash
# For each placement, sweep epsilon values
for placement in no_dp adapter_only head_adapter last_layer full_dp partial_backbone; do
    python scripts/run_privacy_curve.py \
        --model bert \
        --dataset agnews \
        --placement $placement \
        --epochs 20 \
        --batch_size 32 \
        --lr 5e-4
done
```

This generates data for the main paper figure.

---

## Monitoring & Debugging

### Check if Experiment is Learning
```bash
# Should see accuracy increasing over epochs
tail -100 results/experiment_log.txt | grep "Test Acc"
```

**Good sign:** Test Acc increasing from ~0.25 → 0.70+  
**Bad sign:** Test Acc stuck at ~0.25 (still broken hyperparameters)

### GPU Memory Issues
```bash
# If OOM errors:
# 1. Reduce batch size to 16
# 2. Use gradient accumulation
# 3. Check memory: nvidia-smi
```

### Experiment Taking Too Long
```bash
# Kill and restart with fewer samples
tmux kill-session -s dp_full_experiments

# Restart with 50K samples instead of 120K
python scripts/run_all_placements.py \
    --max_samples 50000 \
    ...
```

---

## Expected Timeline

### Optimistic (Everything Works)
- **Day 1:** Fix hyperparameters, validate on small dataset ✅
- **Days 2-3:** Full text experiments (ε=8) running
- **Days 4-5:** Privacy curves (ε sweep)
- **Days 6-7:** Analysis and figures
- **Total: 1 week**

### Realistic (Some Debugging Needed)
- **Days 1-2:** Hyperparameter tuning and debugging
- **Days 3-5:** Full text experiments
- **Days 6-8:** Privacy curves
- **Days 9-10:** Analysis
- **Total: 2 weeks**

### Pessimistic (Major Issues)
- **Days 1-5:** Extensive hyperparameter search
- **Days 6-10:** Full experiments
- **Days 11-14:** Analysis
- **Total: 3 weeks**

---

## Quick Reference Commands

### Activate Environment
```bash
cd /home/asami/privacy/dp_peft
source venv/bin/activate
```

### Check GPU
```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Run Single Quick Test
```bash
python scripts/run_experiment.py \
    --model bert --dataset agnews --placement adapter_only \
    --epsilon 8.0 --epochs 5 --batch_size 32 --lr 5e-4
```

### View Results
```bash
cat results/bert_agnews_adapter_only_eps8.0.json | python -m json.tool
```

### Kill Stuck Experiment
```bash
tmux kill-session -s dp_full_experiments
pkill -f run_experiment.py
```

---

## Red Flags to Watch For

🔴 **Accuracy stuck at ~25%** → Hyperparameters still wrong  
🔴 **OOM errors** → Reduce batch size  
🔴 **Very slow training** → Check GPU utilization (nvidia-smi)  
🔴 **ε not converging to target** → Noise multiplier calibration issue  
🔴 **NaN losses** → Learning rate too high or numerical instability  

---

## Success Metrics

### After Day 1 (Hyperparameter Fix)
- [ ] Adapter-Only DP reaches >50% accuracy in 5 epochs
- [ ] Training loss decreasing consistently
- [ ] No OOM errors on batch_size=32

### After Week 1 (Full Text Experiments)
- [ ] All 6 placements tested at ε=8
- [ ] Adapter-Only DP: 70-85% accuracy
- [ ] Full-Model DP: 55-70% accuracy
- [ ] Results saved in `results/` directory

### After Week 2 (Complete Text Track)
- [ ] Privacy-utility curves generated
- [ ] MIA evaluation completed
- [ ] Figures generated in notebook
- [ ] Ready for vision experiments

---

## Next Steps After Text Experiments

1. **Vision Experiments** (ViT + CIFAR-10)
2. **Results Analysis** (notebook updates)
3. **Paper Writing** (methodology, results, discussion)
4. **Reproducibility** (document final hyperparameters)

---

## Contact Points for Help

- **Opacus Issues:** Check GitHub issues, examples in opacus/examples/
- **PEFT Issues:** HuggingFace PEFT documentation
- **DP-SGD Theory:** Abadi et al. 2016 paper, DP-SGD tutorials

---

## Emergency Fallback Plan

If DP hyperparameters cannot be fixed after 3 days:

1. **Pivot to analysis of infrastructure:** Document the implementation
2. **Use synthetic results:** Show what *should* happen theoretically
3. **Focus on No-DP baseline:** Deep analysis of PEFT without privacy
4. **Discuss challenges:** Make the hyperparameter sensitivity a finding

But this should NOT be necessary - DP-SGD is well-established and should work with proper tuning.
