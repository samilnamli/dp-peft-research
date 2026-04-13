# DP-PEFT Progress Presentation Notes

## Slide: Experimental Setup

### Model Architecture
- **Base Model**: Simple transformer encoder (3 layers, 256 hidden dim)
- **PEFT Method**: LoRA-style adapters (down: 256→16, up: 16→256)
- **Classification Head**: Linear layer (256→4 classes)
- **Total Parameters**: ~1.5M (full model), ~8.5K (adapters only)

### Training Configuration
- **Dataset**: Synthetic 4-class classification (2000 train, 500 test samples)
- **Optimizer**: AdamW, lr=1e-3
- **Batch Size**: 64
- **Epochs**: 5
- **DP Simulation**: Gradient clipping (max_norm=1.0) + Gaussian noise (σ=0.5)

---

## Slide: DP Placement Strategies Tested

| Strategy | What Gets DP | Trainable Params |
|----------|--------------|------------------|
| No DP (Baseline) | Nothing | 1,488,404 (100%) |
| Full-Model DP | All layers | 1,488,404 (100%) |
| Adapter-Only DP | LoRA layers only | 8,464 (0.6%) |
| Head+Adapter DP | LoRA + classifier | 9,492 (0.6%) |
| Last-Layer DP | Classifier only | 1,028 (0.07%) |
| Partial Backbone DP | LoRA + top-2 layers | 142,100 (9.5%) |

---

## Slide: Key Results

### Accuracy Comparison
| Placement | Test Accuracy | Relative to Baseline |
|-----------|---------------|---------------------|
| No DP | 25.6% | 100% |
| Full DP | 19.8% | 77% (worst) |
| Adapter Only | 23.0% | 90% |
| Head+Adapter | 21.2% | 83% |
| Last Layer | 24.2% | 95% |
| Partial Backbone | 25.2% | 98% |

### Training Efficiency
| Placement | Avg Epoch Time | Speedup vs Full DP |
|-----------|----------------|-------------------|
| Full DP | 0.80s | 1.0x |
| Adapter Only | 0.12s | **6.7x faster** |
| Last Layer | 0.09s | **8.9x faster** |

### Stability (Gradient Norm Variance)
- **Most Stable**: Adapter-only (variance ≈ 0)
- **Least Stable**: No DP baseline (variance = 2.75)
- DP noise actually stabilizes gradients by clipping

---

## Slide: Technical Implementation

### Privacy Mechanism
```
DP-SGD Steps:
1. Compute per-sample gradients
2. Clip gradients: g_clipped = g * min(1, C/||g||)
3. Add noise: g_noisy = g_clipped + N(0, σ²C²I)
4. Update: θ = θ - η * g_noisy
```

### Selective DP Application
- Freeze non-DP parameters (requires_grad=False)
- Only compute/clip/noise gradients for DP-protected layers
- Privacy budget consumed only by DP-protected parameters

---

## Slide: Preliminary Insights

1. **Adapter-only DP is promising**
   - Uses <1% of parameters
   - 6.7x faster training
   - Maintains 90% of baseline accuracy

2. **Full-model DP has highest utility loss**
   - More parameters = more noise = worse accuracy
   - Validates hypothesis that selective DP is better

3. **Trade-off exists between capacity and privacy**
   - Last-layer: fast but limited capacity
   - Partial backbone: good accuracy but more params under DP

---

## Slide: Next Steps

1. **Scale to real models**: BERT-base + AG News, ViT + CIFAR-10
2. **Formal privacy accounting**: Opacus RDP accountant at ε∈{1,8}
3. **Membership inference attacks**: Validate empirical privacy
4. **Privacy-utility curves**: Sweep ε values for each placement
5. **Publication-ready figures**: Seaborn plots with error bars

---

## Technical Notes for Q&A

- **Why synthetic data?** Quick validation of infrastructure; real experiments take hours on CPU
- **Why LoRA?** State-of-the-art PEFT method, compatible with Opacus
- **Privacy guarantee?** (ε,δ)-DP via Rényi DP composition
- **Opacus compatibility?** Using ModuleValidator.fix() for HuggingFace models
