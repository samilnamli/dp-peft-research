# GitHub Repository Setup Guide

Step-by-step instructions to push this project to GitHub.

---

## Step 1: Create GitHub Repository

1. **Go to GitHub.com** and log in
2. **Click "+" → "New repository"**
3. **Repository settings:**
   - **Name:** `dp-peft-research` (or your preferred name)
   - **Description:** "Differential Privacy Placement in Parameter-Efficient Fine-Tuning"
   - **Visibility:** Public or Private (your choice)
   - **DO NOT** initialize with README, .gitignore, or license (we have these)
4. **Click "Create repository"**
5. **Copy the repository URL** (e.g., `https://github.com/YOUR_USERNAME/dp-peft-research.git`)

---

## Step 2: Initialize Git Repository Locally

```bash
# Navigate to project directory
cd /home/asami/privacy/dp_peft

# Initialize git repository
git init

# Add all files
git add .

# Check what will be committed
git status

# Create initial commit
git commit -m "Initial commit: DP-PEFT research project

- Complete implementation of 6 DP placement strategies
- BERT text model with LoRA/Adapter support
- Opacus DP-SGD integration
- Training pipeline with metrics tracking
- Membership Inference Attack implementation
- Experiment scripts and configuration files
- Analysis notebooks and documentation"
```

---

## Step 3: Connect to GitHub and Push

```bash
# Add remote repository (replace with your GitHub URL)
git remote add origin https://github.com/YOUR_USERNAME/dp-peft-research.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

**If prompted for credentials:**
- **Username:** Your GitHub username
- **Password:** Use a Personal Access Token (not your password)
  - Generate at: https://github.com/settings/tokens
  - Select scopes: `repo` (full control)

---

## Step 4: Verify Upload

1. **Go to your GitHub repository** in browser
2. **Check that all files are present:**
   - ✅ `README.md`
   - ✅ `requirements.txt`
   - ✅ `setup.py`
   - ✅ `dp_peft/` directory
   - ✅ `scripts/` directory
   - ✅ `configs/` directory
   - ✅ `RUNPOD_SETUP.md`
   - ✅ `.gitignore`

3. **Verify .gitignore is working:**
   - ❌ `venv/` should NOT be uploaded
   - ❌ `results/*.json` should NOT be uploaded
   - ❌ `__pycache__/` should NOT be uploaded
   - ✅ `results/.gitkeep` should be present

---

## Step 5: Add Repository Description and Topics

On GitHub repository page:

1. **Click "About" gear icon** (top right)
2. **Add description:**
   ```
   Research project investigating optimal differential privacy placement in parameter-efficient fine-tuning (PEFT) for NLP and vision models
   ```
3. **Add topics (tags):**
   - `differential-privacy`
   - `machine-learning`
   - `deep-learning`
   - `privacy-preserving-ml`
   - `peft`
   - `lora`
   - `opacus`
   - `pytorch`
   - `transformers`
   - `research`

4. **Save changes**

---

## Step 6: Create Releases (Optional)

For versioning your experiments:

```bash
# Tag current version
git tag -a v0.1.0 -m "Initial release: Infrastructure complete, hyperparameter tuning needed"
git push origin v0.1.0
```

On GitHub:
1. Go to "Releases" → "Create a new release"
2. Select tag `v0.1.0`
3. Add release notes
4. Publish release

---

## Step 7: Update README with GitHub Badge

Add to top of `README.md`:

```markdown
# Where to Privatize? Differential Privacy Placement in PEFT

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Rest of README...]
```

Commit and push:
```bash
git add README.md
git commit -m "Add badges to README"
git push
```

---

## Step 8: Set Up GitHub Actions (Optional)

Create `.github/workflows/tests.yml` for automated testing:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
    - name: Run quick test
      run: |
        python -c "import dp_peft; print('Package imported successfully')"
```

---

## Common Issues and Solutions

### Issue: Large files rejected

```bash
# If you accidentally committed large files
git rm --cached results/*.json
git rm --cached checkpoints/*.pt
git commit -m "Remove large files"
git push
```

### Issue: Authentication failed

**Solution:** Use Personal Access Token instead of password
1. Go to https://github.com/settings/tokens
2. Generate new token (classic)
3. Select `repo` scope
4. Copy token and use as password

**Or use SSH:**
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub: Settings → SSH Keys → New SSH key
cat ~/.ssh/id_ed25519.pub

# Change remote to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/dp-peft-research.git
```

### Issue: Merge conflicts

```bash
# If you made changes on GitHub and locally
git pull --rebase origin main
# Resolve conflicts if any
git push
```

---

## Maintenance Commands

### Update repository after changes

```bash
# Check status
git status

# Add specific files
git add dp_peft/models/text_model.py
git add scripts/run_experiment.py

# Or add all changes
git add .

# Commit with descriptive message
git commit -m "Fix hyperparameters for DP placements"

# Push to GitHub
git push
```

### Create branches for experiments

```bash
# Create branch for hyperparameter tuning
git checkout -b hyperparameter-tuning

# Make changes and commit
git add .
git commit -m "Test new hyperparameters"
git push -u origin hyperparameter-tuning

# Create pull request on GitHub
# After merging, switch back to main
git checkout main
git pull
```

---

## Repository Structure on GitHub

```
dp-peft-research/
├── .github/
│   └── workflows/          # CI/CD (optional)
├── configs/                # YAML configs
├── dp_peft/               # Main package
│   ├── models/
│   ├── privacy/
│   ├── training/
│   ├── data/
│   ├── attacks/
│   └── utils/
├── scripts/               # Experiment runners
├── notebooks/             # Analysis notebooks
├── results/               # Results (mostly gitignored)
│   └── .gitkeep
├── checkpoints/           # Model checkpoints (gitignored)
│   └── .gitkeep
├── .gitignore
├── README.md
├── RUNPOD_SETUP.md       # RunPod deployment guide
├── GPU_READINESS_ANALYSIS.md
├── IMMEDIATE_ACTION_PLAN.md
├── PROJECT_STATUS_SUMMARY.md
├── requirements.txt
└── setup.py
```

---

## Next Steps

After pushing to GitHub:

1. ✅ **Share repository URL** with collaborators
2. ✅ **Clone on RunPod** following RUNPOD_SETUP.md
3. ✅ **Start experiments** on GPU server
4. ✅ **Commit results** (if small) or document in issues
5. ✅ **Update README** with actual results as they come in

---

## Quick Reference

```bash
# Daily workflow
git status                    # Check changes
git add .                     # Stage all changes
git commit -m "message"       # Commit
git push                      # Push to GitHub

# Sync with remote
git pull                      # Get latest changes

# View history
git log --oneline            # Compact history
git log --graph --all        # Visual history

# Undo changes
git checkout -- file.py      # Discard local changes
git reset HEAD~1             # Undo last commit (keep changes)
git reset --hard HEAD~1      # Undo last commit (discard changes)
```

---

## Your Repository is Ready! 🎉

Once pushed to GitHub, you can:
- Clone it on any machine (including RunPod)
- Collaborate with others
- Track changes and experiments
- Share your research
- Deploy to cloud GPU servers

**Next:** Follow RUNPOD_SETUP.md to deploy on GPU server
