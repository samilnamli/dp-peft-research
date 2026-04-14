#!/bin/bash
# Quick script to push to GitHub

echo "=========================================="
echo "Push DP-PEFT to GitHub"
echo "=========================================="
echo ""
echo "Step 1: Create GitHub repository"
echo "  Go to: https://github.com/new"
echo "  Name: dp-peft-research"
echo "  Don't initialize with anything"
echo ""
read -p "Press Enter when repository is created..."
echo ""
echo "Step 2: Enter your GitHub username:"
read -p "Username: " username
echo ""
echo "Step 3: Adding remote and pushing..."
git remote add origin https://github.com/$username/dp-peft-research.git
git push -u origin main
echo ""
echo "=========================================="
echo "✓ Code pushed to GitHub!"
echo "=========================================="
echo ""
echo "Repository URL:"
echo "https://github.com/$username/dp-peft-research"
echo ""
echo "Next steps:"
echo "1. Go to RunPod.io and deploy L40S GPU"
echo "2. Follow: COMPLETE_EXECUTION_GUIDE.md"
echo ""
