#!/bin/bash
# Run FIXED DP-PEFT experiment in tmux session with proper hyperparameters

SESSION_NAME="dp_peft_fixed"
PROJECT_DIR="/home/asami/privacy/dp_peft"

# Check if tmux session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    echo "Creating new tmux session: $SESSION_NAME"
    
    # Create new tmux session
    tmux new-session -d -s $SESSION_NAME -c $PROJECT_DIR
    
    # Run the experiment
    tmux send-keys -t $SESSION_NAME "cd $PROJECT_DIR && source venv/bin/activate" C-m
    tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
    tmux send-keys -t $SESSION_NAME "echo 'FIXED DP-PEFT Experiment'" C-m
    tmux send-keys -t $SESSION_NAME "echo 'Proper hyperparameters for DP-SGD'" C-m
    tmux send-keys -t $SESSION_NAME "echo 'GPU: RTX 2060'" C-m
    tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
    tmux send-keys -t $SESSION_NAME "echo ''" C-m
    
    # Run with FIXED hyperparameters
    # - LR: 5e-4 for DP (25x higher)
    # - Max grad norm: 5.0 (5x higher clipping)
    # - Samples: 20k (2x more data)
    # - Epochs: 15 (better convergence)
    tmux send-keys -t $SESSION_NAME "python scripts/run_fixed_experiment.py \
        --placements no_dp adapter_only head_adapter last_layer \
        --epsilon 8.0 \
        --epochs 15 \
        --batch_size 32 \
        --max_samples 20000 2>&1 | tee results/fixed_experiment_log.txt" C-m
    
    echo ""
    echo "✓ Fixed experiment started in tmux session: $SESSION_NAME"
    echo ""
    echo "Key improvements:"
    echo "  • Learning rate: 5e-4 (was 2e-5) for DP placements"
    echo "  • Gradient clipping: 5.0 (was 1.0)"
    echo "  • Training samples: 20,000 (was 10,000)"
    echo "  • Epochs: 15 (was 10)"
    echo ""
    echo "To attach to the session and see progress:"
    echo "  tmux attach -t $SESSION_NAME"
    echo ""
    echo "To detach from session (keep running):"
    echo "  Press Ctrl+B, then D"
    echo ""
    echo "To kill the session:"
    echo "  tmux kill-session -t $SESSION_NAME"
    echo ""
    echo "Estimated time: ~3-4 hours for all placements"
    echo ""
else
    echo "Session $SESSION_NAME already exists!"
    echo "Attach with: tmux attach -t $SESSION_NAME"
fi
