#!/bin/bash
# Run full DP-PEFT experiment in tmux session
# This allows the experiment to continue running even if you disconnect

SESSION_NAME="dp_peft_experiment"
PROJECT_DIR="/home/asami/privacy/dp_peft"

# Check if tmux session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    echo "Creating new tmux session: $SESSION_NAME"
    
    # Create new tmux session
    tmux new-session -d -s $SESSION_NAME -c $PROJECT_DIR
    
    # Run the experiment
    tmux send-keys -t $SESSION_NAME "cd $PROJECT_DIR && source venv/bin/activate" C-m
    tmux send-keys -t $SESSION_NAME "echo 'Starting DP-PEFT Full Experiment with BERT + AG News'" C-m
    tmux send-keys -t $SESSION_NAME "echo 'GPU: RTX 2060'" C-m
    tmux send-keys -t $SESSION_NAME "echo ''" C-m
    
    # Run with reasonable settings for RTX 2060 (6GB VRAM)
    # batch_size=16 to fit in memory, 10 epochs, 10k samples for faster results
    tmux send-keys -t $SESSION_NAME "python scripts/run_full_experiment.py \
        --placements no_dp adapter_only head_adapter last_layer \
        --epsilon 8.0 \
        --epochs 10 \
        --batch_size 16 \
        --max_samples 10000 \
        --lr 2e-5 2>&1 | tee results/experiment_log.txt" C-m
    
    echo ""
    echo "Experiment started in tmux session: $SESSION_NAME"
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
else
    echo "Session $SESSION_NAME already exists!"
    echo "Attach with: tmux attach -t $SESSION_NAME"
fi
