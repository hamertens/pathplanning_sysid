# run.sh
#!/usr/bin/env bash
set -euo pipefail
source activate pytorch_gpu

# ─── Config ────────────────────────────────────────────────────────────────
UCT_HORIZON=7
UCT_SIMS=3000
ITERATIONS=250

THRESHOLD=0.2
SCALE=5.0

FT_EPOCHS=30
FT_LR=1e-3

MAX_ITERATIONS=100
EXP_DECAY_RATE=1.0

STATE=output_data/current_state.npy
KNOWN=output_data/known_indices.npy
TRAJ=output_data/trajectory.csv
LOG=output_data/direct_control_log.csv

INIT_MODEL=random_walk_surrogate.pt
MODEL_OUT=surrogate_model.pt
TEST_SET=test_set.npz

rm -f "$STATE" "$KNOWN" "$TRAJ" "$LOG"
mkdir -p output_data

MODEL_IN=$INIT_MODEL

for i in $(seq 1 $ITERATIONS); do
  echo "=== Iteration $i: Planning → Driving → Learning ==="

  python3 mcts_uct.py \
    --uct_horizon        $UCT_HORIZON \
    --uct_simulations    $UCT_SIMS \
    --known-indices-file $KNOWN \
    --trajectory-csv     $TRAJ \
    --current-state-file $STATE \
    --scale-factor       $SCALE

  python3 direct_control_pretrained.py \
    --trajectory-csv   $TRAJ \
    --init-state-file  $STATE \
    --state-out-file   $STATE \
    --log-file         $LOG \
    --threshold        $THRESHOLD \
    --scale-factor     $SCALE \
    --model-file       $MODEL_IN \
    --max-iterations   $MAX_ITERATIONS \
    --strategy         ranked \
    --ranked-k         100 \
    --exp-decay-rate   $EXP_DECAY_RATE





  python3 fine_tune_model.py \
    --iteration $i \
    --log-file   $LOG \
    --model-in   $MODEL_IN \
    --model-out  $MODEL_OUT \
    --test-set   $TEST_SET \
    --epochs     $FT_EPOCHS \
    --lr         $FT_LR

  MODEL_IN=$MODEL_OUT
done

