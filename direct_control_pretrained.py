#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import math
import argparse
import torch
import torch.nn as nn
from car_dynamics import CarDynamics

# --------------------------
# 1) CLI
# --------------------------
parser = argparse.ArgumentParser(
    description='Drive to the latest waypoint with different control strategies'
)
parser.add_argument('--trajectory-csv',  default='output_data/trajectory.csv')
parser.add_argument('--init-state-file', default='output_data/current_state.npy')
parser.add_argument('--state-out-file',  default='output_data/current_state.npy')
parser.add_argument('--log-file',        default='output_data/direct_control_log.csv')
parser.add_argument('--error-csv',       default='output_data/test_errors.csv',
                    help='Path to test_errors CSV')
parser.add_argument('--threshold',       type=float, default=0.5)
parser.add_argument('--scale-factor',    type=float, default=5.0)
parser.add_argument('--model-file',      default='random_walk_surrogate.pt')
parser.add_argument('--max-iterations',  type=int,   default=100)

# Strategy selection
parser.add_argument('--strategy', type=str,
                    choices=['greedy', 'ranked', 'similarity'],
                    default='ranked',
                    help='Control strategy: greedy, ranked, similarity')

# Ranked strategy
parser.add_argument('--ranked-k',        type=int,   default=100,
                    help='Number of top actions to sample from (ranked)')
parser.add_argument('--exp-decay-rate',  type=float, default=1.0,
                    help='Exponential decay rate for ranked/similarity')

# Similarity strategy
parser.add_argument('--similarity-k',    type=int,   default=500,
                    help='Number of top actions to sample from (similarity)')
parser.add_argument('--sim-scale',       type=float, default=8.0,
                    help='Cosine similarity scale')
parser.add_argument('--novelty-decay',   type=float, default=0.5,
                    help='Similarity influence decay per step')

args = parser.parse_args()

# --------------------------
# 2) Utilities
# --------------------------
def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def compute_reward(state, d0, te, goal):
    d = np.linalg.norm(state[:2] - goal)
    raw = 2 * ((d0 - d) / 0.2)
    rd = max(raw, 0) if te > math.radians(105) else raw
    tt = math.atan2(goal[1] - state[1], goal[0] - state[0])
    ae = abs(wrap_to_pi(tt - state[4]))
    ra = ((te - ae) / te) if te > 0 else 0
    w = (te / math.pi) * 100
    return rd + w * ra

def get_slip(state, slip_map):
    xi = int(round(state[0] / args.scale_factor))
    yi = int(round(state[1] / args.scale_factor))
    return slip_map[yi, xi]

# --------------------------
# 3) Cosine Similarity Setup
# --------------------------
use_similarity = args.strategy == 'similarity'
fallback_to_ranked = False
X_ref = None

if use_similarity and os.path.exists(args.log_file):
    df_prev = pd.read_csv(args.log_file)
    if len(df_prev) > 0:
        X_ref = df_prev[['theta', 'omega_l', 'omega_r', 'slip']].values.astype(np.float32)
        X_ref /= np.linalg.norm(X_ref, axis=1, keepdims=True)
        print("Cosine similarity enabled.")
    else:
        print("Log file is empty. Falling back to ranked strategy for this iteration.")
        fallback_to_ranked = True
elif use_similarity:
    print("No log file found. Falling back to ranked strategy for this iteration.")
    fallback_to_ranked = True

# --------------------------
# 4) Data Setup
# --------------------------
data = np.load("dem_box_full.npz")
slip_map = np.squeeze(data["slip_coefficients"]) / 100.0
traj = pd.read_csv(args.trajectory_csv)
traj[['X', 'Y']] *= args.scale_factor
goal = traj[['X', 'Y']].values[-1]
goal_x, goal_y = goal

# --------------------------
# 5) Initial State
# --------------------------
if os.path.exists(args.init_state_file):
    state = np.load(args.init_state_file)
else:
    s0, s1 = traj[['X', 'Y']].values[:2]
    t0 = math.atan2(s1[1] - s0[1], s1[0] - s0[0])
    state = np.array([s0[0], s0[1], 0., 0., t0 + math.radians(45), 0.])

# --------------------------
# 6) Load Surrogate Model
# --------------------------
class SurrogateNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 6)
        self.act = nn.Tanh()
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SurrogateNN().to(device)
model.load_state_dict(torch.load(args.model_file, map_location=device))
model.eval()
car = CarDynamics(r=0.09, B=0.25)

# --------------------------
# 7) Control Loop
# --------------------------
dt, sub = 2.0, 10
omegas = np.linspace(0, 2.0, 100)
log_rows = []

for step in range(args.max_iterations):
    d0 = np.linalg.norm(state[:2] - goal)
    if d0 <= args.threshold:
        print("Reached goal.")
        break

    te = abs(wrap_to_pi(math.atan2(goal[1] - state[1], goal[0] - state[0]) - state[4]))
    slip = get_slip(state, slip_map)

    cands = []
    for wl in omegas:
        for wr in omegas:
            if not ((wl == 0 and wr == 0) or (wl <= 3 * wr and wr <= 3 * wl)):
                continue
            vec = np.array([state[2], state[3], wrap_to_pi(state[4]),
                            state[5], wl, wr, slip], dtype=np.float32)
            with torch.no_grad():
                pred = model(torch.from_numpy(vec).to(device)).cpu().numpy()
            npred = state.copy()
            npred[:6] += pred
            reward = compute_reward(npred, d0, te, goal)
            cands.append((wl, wr, pred, reward))

    cands.sort(key=lambda x: x[3], reverse=True)

    # Strategy-Based Action Selection
    if args.strategy == 'greedy':
        wl, wr, best_pred, _ = cands[0]

    elif args.strategy == 'ranked' or (args.strategy == 'similarity' and fallback_to_ranked):
        k = 100 if fallback_to_ranked else args.ranked_k
        topk = cands[:k]
        ranks = np.arange(len(topk), dtype=np.float32)
        weights = np.exp(-args.exp_decay_rate * ranks)
        weights /= weights.sum()
        idx = np.random.choice(len(topk), p=weights)
        wl, wr, best_pred, _ = topk[idx]

    elif args.strategy == 'similarity' and use_similarity:
        topk = cands[:args.similarity_k]
        weights = []
        novelty_weight = np.exp(-args.novelty_decay * step)
        for i, (wl, wr, pred, reward) in enumerate(topk):
            v = np.array([wrap_to_pi(state[4]), wl, wr, slip], dtype=np.float32)
            v_norm = v / np.linalg.norm(v)
            cos_sims = X_ref @ v_norm
            avg_sim = np.mean(cos_sims)
            w_exp = np.exp(-args.exp_decay_rate * i)
            w_sim = np.exp(-args.sim_scale * avg_sim * novelty_weight)
            weights.append(w_exp * w_sim)
        weights = np.array(weights)
        weights /= weights.sum()
        idx = np.random.choice(len(topk), p=weights)
        wl, wr, best_pred, _ = topk[idx]

    else:
        raise ValueError("Invalid strategy or unknown fallback condition.")

    # Apply selected action
    nxt = car.advance(state, dt, sub, slip, slip, wl, wr)
    delta = nxt - state
    retro = float(np.sqrt(((best_pred - delta) ** 2).mean()))
    log_rows.append([*state, slip, wl, wr, retro, goal_x, goal_y])
    state = nxt
else:
    print("Max iterations reached without reaching goal.")

# --------------------------
# 8) Save state and log
# --------------------------
os.makedirs(os.path.dirname(args.state_out_file), exist_ok=True)
np.save(args.state_out_file, state)

# 1) read last row of test_errors.csv (if available)
if os.path.exists(args.error_csv):
    err_df = pd.read_csv(args.error_csv)
    if not err_df.empty:
        last_err = err_df.iloc[-1].to_dict()
    else:
        last_err = {c: float('nan') for c in err_df.columns}
else:
    last_err = {}

# 2) build DataFrame of this run’s new rows
base_cols = [
    'x','y','xdot','ydot','theta','theta_dot',
    'slip','omega_l','omega_r','retroactive_error',
    'x_goal','y_goal'
]
df_new = pd.DataFrame(log_rows, columns=base_cols)
for col, val in last_err.items():
    df_new[col] = val

# 3) merge with existing log (if any) and rewrite
if os.path.exists(args.log_file):
    df_old = pd.read_csv(args.log_file)
    df_full = pd.concat([df_old, df_new], ignore_index=True, sort=False)
else:
    df_full = df_new

df_full.to_csv(args.log_file, index=False)

print(f"Drive complete: steps={len(log_rows)}")
sys.exit(0)

