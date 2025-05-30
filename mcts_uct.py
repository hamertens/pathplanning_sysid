#!/usr/bin/env python3
import os
import csv
import numpy as np
import pandas as pd
import random
import math
import argparse
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

warnings.filterwarnings("ignore", category=UserWarning)

# 1) CLI
p = argparse.ArgumentParser(description='Single-step UCT planner (always from actual state)')
p.add_argument('--uct_horizon',        type=int,   default=15,   help='Rollout horizon')
p.add_argument('--uct_simulations',    type=int,   default=2000, help='Number of simulations')
p.add_argument('--known-indices-file', default='output_data/known_indices.npy',
               help='Path to known-indices .npy')
p.add_argument('--trajectory-csv',     default='output_data/trajectory.csv',
               help='Path to trajectory CSV')
p.add_argument('--current-state-file', default='output_data/current_state.npy',
               help='True state .npy')
p.add_argument('--scale-factor',       type=float, default=5.0,
               help='Grid→real scaling')
args = p.parse_args()

# 2) Load terrain + slip
data = np.load("dem_box_full.npz")
slope_degrees     = data["slope_degrees"]
slip_coefficients = data["slip_coefficients"]
grid_size = slope_degrees.shape[0]

# 3) Movement helpers
actions = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
def get_allowed_actions(state, gs, prev=None):
    r, c = state
    allowed = []
    for dr, dc in actions:
        nr, nc = r+dr, c+dc
        if 0 <= nr < gs and 0 <= nc < gs:
            allowed.append((dr,dc))
    if prev is not None:
        forb = (prev[0]-r, prev[1]-c)
        allowed = [a for a in allowed if a != forb]
    return allowed

# 4) UCT core
class Node:
    def __init__(self, state, visited, parent=None, prev_state=None):
        self.state = state
        self.visited = set(visited)
        self.parent = parent
        self.prev_state = prev_state
        self.untried_actions = get_allowed_actions(state, grid_size, prev=prev_state)
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0

def best_child(node, c=1.41):
    best, best_val = None, -1e9
    for ch in node.children.values():
        exploit = ch.total_reward / ch.visits
        explore = c * math.sqrt(2 * math.log(node.visits) / ch.visits)
        val = exploit + explore
        if val > best_val:
            best_val, best = val, ch
    return best

def expand(node):
    a = node.untried_actions.pop()
    nr, nc = node.state[0] + a[0], node.state[1] + a[1]
    idx = nr * grid_size + nc
    vis2 = node.visited.copy(); vis2.add(idx)
    child = Node((nr,nc), vis2, parent=node, prev_state=node.state)
    node.children[a] = child
    return child

def rollout(st, vis, prev, horizon, gamma, unc):
    total, disc = 0.0, 1.0
    cur, pr = st, prev
    for _ in range(horizon):
        acts = get_allowed_actions(cur, grid_size, prev=pr)
        if not acts:
            break
        a = random.choice(acts)
        nr, nc = cur[0] + a[0], cur[1] + a[1]
        idx = nr * grid_size + nc
        total += disc * unc[idx]
        disc *= gamma
        vis.add(idx)
        pr, cur = cur, (nr, nc)
    return total

def backup(node, reward):
    while node:
        node.visits += 1
        node.total_reward += reward
        node = node.parent

def uct_search(root_state, known, unc, horizon, sims, gamma=1.0, prev_state=None):
    root = Node(root_state, known, parent=None, prev_state=prev_state)
    for _ in range(sims):
        node = root
        # selection
        while not node.untried_actions and node.children:
            node = best_child(node)
        # expansion
        if node.untried_actions:
            node = expand(node)
        # simulation
        r = rollout(node.state, node.visited.copy(), node.prev_state, horizon, gamma, unc)
        # backprop
        backup(node, r)
    # pick best action
    best_a, best_v = None, -1e9
    for a, ch in root.children.items():
        val = ch.total_reward / ch.visits
        if val > best_v:
            best_v, best_a = val, a
    return best_a

# 5) Always root at your true location (fallback to 20,20 if missing)
if os.path.exists(args.known_indices_file) and os.path.exists(args.trajectory_csv):
    known   = np.load(args.known_indices_file).tolist()
    traj_df = pd.read_csv(args.trajectory_csv)
    traj_list = list(zip(traj_df.X.astype(int), traj_df.Y.astype(int)))
else:
    known, traj_list = [], []

if os.path.exists(args.current_state_file):
    st = np.load(args.current_state_file)
    row = int(round(st[1] / args.scale_factor))
    col = int(round(st[0] / args.scale_factor))
else:
    row, col = 20, 20

idx = row * grid_size + col
if idx not in known:
    known.append(idx)
    traj_list.append((col, row))

current_state = (row, col)
prev_state = None

# 6) Fit GP and get uncertainties
Xtr = np.array([slope_degrees[i//grid_size, i%grid_size] for i in known]).reshape(-1,1)
ytr = np.array([slip_coefficients[i//grid_size, i%grid_size] for i in known])
kernel = C(1.0, (1e-3,1e3)) * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(Xtr, ytr)

# 6a) Compute test‐error on unsampled cells
all_idxs  = np.arange(grid_size * grid_size)
unsampled = np.setdiff1d(all_idxs, np.array(known))
if unsampled.size > 0:
    X_test = slope_degrees.flatten()[unsampled].reshape(-1,1)
    y_true = slip_coefficients.flatten()[unsampled]
    y_pred = gp.predict(X_test)
    rmse   = math.sqrt(np.mean((y_pred - y_true)**2))
else:
    rmse = float('nan')

# Append RMSE to environment_error.csv
out_path = 'output_data/environment_error.csv'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
write_header = not os.path.exists(out_path)
with open(out_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['test_error'])
    if write_header:
        writer.writeheader()
    writer.writerow({'test_error': rmse})

print(f"GP test RMSE on {unsampled.size} unsampled cells: {rmse:.6f}")

# 7) Predict uncertainties for UCT
uncs = gp.predict(slope_degrees.flatten().reshape(-1,1), return_std=True)[1]

# 8) Single UCT step
move = uct_search(current_state, set(known), uncs,
                  args.uct_horizon, args.uct_simulations, 1.0, prev_state)
nr, nc = current_state[0] + move[0], current_state[1] + move[1]
new_idx = nr * grid_size + nc
if new_idx not in known:
    known.append(new_idx)
    
# append the new waypoint to the trajectory list
traj_list.append((nc, nr))

# 9) Persist back out
os.makedirs(os.path.dirname(args.known_indices_file), exist_ok=True)
np.save(args.known_indices_file, np.array(known))
pd.DataFrame(traj_list, columns=["X","Y"]) \
  .to_csv(args.trajectory_csv, index=False)

print(f"Next waypoint: (X={nc}, Y={nr})")





