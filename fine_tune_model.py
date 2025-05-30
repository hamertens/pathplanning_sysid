#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ─── 1) CLI ────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Fine-tune surrogate model and track test errors")
p.add_argument('--iteration',  type=int,   required=True)
p.add_argument('--log-file',   default='output_data/direct_control_log.csv')
p.add_argument('--model-in',   default='random_walk_surrogate.pt')
p.add_argument('--model-out',  default='surrogate_model.pt')
p.add_argument('--test-set',   default='test_set.npz')
p.add_argument('--epochs',     type=int,   default=5)
p.add_argument('--lr',         type=float, default=1e-4)
args = p.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── 2) Network definition ──────────────────────────────────────────────────
class SurrogateNN(nn.Module):
    def __init__(self, input_dim=7, hidden=50, output_dim=6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden,    hidden)
        self.fc3 = nn.Linear(hidden, output_dim)
        self.act = nn.Tanh()
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)

# ─── 3) Compute test-error before fine-tuning ───────────────────────────────
test_data = np.load(args.test_set)
X_test = test_data['X_test'].astype(np.float32)
Y_test = test_data['Y_test'].astype(np.float32)

X_test_t = torch.from_numpy(X_test).to(device)
Y_test_t = torch.from_numpy(Y_test).to(device)

model = SurrogateNN(input_dim=7).to(device)
model.load_state_dict(torch.load(args.model_in, map_location=device))
model.eval()

with torch.no_grad():
    pred_test = model(X_test_t)

    # Breakdown of losses
    mse = nn.MSELoss()
    loss_total = mse(pred_test, Y_test_t).item()
    loss_pos = mse(pred_test[:, 0:2], Y_test_t[:, 0:2]).item()               # dx, dy
    loss_ori = mse(pred_test[:, 4], Y_test_t[:, 4]).item()                   # dtheta
    loss_vel = mse(pred_test[:, 2:4], Y_test_t[:, 2:4]).item() + \
               mse(pred_test[:, 5], Y_test_t[:, 5]).item()                   # ddx, ddy, dtheta_dot

print(f"[Iteration {args.iteration}] Test MSE before fine-tune → "
      f"Total: {loss_total:.6f}, Pos: {loss_pos:.6f}, Ori: {loss_ori:.6f}, Vel: {loss_vel:.6f}")

# ─── 4) Append to CSV ───────────────────────────────────────────────────────
csv_path = 'output_data/test_errors.csv'
header = ['iteration','test_error_total','test_error_pos','test_error_ori','test_error_vel']
write_header = not os.path.exists(csv_path)
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

with open(csv_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    if write_header:
        writer.writeheader()
    writer.writerow({
        'iteration': args.iteration,
        'test_error_total': loss_total,
        'test_error_pos': loss_pos,
        'test_error_ori': loss_ori,
        'test_error_vel': loss_vel,
    })

# ─── 5) Load and prepare fine-tune dataset ──────────────────────────────────
df = pd.read_csv(args.log_file)
df_cur  = df.iloc[:-1].reset_index(drop=True)
df_next = df.iloc[1:].reset_index(drop=True)

features = pd.DataFrame({
    'xdot':      df_cur['xdot'],
    'ydot':      df_cur['ydot'],
    'theta':     df_cur['theta'],
    'theta_dot': df_cur['theta_dot'],
    'omega_l':   df_cur['omega_l'],
    'omega_r':   df_cur['omega_r'],
    'slip':      df_cur['slip'],
})
labels = pd.DataFrame({
    'dx':    df_next['x']    - df_cur['x'],
    'dy':    df_next['y']    - df_cur['y'],
    'ddx':   df_next['xdot'] - df_cur['xdot'],
    'ddy':   df_next['ydot'] - df_cur['ydot'],
    'dtheta':     df_next['theta']     - df_cur['theta'],
    'dtheta_dot': df_next['theta_dot'] - df_cur['theta_dot'],
})

train_df = pd.concat([features, labels], axis=1)
before = len(train_df)
train_df = train_df.drop_duplicates()
after = len(train_df)
print(f"Removed {before-after} duplicate samples from fine-tune dataset.")

X = train_df[['xdot','ydot','theta','theta_dot','omega_l','omega_r','slip']].values.astype(np.float32)
Y = train_df[['dx','dy','ddx','ddy','dtheta','dtheta_dot']].values.astype(np.float32)

X_t = torch.from_numpy(X).to(device)
Y_t = torch.from_numpy(Y).to(device)

# ─── 6) Fine-tune ───────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
model.train()
for epoch in range(1, args.epochs+1):
    optimizer.zero_grad()
    pred = model(X_t)
    loss = mse(pred, Y_t)
    loss.backward()
    optimizer.step()
    print(f"[Iteration {args.iteration}] Epoch {epoch}/{args.epochs} — finetune MSE {loss.item():.6f}")

# ─── 7) Save model ──────────────────────────────────────────────────────────
torch.save(model.state_dict(), args.model_out)
print(f"✔ Saved fine-tuned model to {args.model_out}")






