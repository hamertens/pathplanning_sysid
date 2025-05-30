#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from car_dynamics import CarDynamics

def wrap_to_pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def main():
    # ─── Parameters ─────────────────────────────────────────────────────────
    SCALE         = 5.0          # grid → meters
    DT_TOTAL      = 2.0
    NUM_SUBSTEPS  = 10
    NUM_WALKS     = 2
    WALK_LENGTH   = 20
    EPOCHS        = 100
    LR            = 1e-3

    # build allowed action list
    omega_ls = np.linspace(0, 2.0, 100)
    omega_rs = np.linspace(0, 2.0, 100)
    action_space = [
        (wl, wr)
        for wl in omega_ls for wr in omega_rs
        if (wl == 0 and wr == 0) or (wl <= 3*wr and wr <= 3*wl)
    ]

    # load slip map
    data        = np.load("dem_box_full.npz")
    slip_map    = np.squeeze(data["slip_coefficients"]) / 100.0
    grid_size   = slip_map.shape[0]

    # instantiate true dynamics
    car_model = CarDynamics(r=0.09, B=0.25)

    # start real-world coords for grid (20,20)
    start_grid = (20, 20)
    start_real = np.array([start_grid[0] * SCALE,
                           start_grid[1] * SCALE], dtype=float)

    # collect (X, Y)
    X_data = []
    Y_data = []

    for walk in range(NUM_WALKS):
        # initial state: [x, y, xdot, ydot, theta, theta_dot]
        state = np.array([*start_real, 0.0, 0.0, 0.0, 0.0], dtype=float)

        for _ in range(WALK_LENGTH):
            # 1) compute slip at current cell
            xi = int(round(state[0] / SCALE))
            yi = int(round(state[1] / SCALE))
            slip = slip_map[yi, xi]

            # 2) pick a random valid action
            wl, wr = action_space[np.random.randint(len(action_space))]

            # 3) build feature vector
            feat = np.array([
                state[2],                # xdot
                state[3],                # ydot
                wrap_to_pi(state[4]),    # theta
                state[5],                # theta_dot
                wl, wr, slip
            ], dtype=np.float32)

            # 4) step true dynamics
            nxt = car_model.advance(
                state, DT_TOTAL, NUM_SUBSTEPS,
                slip, slip, wl, wr
            )

            # 5) compute Δ-state label
            delta = (nxt[:6] - state[:6]).astype(np.float32)

            # 6) log
            X_data.append(feat)
            Y_data.append(delta)

            # 7) advance
            state = nxt

    X = np.stack(X_data)   # shape (2*20, 7)
    Y = np.stack(Y_data)   # shape (2*20, 6)
    print(f"Collected {X.shape[0]} samples for training.")

    # ─── Define & train SurrogateNN ─────────────────────────────────────────
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SurrogateNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y).to(device)

    for epoch in range(1, EPOCHS+1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, Y_t)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS} — loss {loss.item():.6f}")

    # ─── Save the trained weights ────────────────────────────────────────────
    torch.save(model.state_dict(), "random_walk_surrogate.pt")
    print("✅ Saved trained surrogate to random_walk_surrogate.pt")

if __name__ == "__main__":
    main()
