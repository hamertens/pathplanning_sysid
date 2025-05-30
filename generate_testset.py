#!/usr/bin/env python3
import numpy as np
import math
import random
from car_dynamics import CarDynamics

# -------------------------------
# 0. Helpers
# -------------------------------
MAX_V = 0.17
def sample_vel_and_heading(max_v=MAX_V):
    # pick a random heading
    theta = random.uniform(-math.pi, math.pi)
    # pick a speed with uniform area‐density in disk
    r = max_v * math.sqrt(random.random())
    x_dot = r * math.cos(theta)
    y_dot = r * math.sin(theta)
    return x_dot, y_dot, theta

# -------------------------------
# 1. Load friction map
# -------------------------------
data = np.load("dem_box_full.npz")
slip_values = np.squeeze(data["slip_coefficients"]) / 100.0
slip_min, slip_max = slip_values.min(), slip_values.max()
print("Friction range:", slip_min, slip_max)

# -------------------------------
# 2. Params & storage
# -------------------------------
dt_total      = 2.0
num_sub_steps = 10
num_samples   = 300_000

X_test = np.zeros((num_samples, 7), dtype=np.float32)
Y_test = np.zeros((num_samples, 6), dtype=np.float32)

car_model = CarDynamics(r=0.09, B=0.25)

# -------------------------------
# 3. Generate
# -------------------------------
for i in range(num_samples):
    # sample a consistent velocity & heading
    x_dot, y_dot, theta = sample_vel_and_heading()
    theta_dot = random.uniform(-0.46, 0.46)

    # sample controls & slip
    omega_l = random.uniform(0, 2)
    omega_r = random.uniform(0, 2)
    slip    = random.uniform(slip_min, slip_max)

    state = np.array([0.0, 0.0, x_dot, y_dot, theta, theta_dot], dtype=np.float32)
    final = car_model.advance(state, dt_total, num_sub_steps,
                              slip, slip, omega_l, omega_r)

    # compute deltas
    delta = final - state
    delta_x, delta_y, delta_xdot, delta_ydot, delta_theta, delta_thetadot = delta

    # store
    X_test[i] = [x_dot, y_dot, theta, theta_dot, omega_l, omega_r, slip]
    Y_test[i] = [delta_x, delta_y, delta_xdot, delta_ydot, delta_theta, delta_thetadot]

    if (i+1) % 50000 == 0:
        print(f"{i+1} samples generated")

print("Done:", X_test.shape, Y_test.shape)

# -------------------------------
# 4. Save
# -------------------------------
np.savez("test_set.npz", X_test=X_test, Y_test=Y_test)
print("Saved test_set.npz")


