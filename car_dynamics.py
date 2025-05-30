import numpy as np

class CarDynamics:
    def __init__(self, r: float, B: float):
        """
        :param r: wheel radius [m]
        :param B: track width (distance between left/right wheels) [m]
        """
        self.r = r
        self.B = B

    def advance(self,
                state: np.ndarray,
                dt_total: float,
                num_sub_steps: int,
                slip_l: float,
                slip_r: float,
                omega_l: float,
                omega_r: float
               ) -> np.ndarray:
        """
        Advance the state by dt_total, splitting into num_sub_steps Euler steps.

        :param state:   array_like [x, y, x_dot, y_dot, theta, theta_dot]
        :param dt_total:     total integration time [s]
        :param num_sub_steps: how many Euler sub-steps to take
        :param slip_l:  left-wheel slip fraction (0=no slip, 1=full slip)
        :param slip_r:  right-wheel slip fraction
        :param omega_l: left-wheel angular speed [rad/s]
        :param omega_r: right-wheel angular speed [rad/s]
        :returns: next_state: np.ndarray [x, y, x_dot, y_dot, theta, theta_dot]
        """
        dt = dt_total / num_sub_steps
        x, y, x_dot, y_dot, theta, theta_dot = state

        for _ in range(num_sub_steps):
            # local‐frame kinematics
            v_x   = 0.5 * self.r * ((1 - slip_l) * omega_l + (1 - slip_r) * omega_r)
            v_phi = (self.r / self.B) * ((1 - slip_r) * omega_r - (1 - slip_l) * omega_l)

            # global‐frame velocities
            x_dot     = v_x * np.cos(theta)
            y_dot     = v_x * np.sin(theta)
            theta_dot = v_phi

            # Euler integration
            x     += x_dot * dt
            y     += y_dot * dt
            theta += theta_dot * dt

        return np.array([x, y, x_dot, y_dot, theta, theta_dot])
