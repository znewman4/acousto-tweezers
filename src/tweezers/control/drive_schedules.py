import numpy as np

class MovingGaussianSchedule:
    def __init__(self, x0_start, y0_start, vx_mps, vy_mps, sigma_m, phase=0.0):
        self.x0_start = x0_start
        self.y0_start = y0_start
        self.vx = vx_mps
        self.vy = vy_mps
        self.sigma = sigma_m
        self.phase = phase

    def p_bot_unit(self, grid, t_s):
        x0 = self.x0_start + self.vx * t_s
        y0 = self.y0_start + self.vy * t_s
        X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
        gauss = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * self.sigma**2))
        return gauss * np.exp(1j * self.phase)
