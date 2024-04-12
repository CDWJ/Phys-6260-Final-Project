import run_grid_PML as grid
import numpy as np

# Magnetic field and electric field in 2D

Hx = grid.Hx
Hy = grid.Hy
Hz = np.array(Hx.shape)

Ez = grid.Ez
Ex = np.array(Ez.shape)
Ey = np.array(Ez.shape)