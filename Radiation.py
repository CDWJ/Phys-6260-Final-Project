import run_grid_PML as grid
import numpy as np
import matplotlib.pyplot as plt
from hyperparams import *

# Constants
c = sp_light
# q = 1.6e-19  # Charge of the electron (Coulombs)
# m = 9.11e-31  # Mass of the electron (kg)
# epsilon0 = 8.85e-12     # vaccum permittivity
# mu0 = 1.26e-6

# create the grid map
x = np.linspace(-scene_x/2, scene_x/2, res_x)
y = np.linspace(-scene_y/2, scene_y/2, res_y)
x, y = np.meshgrid(x, y)

src_x = 0.
src_y = 0.

# Magnetic field and electric field in 2D

# Hx = grid.Hx.to_numpy()
# Hy = grid.Hy.to_numpy()
Hx = Hx.to_numpy()
Hy = Hy.to_numpy()
Hz = np.zeros(np.shape(Hx))

Bx = Hx*mu0
By = Hy*mu0
Bz = Hz*mu0

Ez = grid.Ez
Ex = np.zeros(Ez.shape)
Ey = np.zeros(Ez.shape)

# number density of electrons
n = np.full(Ez.shape, n0)

avgVel = 0.1*c #system velocity which will probably be set to 0.1 times the speed of light
standDev = 0.01*c
normVelMag = np.random.normal(avgVel, standDev, 1)[0] #magnitude of the velocity pulled from a gaussian distribution with a standard deviation of 0.01
dist = np.sqrt(np.power(x - src_x) + np.power(y - src_y))
theta = np.arctan(np.divide(y - src_y, x - src_x))
vx = np.sin(theta) * normVelMag
vy = np.cos(theta) * normVelMag
vz = np.zeros(vx.shape)      

#return vector of dv/dt from lorentz force and initial velocity
def acceleration(vx, vy, vz, t): 

    ax = (q/m)*(Ex + (vy * Bz) - (vz * By))
    ay = (q/m)*(Ey + (vz * Bx) - (vx * Bz))
    az = (q/m)*(Ez + (vx * By) - (vy * Bx))

    return ax, ay, az


tpoints = np.arange(0., duration, dt)
power_all = np.empty(Ez.shape)


# we are doing 2D computation first 
# if we are looking at xy plane with z = 0 
vx = np.zeros(Ez.shape)
vy = np.zeros(Ez.shape)
vz = np.full(Ez.shape, 0.5*c)


def evo(tpoints, vx, vy, vz):
    for i,t in enumerate(tpoints,1):

        if i != 0:

            # updating velociteis
            k1_x, k1_y, k1_z = dt * acceleration(vx, vy, vz, t)
            k2_x, k2_y, k2_z = dt * acceleration(vx + 0.5*k1, vy + 0.5*k1, vz + 0.5*k1)
            k3_x, k3_y, k3_z = dt * acceleration(vx + 0.5*k2, vy + 0.5*k2, vz + 0.5*k2)
            k4_x, k4_y, k4_z = dt * acceleration(vx + k3, vy + k3, vz + k3)
            vx += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6.0
            vy += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6.0
            vx += (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6.0   
        
        # computing acceleratioin and lorentz factor
        ax, ay, az = acceleration(vx, vy, vz, t)
        gamma = np.power(np.sqrt(1 - (np.square(vx) + np.square(vy) + np.square(vz)) / c**2), -1)


        fluc = 0.5 * n

        for j in np.shape(vx)[0]:
            for k in np.shape(vx)[1]:

                # updating the number density of electrons by tracking their motion
                # since we do only 2D computation, we currently ignore the fluctuation in z direction only track 
                # their changing in x and y direction
                # if their velocity is large enough for them to reach the other grid point, we allow a change of
                # 0.5(this can be set to other fraction) of the orginal number density

                step_x = int( ((vx[j,k]*dt) % dx) / (0.5 * dx) )
                step_y = int( ((vy[j,k]*dt) % dx) / (0.5 * dx) )
                n[j,k] -= fluc
                newj = (j + step_x) % res_x
                newk = (k + step_y) % res_y
                n[newj, newk] += fluc 

                # we only care about the acceleration perpendicular to velocity, as required by 
                # the equation of synchrontron radiation

                v = np.array([vx[j,k], vy[j,k], vz[j,k]])
                a = np.array([ax[j,k], ay[j,k], az[j,k]])
                proj = np.dot(a, v) / np.dot(v, v) * v
                a -= proj

                # computing the radiation power
                power = (n[j,k]*q)**2 / (6*np.pi*epsilon0*c**2) * np.abs(a)**2 * gamma[j,k]**4
                power_all[j,k] = power

    return power_all
        
    

