from hyperparams import *
from taichi_utils import *
# from mgpcg import *
from init_conditions import *
from io_utils import *
import sys
import shutil
from math import pi
import time
import numpy as np
ti.init(arch=ti.cuda, device_memory_GB=4.0, debug=False)

def write_image(img_xy, outdir, i):
    img = np.flip(img_xy.transpose([1,0,2]), 0)
    # take the predicted c map
    img8b = to8b(img)
    save_filepath = os.path.join(outdir, '{:04d}.png'.format(i))
    imageio.imwrite(save_filepath, img8b)
    
def write_field(img, outdir, i, cell_type=None, vmin=0, vmax=1, dpi=512//8):
    array = img[:,:,np.newaxis]
    scale_x = array.shape[0]
    scale_y = array.shape[1]
    array = np.transpose(array, (1, 0, 2)) # from X,Y to Y,X
    x_to_y = array.shape[1]/array.shape[0]
    y_size = 7
    fig = plt.figure(num=1, figsize=(x_to_y * y_size + 1, y_size), clear=True)
    ax = fig.add_subplot()
    ax.set_xlim([0, array.shape[1]])
    ax.set_ylim([0, array.shape[0]])
    cmap = 'jet'
    p = ax.imshow(array, alpha = 0.4, cmap=cmap, vmin =vmin, vmax =vmax)
    fig.colorbar(p, fraction=0.046, pad=0.04)
    
    plt.text(0.87 * scale_x, 0.87 * scale_y, str(i), fontsize = 20, color = "red")
    fig.savefig(os.path.join(outdir, '{:04d}.png'.format(i)), dpi = dpi)
    plt.close()

dx = scene_y / res_y
inv_dx = 1.0 / dx
half_dx = 0.5 * dx
upper_boundary = 1 - half_dx
lower_boundary = half_dx
right_boundary = res_x * dx - half_dx
left_boundary = half_dx
NPML=[40, 40, 40, 40]
Nx2 = 2*res_x
Ny2 = 2*res_y

def choose_dt():
    cell_size = scene_y / res_y
    ep_mu = ep * mu
    cs = (cell_size) ** 2
    ref_idx = np.sqrt(ep_mu)
    dt = ref_idx * (1 / np.sqrt((1/cell_size**2) * 2))
    cour_num = dt / cell_size
    dt /= sp_light
    return dt, cell_size, cour_num


@ti.func
def sphere_contains(loc, center_x, center_y, radius):
    px = loc[0]
    py = loc[1]
    result = False
    if (px - center_x)**2 + (py - center_y)**2 <= radius**2:
        result = True
    return result


dt, cell_size, cour_num = choose_dt()
dt = 1e-12
total_steps = int(np.ceil(duration / dt))

# boundary_mask = ti.field(ti.i32, shape=(res_y, res_x))
# res_x_in_use = res_x
# res_y_in_use = res_y
Ez_coord = ti.Vector.field(2, float, shape=(res_x, res_y))
Hy_coord = ti.Vector.field(2, float, shape=(res_x, res_y))
Hx_coord = ti.Vector.field(2, float, shape=(res_x, res_y))
center_coords_func(Ez_coord, dx)
horizontal_coords_func(Hx_coord, dx)
vertical_coords_func(Hy_coord, dx)

total_t = ti.field(float, shape=())

Ez = ti.field(float, shape=(res_x, res_y))
Hy = ti.field(float, shape=(res_x, res_y))
Hx = ti.field(float, shape=(res_x, res_y))

sigx = np.zeros([res_x * 2, res_y * 2])
sigy = np.zeros([res_x * 2, res_y * 2])


for nx in range(1, 2 * NPML[0] + 1):
    nx1 = 2 * NPML[0] - nx + 1
    sigx[nx1 - 1,:] = (0.5 * ep / dt) * (nx / 2 / NPML[0])**3

for nx in range(1, 2 * NPML[1] + 1):
    nx1 = Nx2 - 2 * NPML[1] + nx
    sigx[nx1 - 1, :] = (0.5 * ep / dt) * (nx / 2 / NPML[1])**3

for ny in range(1, 2 * NPML[2] + 1):
    ny1 = 2 * NPML[2] - ny + 1
    sigy[:, ny1 - 1] = (0.5 * ep / dt) * (ny / 2 / NPML[2])**3

for ny in range(1, 2 * NPML[3] + 1):
    ny1 = Ny2 - 2 * NPML[3] + ny
    sigy[:, ny1 - 1] = (0.5 * ep / dt) * (ny / 2 / NPML[3])**3

    
cHx1 = ti.field(float, shape=(res_x, res_y))
cHx2 = ti.field(float, shape=(res_x, res_y))
cHx3 = ti.field(float, shape=(res_x, res_y))

cHy1 = ti.field(float, shape=(res_x, res_y))
cHy2 = ti.field(float, shape=(res_x, res_y))
cHy3 = ti.field(float, shape=(res_x, res_y))

cDz1 = ti.field(float, shape=(res_x, res_y))
cDz2 = ti.field(float, shape=(res_x, res_y))
cDz4 = ti.field(float, shape=(res_x, res_y))

def setup_coeff():
    URxx = 1.
    URyy = 1.
    URzz = 1.
    sigHx = sigx[1::2, ::2]
    sigHy = sigy[1::2, ::2]
    mHx0  = (1 / dt) + sigHy / (2 * ep)
    mHx1 = ((1 / dt) - sigHy / (2 * ep)) / mHx0
    mHx2 = -sp_light / URxx / mHx0
    mHx3 = -(sp_light * dt / ep)* ((sigHx / URxx) / mHx0)
    
    cHx1.from_numpy(mHx1)
    cHx2.from_numpy(mHx2)
    cHx3.from_numpy(mHx3)

    sigHx = sigx[::2, 1:Ny2:2]
    sigHy = sigy[::2, 1:Ny2:2]
    mHy0  = (1 / dt) + sigHx / (2 * ep)
    mHy1 = ((1 / dt) - sigHx / (2 * ep)) / mHy0
    mHy2 = -sp_light / URyy / mHy0
    mHy3 = -(sp_light * dt / ep) * ((sigHy / URyy) / mHy0)
    
    cHy1.from_numpy(mHy1)
    cHy2.from_numpy(mHy2)
    cHy3.from_numpy(mHy3)

    sigDx = sigx[0:Nx2-1:2, 0:Ny2-1:2]
    sigDy = sigy[0:Nx2-1:2, 0:Ny2-1:2]
    mDz0  = (1/dt) + (sigDx + sigDy) / (2 * ep) + (sigDx * sigDy) * (dt / 4 / ep**2)
    mDz1 = (1/dt) - (sigDx + sigDy)/(2*ep) - (sigDx * sigDy) * (dt / 4 / ep**2)
    mDz1  = mDz1 / mDz0
    mDz2  = sp_light / mDz0;
    mDz4 = - (dt / ep**2) * (sigDx * sigDy) / mDz0
    
    cDz1.from_numpy(mDz1)
    cDz2.from_numpy(mDz2)
    cDz4.from_numpy(mDz4)
    
    

setup_coeff()

CEx = ti.field(float, shape=(res_x, res_y))
ICEx = ti.field(float, shape=(res_x, res_y))
CEy = ti.field(float, shape=(res_x, res_y))
ICEy = ti.field(float, shape=(res_x, res_y))
Hx = ti.field(float, shape=(res_x, res_y))
Hy = ti.field(float, shape=(res_x, res_y))
CHz = ti.field(float, shape=(res_x, res_y))
Dz = ti.field(float, shape=(res_x, res_y))
IDz = ti.field(float, shape=(res_x, res_y))
Ez = ti.field(float, shape=(res_x, res_y))
P = ti.field(float, shape=(res_x, res_y))

@ti.kernel
def calculate_radiation():
    for i in range(0, res_x - 1):
        for j in range(0, res_y - 1):
            Ea = ti.Vector([0, 0, Ez[i, j]])
            Hb = ti.Vector([(Hx[i, j] + Hx[i + 1, j]) * 0.5, (Hy[i, j] + Hy[i, j + 1]) * 0.5, 0.0])
            P[i, j] = ti.math.length(ti.math.cross(Ea, Hb))


@ti.kernel
def update_CE_ICE():
    #Calculate CEx
    for i in range(0, res_x):
        for j in range(0, res_y - 1):
            CEx[i, j] = (Ez[i, j + 1] - Ez[i, j]) / dx
        CEx[i, res_y - 1] = (0 - Ez[i, res_y - 1]) / dx
    
    for j in range(0, res_y):
        for i in range(0, res_x - 1):
            CEy[i, j] = -(Ez[i + 1, j] - Ez[i, j]) / dx
        CEy[res_x - 1, j] = -(0 - Ez[res_x - 1, j]) / dx
        
        
    for I in ti.grouped(ICEx):
        ICEx[I] += CEx[I]
    for I in ti.grouped(ICEy):
        ICEy[I] += CEy[I]

        
@ti.kernel
def update_H():
    for I in ti.grouped(Hx):
        Hx[I] = cHx1[I] * Hx[I] + cHx2[I] * CEx[I] + cHx3[I] * ICEx[I]
    for I in ti.grouped(Hy):
        Hy[I] = cHy1[I] * Hy[I] + cHy2[I] * CEy[I] + cHy3[I] * ICEy[I]
        
@ti.kernel
def update_CHz():
    CHz[0, 0]=(Hy[0, 0] - 0.0) / dx - (Hx[0, 0] - 0.0) / dx
    
    for i in range(1, res_x):
        CHz[i, 1] = (Hy[i, 1] - Hy[i - 1 , 1]) / dx - (Hx[i, 1] - 0.0) / dx
        
    for j in range(1, res_y):
        CHz[1, j] = (Hy[1, j] - 0.0) / dx - (Hx[1, j] - Hx[1, j - 1]) / dx
    
    for i in range(1, res_x - 1):
        for j in range(1, res_y):
            CHz[i, j] = (Hy[i, j] - Hy[i - 1, j]) / dx - (Hx[i, j] - Hx[i, j - 1]) / dx

@ti.kernel
def update_Ez():
    for I in ti.grouped(IDz):
        IDz[I] += Dz[I]
    
    for I in ti.grouped(Dz):
        Dz[I] = cDz1[I] * Dz[I] + cDz2[I] * CHz[I] + cDz4[I] * IDz[I]
        
        
# @ti.kernel
# def add_source(amp:float, mu:float, sigma:float, prime:ti.template(), time:float, position_x:float, position_y:float):
#     denom_ = -0.5 / (sigma * sigma)
#     for i, j in prime:
#         if i == position_x and j == position_y:
#             prime[i, j] += ti.exp(-(((time-t0)/tau)**2))


@ti.kernel
def add_source(amp:float, mu:float, sigma:float, prime:ti.template(), time:float, position_x:float, position_y:float):       # in cartesian coordinates
    omega = pi/3             # angular speed for the spin of neutron star
    # R = 1e-2
    R = 11e3 * 1e-5
    # I = 1
    I = 1.056e23 
    m_mag = I * pi*R**2      # magnitude of magnetic dipole moment of the current loop (representing the neutron star)
    phi = omega*time*1e10               # phi angle in a spherical coordinates
    theta = pi/6             # theta angle in a spherical coordinates, also the inclination between m and the spin axis
    # magnetic dipole moment in cartesian coordinates
    m = ti.Vector([m_mag*ti.sin(theta)*ti.cos(phi), m_mag*ti.sin(theta)*ti.sin(phi), m_mag*ti.cos(theta)]) 
    mu_0 = 8.8541878176e-6
    for i, j in Hx:
        # if i == position_x and j == position_y:
        if i >= NPML[0] and i <= res_x - NPML[1] - 1 and j >= NPML[2] and j <= res_y - NPML[3]:
            theta = pi/6 
            m = ti.Vector([m_mag*ti.sin(theta)*ti.cos(phi), m_mag*ti.sin(theta)*ti.sin(phi), m_mag*ti.cos(theta)]) 
            r = ti.Vector([(i - 0.5) * dx - 0.5, (j) * dx - 0.4, 0.0])
            r_mag = magnitude = ti.math.length(r)
            r_hat = r/r_mag
            Hx[i, j] += (mu_0/4/pi * (3*r_hat*ti.math.dot(r_hat, m) - m)/r_mag**3)[0]
            # Hx[i, j] += ti.exp(-(((time-t0)/tau)**2))
        if i >= NPML[0] and i <= res_x - NPML[1] - 1 and j >= NPML[2] and j <= res_y - NPML[3]:
            theta = -pi/6 
            m = ti.Vector([m_mag*ti.sin(theta)*ti.cos(phi), m_mag*ti.sin(theta)*ti.sin(phi), m_mag*ti.cos(theta)]) 
            r = ti.Vector([(i - 0.5) * dx - 0.5, (j) * dx - 0.6, 0.0])
            r_mag = magnitude = ti.math.length(r)
            r_hat = r/r_mag
            Hx[i, j] += (mu_0/4/pi * (3*r_hat*ti.math.dot(r_hat, m) - m)/r_mag**3)[0]
            # Hx[i, j] += ti.exp(-(((time-t0)/tau)**2))

    for i, j in Hy:
        if i >= NPML[0] and i <= res_x - NPML[1] and j >= NPML[2] and j <= res_y - NPML[3] - 1:
            theta = pi/6 
            m = ti.Vector([m_mag*ti.sin(theta)*ti.cos(phi), m_mag*ti.sin(theta)*ti.sin(phi), m_mag*ti.cos(theta)]) 

            r = ti.Vector([i * dx - 0.5, (j - 0.5) * dx - 0.4, 0.0])
            r_mag = magnitude = ti.math.length(r)
            r_hat = r/r_mag
            Hy[i, j] += (mu_0/4/pi * (3*r_hat*ti.math.dot(r_hat, m) - m)/r_mag**3)[1]
            # Hy[i, j] += ti.exp(-(((time-t0)/tau)**2))

        if i >= NPML[0] and i <= res_x - NPML[1] and j >= NPML[2] and j <= res_y - NPML[3] - 1:
            theta = -pi/6 
            m = ti.Vector([m_mag*ti.sin(theta)*ti.cos(phi), m_mag*ti.sin(theta)*ti.sin(phi), m_mag*ti.cos(theta)]) 

            
            r = ti.Vector([i * dx - 0.5, (j - 0.5) * dx - 0.6, 0.0])
            r_mag = magnitude = ti.math.length(r)
            r_hat = r/r_mag
            Hy[i, j] += (mu_0/4/pi * (3*r_hat*ti.math.dot(r_hat, m) - m)/r_mag**3)[1]
            # Hy[i, j] += ti.exp(-(((time-t0)/tau)**2))


# @ti.kernel
# def add_source(amp:float, mu:float, sigma:float, prime:ti.template(), time:float, position_x:float, position_y:float):       # in cartesian coordinates
#     omega = pi/3             # angular speed for the spin of neutron star
#     R = 1e-2
#     I = 1
#     m_mag = I * pi*R**2      # magnitude of magnetic dipole moment of the current loop (representing the neutron star)
#     phi = omega*time*1e10               # phi angle in a spherical coordinates
#     theta = pi/6             # theta angle in a spherical coordinates, also the inclination between m and the spin axis
#     # magnetic dipole moment in cartesian coordinates
#     m = ti.Vector([m_mag*ti.sin(theta)*ti.cos(phi), m_mag*ti.sin(theta)*ti.sin(phi), m_mag*ti.cos(theta)]) 
#     mu_0 = 8.8541878176e-6
#     for i, j in Hx:
#         # if i == position_x and j == position_y:
#         if i >= NPML[0] and i <= res_x - NPML[1] - 1 and j >= NPML[2] and j <= res_y - NPML[3]:
#             m = ti.Vector([m_mag*ti.sin(theta)*ti.cos(phi), m_mag*ti.sin(theta)*ti.sin(phi), m_mag*ti.cos(theta)]) 
#             r = ti.Vector([(i - 0.5) * dx - 0.5, (j) * dx - 0.5, 0.0])
#             r_mag = magnitude = ti.math.length(r)
#             r_hat = r/r_mag
#             Hx[i, j] += (mu_0/4/pi * (3*r_hat*ti.math.dot(r_hat, m) - m)/r_mag**3)[1]
#             # Hx[i, j] += ti.exp(-(((time-t0)/tau)**2))

#     for i, j in Hy:
#         if i >= NPML[0] and i <= res_x - NPML[1] and j >= NPML[2] and j <= res_y - NPML[3] - 1:
#             r = ti.Vector([i * dx - 0.5, (j - 0.5) * dx - 0.5, 0.0])
#             r_mag = magnitude = ti.math.length(r)
#             r_hat = r/r_mag
#             Hy[i, j] += (mu_0/4/pi * (3*r_hat*ti.math.dot(r_hat, m) - m)/r_mag**3)[1]


# main function
def main(testing=False):
    # create some folders
    logsdir = os.path.join('logs', exp_name)
    os.makedirs(logsdir, exist_ok=True)
    remove_everything_in(logsdir)

    fielddir = 'field'
    fielddir = os.path.join(logsdir, fielddir)
    os.makedirs(fielddir, exist_ok=True)

    fielddir2 = 'field2'
    fielddir2 = os.path.join(logsdir, fielddir2)
    os.makedirs(fielddir2, exist_ok=True)

    ckptdir = 'ckpts'
    ckptdir = os.path.join(logsdir, ckptdir)
    os.makedirs(ckptdir, exist_ok=True)

    shutil.copyfile('/Users/noahzhang/OneDrive - Georgia Institute of Technology/Comp Phy/Phys-6260-Final-Project/hyperparams.py', f'{logsdir}/hyperparams.py')
    
    write_field(Ez.to_numpy(), fielddir, 0, vmin=0, vmax=src_amp/20)
    
    from_frame = 0
    if save_ckpt:
        np.save(os.path.join(ckptdir, "Ez_numpy_" + str(from_frame)), Ez.to_numpy())
        np.save(os.path.join(ckptdir, "Hy_numpy_" + str(from_frame)), Hy.to_numpy())
        np.save(os.path.join(ckptdir, "Hx_numpy_" + str(from_frame)), Hx.to_numpy())
        # np.save(os.path.join(ckptdir, "w_numpy_" + str(from_frame)), w.to_numpy())
     
    i = -1
    saved = 1
    for frame_idx in range(total_steps):
        i += 1
        j = i % save_per_frame
        print("[Simulate] Running step: ", i, " / substep: ", j)
        
        update_CE_ICE()
        
        update_H()

        add_source(src_amp, src_mu, src_sigma, Dz, i * dt, src_x, src_y)
        
        update_CHz()
        
        update_Ez()
        
        Ez.copy_from(Dz)

        calculate_radiation()
        
        if frame_idx % save_per_frame == 0:
            np.save(os.path.join(ckptdir, "Ez_numpy_" + str(frame_idx)), Ez.to_numpy())
            np.save(os.path.join(ckptdir, "Hy_numpy_" + str(frame_idx)), Hy.to_numpy())
            np.save(os.path.join(ckptdir, "Hx_numpy_" + str(frame_idx)), Hx.to_numpy())
            write_field(Hx.to_numpy(), fielddir, saved, vmin=-0.008, vmax=0.008)
            write_field(Hy.to_numpy(), fielddir2, saved, vmin=-0.008, vmax=0.008)
            
            

            # write_field(Ez.to_numpy(), fielddir, saved, vmin=-0.05, vmax=0.05)
            saved += 1
            
            
if __name__ == '__main__':
    print("[Main] Begin")
    main()
    print("[Main] Complete")

