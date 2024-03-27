from hyperparams import *
from taichi_utils import *
from init_conditions import *
from io_utils import *
import sys
import shutil
import time
import numpy as np
ti.init(arch=ti.gpu, device_memory_GB=4.0, debug=False)

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

dx = 1.0 / res_y
half_dx = 0.5 * dx
upper_boundary = 1 - half_dx
lower_boundary = half_dx
right_boundary = res_x * dx - half_dx
left_boundary = half_dx

def choose_dt():
    cell_size = scene_y / res_y
    ep_mu = ep * mu
    cs = (cell_size) ** 2
    ref_idx = np.sqrt(ep_mu)
    dt = ref_idx * (1 / np.sqrt((1/cell_size**2) * 2))
    cour_num = dt / cell_size
    dt /= sp_light
    return dt, cell_size, cour_num

dt, cell_size, cour_num = choose_dt()
total_steps = int(np.ceil(duration / dt))

# boundary_mask = ti.field(ti.i32, shape=(res_y, res_x))

Ez_coord = ti.Vector.field(2, float, shape=(res_y, res_x))
Hy_coord = ti.Vector.field(2, float, shape=(res_y, res_x + 1))
Hx_coord = ti.Vector.field(2, float, shape=(res_y + 1, res_x))
center_coords_func(Ez_coord, dx)
horizontal_coords_func(Hx_coord, dx)
vertical_coords_func(Hy_coord, dx)

total_t = ti.field(float, shape=())

Ez = ti.field(float, shape=(res_y, res_x))
Hy = ti.field(float, shape=(res_y, res_x + 1))
Hx = ti.field(float, shape=(res_y + 1, res_x))

chx = ti.field(float, (res_y - 1, res_x))
chx.fill(mu)
chy =  ti.field(float, (res_y, res_x - 1))
chy.fill(mu)
ce1 =  ti.field(float, (res_y, res_x))
ce1.fill(ep)
ce2 =  ti.field(float, (res_y, res_x))
ce2.fill(ep)

@ti.func
def sphere_contains(loc, center_x, center_y, radius):
    px = loc[0]
    py = loc[1]
    result = False
    if (px - center_x)**2 + (py - center_y)**2 <= radius**2:
        result = True
    return result

@ti.kernel
def setup_coeff(cour_num: float):
    s = cour_num * ti.sqrt(mu / ep) # update coeff

    for i, j in chx:
        chx[i, j] = -s / chx[i, j]
        if i >= 1 and i < res_y:
            if (sphere_contains(Hx_coord[i, j] * scene_x, center_x, center_y, radius)):
                chx[i - 1, j] = -s / geo_mu
    for i, j in chy:
        chy[i, j] = s / chy[i, j]
        if j >= 1 and j < res_x:
            if (sphere_contains(Hy_coord[i, j] * scene_y, center_x, center_y, radius)):
                chy[i, j - 1] = s / geo_mu
    for i, j in ce1:
        ce1[i, j] = -s / ce1[i, j]
        ce2[i, j] = s / ce2[i, j]
        if (sphere_contains(Ez_coord[i, j] * scene_y, center_x, center_y, radius)):
            ce1[i, j] = -s * 1.0 / geo_ep
            ce2[i, j] = s * 1.0 / geo_ep

@ti.kernel
def update_prime_grid(prime:ti.template(), dual_x:ti.template(), dual_y:ti.template(), coeff_c1:ti.template(),coeff_c2:ti.template()):
    for i, j in prime:
        vxlo = dual_x[i, j]
        vxup = dual_x[i + 1, j]
        vylo = dual_y[i, j]
        vyup = dual_y[i, j + 1]
        prime[i, j] += coeff_c1[i, j] * (vxup - vxlo) + coeff_c2[i, j] * (vyup - vylo)
        

@ti.kernel
def update_dual_grid(prime:ti.template(), dual_x:ti.template(), dual_y:ti.template(), coeff_x:ti.template(), coeff_y:ti.template()):
    for i, j in dual_x:
        if i >= 1 and i < res_y:
            c = coeff_x[i - 1, j]
            vclo = prime[i - 1, j]
            vcup = prime[i, j]
            dual_x[i, j] += c * (vcup - vclo)
    for i, j in dual_y:
        if j >= 1 and j < res_x:
            c = coeff_y[i, j - 1]
            vclo = prime[i, j - 1]
            vcup = prime[i, j]
            dual_y[i, j] += c * (vcup - vclo)

@ti.kernel
def add_source(amp:float, mu:float, sigma:float, prime:ti.template(), time:float, position_x:float, position_y:float):
    denom_ = -0.5 / (sigma * sigma)
    for i, j in prime:
        if i == position_x and sphere_contains(Ez_coord[i, j] * scene_y, center_x, center_y, radius):
            prime[i, j] += amp * ti.exp(denom_ * ((time - mu) ** 2))
        
        
# main function
def main(testing=False):
    # create some folders
    logsdir = os.path.join('logs', exp_name)
    os.makedirs(logsdir, exist_ok=True)
    remove_everything_in(logsdir)

    fielddir = 'field'
    fielddir = os.path.join(logsdir, fielddir)
    os.makedirs(fielddir, exist_ok=True)

    ckptdir = 'ckpts'
    ckptdir = os.path.join(logsdir, ckptdir)
    os.makedirs(ckptdir, exist_ok=True)

    shutil.copyfile('./hyperparams.py', f'{logsdir}/hyperparams.py')
    
    Ez.fill(0.0)
    Hy.fill(0.0)
    Hx.fill(0.0)
    
    write_field(Ez.to_numpy(), fielddir, 0, vmin=0, vmax=src_amp/20)
    
    from_frame = 0
    if save_ckpt:
        np.save(os.path.join(ckptdir, "Ez_numpy_" + str(from_frame)), Ez.to_numpy())
        np.save(os.path.join(ckptdir, "Hy_numpy_" + str(from_frame)), Hy.to_numpy())
        np.save(os.path.join(ckptdir, "Hx_numpy_" + str(from_frame)), Hx.to_numpy())
        # np.save(os.path.join(ckptdir, "w_numpy_" + str(from_frame)), w.to_numpy())
     
    i = -1
    setup_coeff(cour_num)
    saved = 1
    for frame_idx in range(total_steps):
        i += 1
        j = i % save_per_frame
        print("[Simulate] Running step: ", i, " / substep: ", j)
        
        
        update_prime_grid(Ez, Hx, Hy, ce1, ce2)
        add_source(src_amp, src_mu, src_sigma, Ez, i * dt, src_x, src_y)
        update_dual_grid(Ez, Hx, Hy, chx, chy)
        
        if frame_idx % save_per_frame == 0:
            np.save(os.path.join(ckptdir, "Ez_numpy_" + str(frame_idx)), Ez.to_numpy())
            np.save(os.path.join(ckptdir, "Hy_numpy_" + str(frame_idx)), Hy.to_numpy())
            np.save(os.path.join(ckptdir, "Hx_numpy_" + str(frame_idx)), Hx.to_numpy())
            write_field(Ez.to_numpy(), fielddir, saved, vmin=-10, vmax=10)
            saved += 1
            
            
if __name__ == '__main__':
    print("[Main] Begin")
    main()
    print("[Main] Complete")
