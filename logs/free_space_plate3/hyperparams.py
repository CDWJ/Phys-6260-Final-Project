# scene
res_x = 400
res_y = 400
# Assuming the side of the domain is 1e5 m then we normalize everything accordingly
scene_x = 1 #m 
scene_y = 1 #m
ep = 8.8541878176e-12
mu = 1.
duration = 1e-9 #s
#dt = 1e-12

# source
src_x = 200
src_y = 200
src_mu = 2 #ps
src_sigma = 5 #ps
src_amp = 0.01

# geo
center_x = 20 #m
center_y = 20 #m
radius = 5 #um
geo_ep = 1
geo_mu = 1

# speed of light
sp_light = 299792458 # m/s

save_per_frame = 10
save_ckpt = False
exp_name = "free_space_plate3"

tau = 1.0e-10;
t0 = 5.0e-10;
#Source Calculations
# t=np.arange(0,steps-1)*dt;
# s=dx/(2*c0)+dt/2;
# Esrc =np.exp(-(np.power((t-t0)/tau,2)));
