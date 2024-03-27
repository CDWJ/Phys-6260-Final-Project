# scene
res_x = 512
res_y = 512
scene_x = 40 #um
scene_y = 40 #um
ep = 1.
mu = 1.
duration = 0.5 #ps

# source
src_x = 256
src_y = 256
src_mu = 0.02 #ps
src_sigma = 0.005 #ps
src_amp = 5

# geo
center_x = 20 #um
center_y = 20 #um
radius = 5 #um
geo_ep = 12
geo_mu = 5

# speed of light
sp_light = 299792458 / 1000000 # m/s to um/ps

save_per_frame = 10
save_ckpt = False
exp_name = "free_space_plate2"