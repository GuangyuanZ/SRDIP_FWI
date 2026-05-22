###############################################################################
#
# SWIT: Seismic Waveform Inversion Toolbox
#
# by Haipeng Li at USTC, haipengl@mail.ustc.edu.cn
# 
# June, 2021  
#
# Workflow
#
########################################################################################

# import modules
import numpy as np
import base
from inversion import inversion
from plot import plot_geometry, plot_model2D, plot_stf, plot_trace
from preprocess import process_workflow
from solver import forward, source_wavelet
from tools import saveparjson, smooth2d
from DIP_FWI import DIP_inversion_unet_dn

### system setup
homepath  = '/xxx/case_full_BP/FBF_SRDIPFWI/'   # working path
mpiproc   = 81                                     # mpi process for fd2dmpi
figaspect = 1.0                                    # Figure aspect

### model setup
nx,  nz      = [401, 91]                         # Grid number along x and z directions
pml, fs      = [50,  True]                         # Grid number for PML layers (use a large one) and free surface
dx,  dt,  nt = [62.5, 0.004, 5001]                   # Grid size, time interval, and time step

# velocity and density
vp_true = np.zeros((nx, nz))
vp_init = np.zeros((nx, nz))
rho_true = np.zeros((nx, nz))
rho_init = np.zeros((nx, nz))

# true model
vp_true = np.loadtxt(homepath + 'model/vp_true.dat')
print(vp_true.min())
print(vp_true.max())

vp_init = np.loadtxt(homepath + 'model/vp_init.dat')
grad_mask = np.ones_like(vp_true)



# density models, (Gardner, 1974)
rho_true = np.power(vp_true, 0.25) * 310 
rho_init = np.power(vp_init, 0.25) * 310 

### sources setup 
f0    = 2.3                                        # Dominant frequency in Hz
srcxz = np.loadtxt(homepath + 'model/source_coordinate.dat')
# print(srcxz)
srcn  = srcxz.shape[0]                             # Source number along x axis
wavelet  = np.zeros((srcn, nt))                    # source wavelet
for isrc in range(srcn):
    wavelet[isrc,:] = source_wavelet(nt, dt, f0, 'ricker')

### receivers setup
temp = np.loadtxt(homepath + 'model/receiver_coordinate.dat')
recn  = temp.shape[0]                              # receiver number
recxz = np.zeros((srcn, recn, 2))                  # receiver positions
for isrc in range(srcn):
    recxz[isrc,:,0] = temp[:,0]                    # receiver x position (m)
    recxz[isrc,:,1] = temp[:,1]                    # receiver z position (m)


### inversion parameter
misfit_type = 'Globalcorrelation'                                # 'Traveltime', 'Waveform', 'Globalcorrelation'
scheme      = 'NLCG'                               # 'NLCG', 'LBFGS'
maxiter     = 3000                                  # maximum iteration number,  i.e., 20
step_length = 0.01                                 # maximum update percentage, i.e., 0.05 Multi-scale FWI
step_length1 = 0.01                                 # Network parameter update step size
step_length2 = 2                                 # Network input update step size(Self-reinforcement DIPFWI)
decay_step1 = 400                                 # Network parameter update decay step size
decay_step2 = 400                                 # Network input update decay step size(Self-reinforcement DIPFWI)
vpmax       = vp_true.max()                                 # maximum allowed velocity
vpmin       = vp_true.min()                                 # minimum allowed velocity
marine_or_land = 'Land'                            # 'Land' or 'Marine'
device = 'cpu' # or device = 'cpu' 
# gradient postprocess
grad_mute =  15                                     # mute source energy for 'Land', or water mask for 'Marine'
grad_smooth = 0                                    # gradient smooth radius 
# grad_mask = np.ones_like(vp_true)
# grad_mask[vp_true==1500] = 0.0

# data filter
fre_filter = 'None'                                # 'Bandpass', 'Lowpass', 'Highpass', 'None'
fre_low  = 10                                      # low  frequency corner (units: Hz)
fre_high = 20                                      # high frequency corner (units: Hz)

# mute later arrival
mute_late_arrival = False                           # pick first break and mute later arrival
mute_late_window = -0.25                           # mute time window (units: time)

# mute offset
mute_offset_short  = False                         # mute short-offset traces 
mute_offset_long   = False                         # mute long-offset traces 
mute_offset_short_dis = 500                        # mute short-offset distance (units: m)
mute_offset_long_dis  = 7000                       # mute long-offset distance (units: m)

# data normalize
normalize = ['None']                               # 'Max-Trace', 'L1-Trace', 'L2-Trace', 'L1-Event', 'L2-Event', 'None'
topography='flat'
lmbda = 5e-6
tau = 1/8
gamma = 0.5
alphax = 0.2
alphaz = 0.0
step_size = 80
norm_iter = 500

### simulate setup 
sys  = base.system(homepath, mpiproc, figaspect=figaspect)
mod  = base.model(nx, nz, dx, dt, nt, fs, pml, vp_true, rho_true,step_size,norm_iter,lmbda,tau,gamma,alphax,alphaz)
# mod  = base.model(nx, nz, dx, dt, nt, fs, pml, vp_true, rho_true)
src  = base.source(f0, srcn, srcxz, wavelet)
rec  = base.receiver(recn, recxz)
simu = base.simulate(mod, src, rec, sys)

vv = simu.model.vp
print(vv.shape)

### optimize setup 
optim = base.optimize(misfit_type, scheme, maxiter, step_length,step_length1,step_length2,decay_step1,decay_step2,vpmax, vpmin, marine_or_land,
                grad_mute, grad_smooth,
                fre_filter, fre_low, fre_high, 
                mute_late_arrival, mute_late_window, normalize,
                mute_offset_short, mute_offset_long, 
                mute_offset_short_dis, mute_offset_long_dis,device,grad_mask = grad_mask)

### Save parameter as json
saveparjson(simu, optim)

### Plots
plot_geometry(simu)
plot_stf(simu, isrc=1,  stf_type='in-use', t_end = 2.0)
plot_model2D(simu, vp_true.T, vpmin, vpmax, 'vp-obs', colormap = 'jet')
plot_model2D(simu, vp_init.T, vpmin, vpmax, 'vp-ini', colormap = 'jet')

## forward data
# forward(simu, simu_type='obs', savesnap=0)
# process_workflow(simu, optim, simu_type='obs')
# plot_trace(simu, 'obs',      simu_type='obs', suffix='',      src_space=1, trace_space=5, scale = 0.8, color='k')
# plot_trace(simu, 'obs-proc', simu_type='obs', suffix='_proc', src_space=1, trace_space=5, scale = 0.8, color='k')

### begin inversion
# rtm(simu, optim, {'vp':vp_init,'rho':rho_init})

DIP_inversion_unet_dn(simu, optim,4,{'vp':vp_init,'rho':rho_init},load_pretrained = False)
