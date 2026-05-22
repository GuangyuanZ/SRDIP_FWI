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
from inversion import inversion, source_inversion
from plot import plot_geometry, plot_model2D, plot_stf, plot_trace
from preprocess import process_workflow
from solver import forward, source_wavelet
from tools import saveparjson, smooth2d,loadbinfloat32
from DIP_FWI import DIP_inversion_unet_dn
import torch
from inversion import rtm
### system setup
homepath  = '/data/guangyuan/SRDIP_FWI/examples/case_partial_BP/PBP_SRDIPFWI/'   # working path
mpiproc   = 32                                     # mpi process for fd2dmpi
figaspect = 1.0                                    # Figure aspect

### model setup
nx,  nz      = [188,  68]                         # Grid number along x and z directions
pml, fs      = [50,  True]                         # Grid number for PML layers (use a large one) and free surface
dx,  dt,  nt = [133, 0.009, 3000]                   # Grid size, time interval, and time step

# velocity and density
vp_true = np.zeros((nx, nz))
vp_init = np.zeros((nx, nz))
rho_true = np.zeros((nx, nz))
rho_init = np.zeros((nx, nz))

vp_true = np.loadtxt(homepath + 'model/vp_true.dat') # True velocity（Any model is ok for real data）

# initial model
# vp_init  = smooth2d(vp_true, span = 10) # Smoothed initial model
# vp_init[np.round(vp_true, 3) == 1486.000] = 1486.000 # Water layers

vp_init = np.loadtxt(homepath + 'model/vp_init1.dat') # Defined initial model

grad_mask = np.ones_like(vp_init) # grad mask(mask water layer)
grad_mask[np.round(vp_true, 3) == 1486.000] = 0


rho_true = np.power(vp_true, 0.25) * 310 # true density model
rho_init = np.power(vp_init, 0.25) * 310 # init density model

### sources setup 
f0    = 1.2                                        # Dominant frequency in Hz
srcxz = np.loadtxt(homepath + 'model/source_coordinate.dat') # Source location file（m） 2D测线，需要自己根据模型定义0参考点以及震源相对0参考点直线距离
srcn  = srcxz.shape[0]                             # Source number along x axis
wavelet  = np.zeros((srcn, nt))                    # source wavelet
for isrc in range(srcn):
    wavelet[isrc,:] = source_wavelet(nt, dt, f0, 'ricker')

### receivers setup
temp = np.loadtxt(homepath + 'model/receiver_coordinate.dat') # Receiver location file（m）2D测线，需要自己根据模型定义0参考点以及检波器相对0参考点直线距离
recn  = temp.shape[0]                              # receiver number
recxz = np.zeros((srcn, recn, 2))                  # receiver positions
for isrc in range(srcn):
    recxz[isrc,:,0] = temp[:,0]                    # receiver x position (m)
    recxz[isrc,:,1] = temp[:,1]                    # receiver z position (m)


### inversion parameter
misfit_type = 'Globalcorrelation'                           # 'Traveltime', 'Waveform', 'Globalcorrelation','Globalcorrelation_tv' Misfit
scheme      = 'NLCG'                               # 'NLCG', 'LBFGS'
maxiter     = 1800                                   # maximum iteration number,  i.e., 20
step_length = 0.01                                 # maximum update percentage, i.e., 0.05 Multi-scale FWI
step_length1 = 0.01                                 # Network parameter update step size
step_length2 = 25                                 # Network input update step size(Self-reinforcement DIPFWI)
decay_step1 = 150                                 # Network parameter update decay step size
decay_step2 = 50                                 # Network input update decay step size(Self-reinforcement DIPFWI)
vpmax       = vp_true.max()                                 # maximum allowed velocity
vpmin       = vp_true.min()                                # minimum allowed velocity
marine_or_land = 'Land'                            # 'Land' or 'Marine'
device = 'cpu' # or device = 'cpu'              # device for DIPFWI or SRDIPFWI 'cuda:0' or 'cpu'

# gradient postprocess
grad_mute = 2                                     # mute source energy for 'Land', or water mask for 'Marine'
grad_smooth = 0                                    # gradient smooth radius 

# data filter
fre_filter = 'Lowpass'                                # 'Bandpass', 'Lowpass', 'Highpass', 'None'
fre_low  = 25                                      # low  frequency corner (units: Hz)
fre_high = 40                                      # high frequency corner (units: Hz) fre_high > fre_low

# mute later arrival
mute_late_arrival = False                         # pick first break and mute later arrival
mute_late_window = -0.2                             # mute time window (units: time)

# mute offset
mute_offset_short  = False                          # mute short-offset traces 
mute_offset_long   = False                         # mute long-offset traces 
mute_offset_short_dis = 500                        # mute short-offset distance (units: m)
mute_offset_long_dis  = 10000                       # mute long-offset distance (units: m)

# data normalize
normalize = ['None']                               # 'Max-Trace', 'L1-Trace', 'L2-Trace', 'L1-Event', 'L2-Event', 'None'

# TV normalization paramaters
# 目标函数带有_tv的会用到，否则用不到
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

### optimize setup 
optim = base.optimize(misfit_type, scheme, maxiter, step_length,step_length1,step_length2,decay_step1,decay_step2,vpmax, vpmin, marine_or_land,
                grad_mute, grad_smooth,
                fre_filter, fre_low, fre_high, 
                mute_late_arrival, mute_late_window, normalize,
                mute_offset_short, mute_offset_long, 
                mute_offset_short_dis, mute_offset_long_dis,device,grad_mask = grad_mask)

# ### Save parameter as json
# saveparjson(simu, optim)

# ### Plots
plot_geometry(simu)
plot_stf(simu, isrc=1,  stf_type='in-use', t_end = 2.0)
plot_model2D(simu, vp_true.T, vpmin, vpmax, 'vp-obs', colormap = 'jet')
plot_model2D(simu, vp_init.T, vpmin, vpmax, 'vp-ini', colormap = 'jet')

# ### forward data 
forward(simu, simu_type='obs', savesnap=0) # 生成观测数据，做实际数据时注释掉
process_workflow(simu, optim, simu_type='obs') # 对数据进行上述处理(滤波，切除，归一化，初至拾取等等)
plot_trace(simu, 'obs',      simu_type='obs', suffix='',      src_space=1, trace_space=5, scale = 0.8, color='k') #观测波形
plot_trace(simu, 'obs-proc', simu_type='obs', suffix='_proc', src_space=1, trace_space=5, scale = 0.8, color='k') #处理后的观测波形

### begin inversion
DIP_inversion_unet_dn(simu, optim,4,{'vp':vp_init,'rho':rho_init},load_pretrained = False)
