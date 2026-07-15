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
import torch
from inversion import rtm
# itera = [   1,    1,    1,    1,    1]
itera = [   250,    250,    250,    250,    3000]
smooth = [   16,    8,    4,     2,     0]
freque = [   0.7,  1.2,  1.8,  5,    10]
window = [   2.5,  5,  10,  15,  20]
step   = [   0.01,  0.01,  0.01,  0.01,  0.01]
mute_flag = ['True','True','True','True','False']
### system setup
for istage in range(5):
    modelpath = '/xxx/FBF_MSFWI/model/' 
    homepath  = '/xxx/FBF_MSFWI/stage%d/'%(istage+1)   # working path
    mpiproc   = 42                                     # mpi process for fd2dmpi
    figaspect = 1.0                                    # Figure aspect

    ### model setup
    nx,  nz      = [401, 91]                       # Grid number along x and z directions
    pml, fs      = [50,  True]                         # Grid number for PML layers (use a large one) and free surface
    dx,  dt,  nt = [62.5, 0.004, 5001]                   # Grid size, time interval, and time step

    # velocity and density
    vp_true = np.zeros((nx, nz))
    vp_init = np.zeros((nx, nz))
    rho_true = np.zeros((nx, nz))
    rho_init = np.zeros((nx, nz))

    vp_true = np.loadtxt(homepath + 'model/vp_true.dat')
    print(vp_true.min())
    print(vp_true.max())
    # initial model
    if istage == 0:
        vp_init = np.loadtxt(homepath + 'model/vp_init.dat')
        grad_mask = np.ones_like(vp_init)
        grad_mask[np.round(vp_true, 3) == 1486.000] = 0
    else:
        vp_init = loadbinfloat32('/xxx/FBF_MSFWI/stage%d/outputs/velocity/vp-%d.bin'%(istage,itera[istage-1])).reshape(nx, nz)
        grad_mask = np.ones_like(vp_true)
        grad_mask[np.round(vp_true, 3) == 1486.000] = 0

    rho_true = np.power(vp_init, 0.25) * 310 
    rho_init = np.power(vp_init, 0.25) * 310 

    f0    = 2.3                                        # Dominant frequency in Hz
    srcxz = np.loadtxt(homepath + 'model/source_coordinate.dat')
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
    misfit_type = 'Globalcorrelation'                           # 'Traveltime', 'Waveform', 'Globalcorrelation'
    scheme      = 'NLCG'                               # 'NLCG', 'LBFGS'

    maxiter     = itera[istage] 
    step_length = 0.01                                 # maximum update percentage, i.e., 0.05 Multi-scale FWI
    step_length1 = 0.01                                 # Network parameter update step size
    step_length2 = 1                                 # Network input update step size(Self-reinforcement DIPFWI)
    decay_step1 = 250                                 # Network parameter update decay step size
    decay_step2 = 250                                 # Network input update decay step size(Self-reinforcement DIPFWI)
    vpmax       = 5000                                 # maximum allowed velocity
    vpmin       = vp_true.min()                                 # minimum allowed velocity
    marine_or_land = 'Land'                            # 'Land' or 'Marine'
    device = 'cuda:0' # or device = 'cpu' 
    marine_or_land = 'Land'                            # 'Land' or 'Marine'

    # gradient postprocess
    grad_mute = 15                                     # mute source energy for 'Land', or water mask for 'Marine'
    grad_smooth = smooth[istage]                                    # gradient smooth radius 

    # data filter
    fre_filter = 'Lowpass'                                # 'Bandpass', 'Lowpass', 'Highpass', 'None'
    fre_low  = freque[istage]                                      # low  frequency corner (units: Hz)
    fre_high = 40                                      # high frequency corner (units: Hz)

    # mute later arrival
    mute_late_arrival = mute_flag[istage]                          # pick first break and mute later arrival
    mute_late_window = window[istage]                              # mute time window (units: time)

    # mute offset
    mute_offset_short  = False                          # mute short-offset traces 
    mute_offset_long   = False                         # mute long-offset traces 
    mute_offset_short_dis = 500                        # mute short-offset distance (units: m)
    mute_offset_long_dis  = 10000                       # mute long-offset distance (units: m)

    # data normalize
    normalize = ['None']                               # 'Max-Trace', 'L1-Trace', 'L2-Trace', 'L1-Event', 'L2-Event', 'None'
    topography = 'flat'
    lmbda = 5e-6
    tau = 1/8
    gamma = 0.5
    alphax = 0.2
    alphaz = 0.0
    step_size = 80
    norm_iter = 500



    ### simulate setup 
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

    # ### Save parameter as json
    # saveparjson(simu, optim)

    # ### Plots
    plot_geometry(simu)
    plot_stf(simu, isrc=1,  stf_type='in-use', t_end = 2.0)
    plot_model2D(simu, vp_true.T, vpmin, vpmax, 'vp-obs', colormap = 'jet')
    plot_model2D(simu, vp_init.T, vpmin, vpmax, 'vp-ini', colormap = 'jet')

    # ### forward data
    forward(simu, simu_type='obs', savesnap=0)
    process_workflow(simu, optim, simu_type='obs')
    plot_trace(simu, 'obs',      simu_type='obs', suffix='',      src_space=1, trace_space=5, scale = 0.8, color='k')
    plot_trace(simu, 'obs-proc', simu_type='obs', suffix='_proc', src_space=1, trace_space=5, scale = 0.8, color='k')

    ### begin inversion
    inversion(simu, optim, {'vp':vp_init,'rho':rho_init})

    # ### posterior source inversion
    # source_inversion(simu, inv_offset=20000)


