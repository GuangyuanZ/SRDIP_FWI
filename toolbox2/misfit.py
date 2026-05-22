###############################################################################
#
# SWIT: Seismic Waveform Inversion Toolbox
#
# by Haipeng Li at USTC, haipengl@mail.ustc.edu.cn
# 
# June, 2021  
#
# Misfit module, some of codes are from: https://github.com/rmodrak/seisflows 
#                                      & https://github.com/pysit/pysit
#
###############################################################################


from multiprocessing import Pool
import numpy as np
from scipy import fftpack
from scipy.signal import hilbert
from base import Regularization
# from PyFWI.fwi_tools import Regularization
from tools import get_su_parameter, loadsu, su2array,array2vector,array2vector1
from postprocess import grad_de

def misfit(simu, misfit_type):
    ''' Calculate the misfit function
    '''
    homepath = simu.system.homepath    
    nproc = simu.system.mpiproc
    srcn = simu.source.n

    pool = Pool(nproc)
    misfit = [pool.apply_async(misfit_serial, (homepath, isrc, misfit_type, ))  for isrc in range(srcn)]
    pool.close()
    misfits = [p.get() for p in misfit]
    pool.join()

    return np.sum(misfits)

def misfit_tv(simu, misfit_type,itera):
    ''' Calculate the misfit function
    '''
    homepath = simu.system.homepath    
    nproc = simu.system.mpiproc
    srcn = simu.source.n

    pool = Pool(nproc)
    misfit = [pool.apply_async(misfit_serial_tv, (homepath, isrc, misfit_type,simu, itera,))  for isrc in range(srcn)]
    pool.close()
    misfits = [p.get() for p in misfit]
    pool.join()

    return np.sum(misfits)


def misfit_tv1(simu, misfit_type,itera):
    ''' Calculate the misfit function
    '''
    homepath = simu.system.homepath    
    nproc = simu.system.mpiproc
    srcn = simu.source.n

    pool = Pool(nproc)
    misfit = [pool.apply_async(misfit_serial_tv1, (homepath, isrc, misfit_type,simu, itera,))  for isrc in range(srcn)]
    pool.close()
    misfits = [p.get() for p in misfit]
    pool.join()

    return np.sum(misfits)


def misfit_serial(homepath, isrc, misfit_type):
    ''' Calculate the misfit function for a single shot
    '''

    obs = loadsu(homepath + 'data/obs/src%d_sg_proc.su'%(isrc+1))
    syn = loadsu(homepath + 'data/syn/src%d_sg_proc.su'%(isrc+1))

    if get_su_parameter(obs) != get_su_parameter(syn):
        raise ValueError('obs and syn are not consistent.')

    # Waveform difference L2-norm (Tarantola, 1984)
    if misfit_type.lower() in ['waveform']:
        rsd = misfit_waveform(obs, syn)
    
    # Envelope difference (Wu et al., 2014; Yuan et al., 2015)
    elif misfit_type.lower() in ['envelope']:
        rsd = misfit_envelope(obs, syn)

    # Cross correlation traveltime (Luo & Schuster, 1991; Tromp et al., 2005)
    elif misfit_type.lower() in ['traveltime']:
        rsd = misfit_traveltime(obs, syn)

    # Normalized global-correlation coefficient (Choi & Alkhalifah, 2012)
    elif misfit_type.lower() in ['globalcorrelation']:
        rsd = misfit_global_correlation(obs, syn)

    return rsd

def misfit_serial_tv(homepath, isrc, misfit_type,simu,itera):
    ''' Calculate the misfit function for a single shot
    '''

    obs = loadsu(homepath + 'data/obs/src%d_sg_proc.su'%(isrc+1))
    syn = loadsu(homepath + 'data/syn/src%d_sg_proc.su'%(isrc+1))

    if get_su_parameter(obs) != get_su_parameter(syn):
        raise ValueError('obs and syn are not consistent.')

    # Waveform difference L2-norm (Tarantola, 1984)
    if misfit_type.lower() in ['globalcorrelation_tv']:
        rsd = misfit_global_correlation_tv(obs, syn,simu,itera)


    return rsd

def misfit_serial_tv1(homepath, isrc, misfit_type,simu,itera):
    ''' Calculate the misfit function for a single shot
    '''

    obs = loadsu(homepath + 'data/obs/src%d_sg_proc.su'%(isrc+1))
    syn = loadsu(homepath + 'data/syn/src%d_sg_proc.su'%(isrc+1))

    if get_su_parameter(obs) != get_su_parameter(syn):
        raise ValueError('obs and syn are not consistent.')
    # Waveform difference L2-norm (Tarantola, 1984)
    if misfit_type.lower() in ['globalcorrelation_tv1']:
        rsd = misfit_global_correlation_tv1(obs, syn,simu,itera)


    return rsd



def adjoint_source(simu, misfit_type):
    ''' Caculate the adjoint source
    '''

    homepath = simu.system.homepath    
    nproc = simu.system.mpiproc
    srcn = simu.source.n

    pool = Pool(nproc)
    adj = [pool.apply_async(adjoint_source_serial, (homepath, isrc, misfit_type, ))  for isrc in range(srcn)]
    pool.close()
    adjs = [p.get() for p in adj]
    pool.join()
    return np.array(adjs)



def adjoint_source_serial(homepath, isrc, misfit_type):
    ''' Caculate the adjoint source for a single source
    '''

    obs = loadsu(homepath + 'data/obs/src%d_sg_proc.su'%(isrc+1))
   # syn = loadsu(homepath + 'data/syn/src%d_sg_proc.su'%(isrc+1))
   # we do not use syn data in RTM
    if misfit_type.lower() not in ['rtm']:
        syn = loadsu(homepath + 'data/syn/src%d_sg_proc.su'%(isrc+1))

        if get_su_parameter(obs) != get_su_parameter(syn):
            raise ValueError('obs and syn are not consistent.')

    # Waveform difference L2-norm (Tarantola, 1984)
    if misfit_type.lower() in ['waveform']:
        adj = adjoint_source_waveform(obs, syn)

    # Envelope difference (Wu et al., 2014; Yuan et al., 2015)
    elif misfit_type.lower() in ['envelope']:
        adj = adjoint_source_envelope(obs, syn)

    # Cross correlation traveltime (Luo & Schuster, 1991; Tromp et al., 2005)
    elif misfit_type.lower() in ['traveltime']:
        adj = adjoint_source_traveltime(obs, syn)

    # Normalized global-correlation coefficient (Choi & Alkhalifah, 2012)
    elif misfit_type.lower() in ['globalcorrelation']:
        adj = adjoint_source_global_correlation(obs, syn)
    
    # Normalized global-correlation coefficient (Choi & Alkhalifah, 2012)
    elif misfit_type.lower() in ['globalcorrelation_tv']:
        adj = adjoint_source_global_correlation(obs, syn)

    elif misfit_type.lower() in ['globalcorrelation_tv1']:
        adj = adjoint_source_global_correlation(obs, syn)
    
    # Reverse Time Migration
    elif misfit_type.lower() in ['rtm']:
        adj = adjoint_source_rtm(obs)

    return adj


def adjoint_source_waveform(obs, syn):
    ''' Waveform difference L2-norm (Tarantola, 1984)
    '''
    # parameters
    recn, nt, _ = get_su_parameter(obs)
    adj = np.zeros((recn, nt))

    for irec in range(recn):
        obs_trace = obs[irec].data
        syn_trace = syn[irec].data
        obs_norm = np.linalg.norm(obs_trace, ord=2)
        syn_norm = np.linalg.norm(syn_trace, ord=2)

        adj_trace = np.zeros(nt)

        if obs_norm > 0. and syn_norm > 0.:
            adj_trace = syn_trace - obs_trace

        adj[irec,:] = adj_trace

    return adj


def adjoint_source_envelope(obs, syn):
    ''' Envelope difference (Wu et al., 2014; Yuan et al., 2015)
    '''
    # parameters
    p = 2.0            # envelope_power
    recn, nt, _ = get_su_parameter(obs)
    adj = np.zeros((recn, nt))

    for irec in range(recn):
        obs_trace = obs[irec].data
        syn_trace = syn[irec].data
        obs_norm = np.linalg.norm(obs_trace, ord=2)
        syn_norm = np.linalg.norm(syn_trace, ord=2)

        adj_trace = np.zeros(nt)
        rsd_envelope = np.zeros(nt)
        
        if obs_norm > 0. and syn_norm > 0.:
            syn_Hilbert = hilbert(syn_trace, axis=0).imag
            obs_Hilbert = hilbert(obs_trace, axis=0).imag

            syn_envelope = syn_trace**2.0 + syn_Hilbert**2.0
            obs_envelope = obs_trace**2.0 + obs_Hilbert**2.0
            rsd_envelope = syn_envelope**(p/2.0) - obs_envelope**(p/2.0)

            denvelope_ddata = p * syn_envelope**(p/2.0 - 1.0) * syn_trace
            adj_trace = denvelope_ddata * rsd_envelope

            denvelope_ddataH = p * syn_envelope**(p/2.0 - 1.0) * syn_Hilbert 
            adj_trace += (-hilbert(denvelope_ddataH * rsd_envelope, axis=0)).imag

        adj[irec,:] = adj_trace

    return adj



def adjoint_source_traveltime(obs, syn):
    ''' Cross correlation traveltime (Luo & Schuster, 1991; Tromp et al., 2005)
    '''
    # parameters
    recn, nt, dt = get_su_parameter(obs)
    adj = np.zeros((recn, nt))

    # compute the cross-correlation
    ccmax = cross_correlate_max(obs, syn)

    for irec in range(recn):
        obs_trace = obs[irec].data
        syn_trace = syn[irec].data
        obs_norm = np.linalg.norm(obs_trace, ord=2)
        syn_norm = np.linalg.norm(syn_trace, ord=2)

        adj_trace = np.zeros(nt)
        adj_trace[1:-1] = (syn_trace[2:] - syn_trace[0:-2])/(2.*dt)

        if obs_norm > 0. and syn_norm > 0. and np.sum(abs(adj_trace)) > 0.:
            adj_trace *= 1./(np.sum(adj_trace*adj_trace)*dt)
            adj_trace *= (ccmax[irec]-nt+1)*dt
        else:
            adj_trace *= 0.

        adj[irec, :] = adj_trace

    return adj



def adjoint_source_global_correlation(obs, syn):
    ''' Normalized global-correlation coefficient (Choi & Alkhalifah, 2012)
    '''
    # parameters
    recn, nt, dt = get_su_parameter(obs)
    adj = np.zeros((recn, nt))

    for irec in range(recn):
        obs_trace = obs[irec].data
        syn_trace = syn[irec].data
        obs_norm = np.linalg.norm(obs_trace, ord=2)
        syn_norm = np.linalg.norm(syn_trace, ord=2)

        adj_trace = np.zeros(nt)
        if obs_norm > 0. and syn_norm > 0.:
            obs_trace_norm = obs_trace / obs_norm
            syn_trace_norm = syn_trace / syn_norm
            adj_trace = 1.0/syn_norm * (syn_trace_norm * np.corrcoef(syn_trace_norm, obs_trace_norm)[0,1] - obs_trace_norm)
        else:
            adj_trace *= 0.

        adj[irec, :] = adj_trace

    return adj



def adjoint_source_rtm(obs):
    ''' Reverse Time Migration
    '''
    # parameters
    recn, nt, _ = get_su_parameter(obs)
    adj = np.zeros((recn, nt))

    for irec in range(recn):
        adj[irec,:] = obs[irec].data

        # # differentiate the seismic trace twice
        # adj[irec,:] = np.diff(adj[irec,:], prepend = adj[irec,0])
        # adj[irec,:] = np.diff(adj[irec,:], prepend = adj[irec,0])

    return adj



def misfit_waveform(obs, syn):
    ''' Waveform difference L2-norm (Tarantola, 1984)
    '''
    # parameters
    recn, nt, dt = get_su_parameter(obs)
    rsd = np.zeros(1)

    for irec in range(recn):
        obs_trace = obs[irec].data
        syn_trace = syn[irec].data
        obs_norm = np.linalg.norm(obs_trace, ord=2)
        syn_norm = np.linalg.norm(syn_trace, ord=2)

        rsd_trace = np.zeros(nt)

        if obs_norm > 0. and syn_norm > 0.:
            rsd_trace = syn_trace - obs_trace
            rsd += np.sqrt(np.sum(rsd_trace*rsd_trace*dt))

    return rsd


def misfit_envelope(obs, syn):
    ''' Envelope difference (Wu et al., 2014; Yuan et al., 2015)
    '''
    # parameters
    p = 2.0            # envelope_power
    recn, nt, dt = get_su_parameter(obs)
    rsd = np.zeros(1)

    for irec in range(recn):
        obs_trace = obs[irec].data
        syn_trace = syn[irec].data
        obs_norm = np.linalg.norm(obs_trace, ord=2)
        syn_norm = np.linalg.norm(syn_trace, ord=2)

        if obs_norm > 0. and syn_norm > 0.:
            syn_Hilbert = hilbert(syn_trace, axis=0).imag
            obs_Hilbert = hilbert(obs_trace, axis=0).imag

            syn_envelope = syn_trace**2.0 + syn_Hilbert**2.0
            obs_envelope = obs_trace**2.0 + obs_Hilbert**2.0
            rsd_envelope = syn_envelope**(p/2.0) - obs_envelope**(p/2.0)

            rsd += np.linalg.norm(rsd_envelope)**2

    return rsd * dt


def misfit_traveltime(obs, syn):
    ''' Cross correlation traveltime (Luo & Schuster, 1991; Tromp et al., 2005)
    '''
    # parameters
    recn, nt, dt = get_su_parameter(obs)
    rsd = np.zeros(1)

    # compute the cross-correlation in an awkward paralle scheme
    ccmax =  cross_correlate_max(obs, syn)

    for irec in range(recn):
        obs_trace = obs[irec].data
        syn_trace = syn[irec].data
        obs_norm = np.linalg.norm(obs_trace, ord=2)
        syn_norm = np.linalg.norm(syn_trace, ord=2)

        adj_trace = np.zeros(nt)
        adj_trace[1:-1] = (syn_trace[2:] - syn_trace[0:-2])/(2.*dt)

        if obs_norm > 0. and syn_norm > 0. and np.sum(abs(adj_trace)) > 0.:
            rsd += 0.5 * np.power(ccmax[irec]-nt+1, 2)*dt

    return rsd



def misfit_global_correlation(obs, syn):
    ''' Normalized global-correlation coefficient (Choi & Alkhalifah, 2012)
    '''
    # get parameters
    recn, nt, dt = get_su_parameter(obs)
    rsd = np.zeros(1)

    for irec in range(recn):
        obs_trace = obs[irec].data
        syn_trace = syn[irec].data
        obs_norm = np.linalg.norm(obs_trace, ord=2)
        syn_norm = np.linalg.norm(syn_trace, ord=2)
        if obs_norm > 0. and syn_norm > 0.:
            obs_trace_norm = obs_trace/obs_norm
            syn_trace_norm = syn_trace/syn_norm
            rsd += - np.corrcoef(syn_trace_norm, obs_trace_norm)[0,1]

    return rsd

def misfit_global_correlation_tv(obs, syn,simu, itera):
    ''' Normalized global-correlation coefficient (Choi & Alkhalifah, 2012)
    '''
    # get parameters
    recn, nt, dt = get_su_parameter(obs)
    rsd = np.zeros(1)

    nx = simu.model.nx
    nz = simu.model.nz

    m = simu.model.vp 

    for irec in range(recn):
        obs_trace = obs[irec].data
        syn_trace = syn[irec].data
        obs_norm = np.linalg.norm(obs_trace, ord=2)
        syn_norm = np.linalg.norm(syn_trace, ord=2)
        if obs_norm > 0. and syn_norm > 0.:
            obs_trace_norm = obs_trace/obs_norm
            syn_trace_norm = syn_trace/syn_norm
            rsd += - np.corrcoef(syn_trace_norm, obs_trace_norm)[0,1]

    # Method1
    # # L0 = np.diag(1*np.ones(nz-1),1) + np.diag(-2*np.ones(nz)) + np.diag(1*np.ones(nz-1),-1)
    # # L0[0,:]  = 0
    # # L0[-1,:] = 0
    # L0 = np.diag(1*np.ones(nz),0) + np.diag(-1*np.ones(nz-1),1)
    # L0[-1,:] = 0
    # L0 = -L0

    # # L1 = np.diag(1*np.ones(nx-1),1) + np.diag(1*np.ones(nx-1),-1) + np.diag(-2*np.ones(nx))
    # # L1[0,:]  = 0
    # # L1[-1,:] = 0
    # L1 = np.diag(1*np.ones(nx),0) + np.diag(-1*np.ones(nx-1),1)
    # L1[-1,:] = 0
    # L1 = -L1

    # # m_norm_z = np.matmul(L0, m.T).T / simu.model.dx 
    # # m_norm_x = np.matmul(L1, m) / simu.model.dx
    # m_norm_z = np.matmul(L0, m.T).T
    # m_norm_x = np.matmul(L1, m)
    
    # alphax = regular_StepLR(itera,simu.model.step_size,simu.model.alphax,simu.model.gamma)
    # alphaz = regular_StepLR(itera,simu.model.step_size,simu.model.alphaz,simu.model.gamma)

    # rsd_norm = np.sum(alphax*np.abs(m_norm_x) + alphaz*np.abs(m_norm_z))
    
    # Method2
    tv_f = 0.0
    for i in range(nx - 1):
        for j in range(nz - 1):
            tv_f += np.sqrt((m[i + 1, j] - m[i, j]) ** 2 + (m[i, j + 1] - m[i, j]) ** 2)
    
    lmbda = simu.model.lmbda
    rsd_norm = lmbda * (tv_f/simu.model.dx**2)
    # rsd_norm = tv_f
    # rsd_norm *= rsd / abs(rsd_norm).max()

    rsd = rsd + rsd_norm

    return rsd


def misfit_global_correlation_tv1(obs, syn,simu, itera):
    ''' Normalized global-correlation coefficient (Choi & Alkhalifah, 2012)
    '''
    # get parameters
    recn, nt, dt = get_su_parameter(obs)
    rsd = np.zeros(1)

    nx = simu.model.nx
    nz = simu.model.nz

    m = simu.model.vp 
    # m_tmp = array2vector1(m)

    for irec in range(recn):
        obs_trace = obs[irec].data
        syn_trace = syn[irec].data
        obs_norm = np.linalg.norm(obs_trace, ord=2)
        syn_norm = np.linalg.norm(syn_trace, ord=2)
        if obs_norm > 0. and syn_norm > 0.:
            obs_trace_norm = obs_trace/obs_norm
            syn_trace_norm = syn_trace/syn_norm
            rsd += - np.corrcoef(syn_trace_norm, obs_trace_norm)[0,1]
    
    # # Method1
    # inpa = {}
    # inpa['tv'] = {
    # 'az': simu.model.alphaz,  # regularizatin in the z-direction
    # 'ax': simu.model.alphax, # regularizatin in the x-direction
    # 'lambda_weight': simu.model.lmbda # regularizatin weight
    # }

    # regularization = Regularization(simu.model.nx, # number of samples in the x-direction
    #                             simu.model.nz, # number of samples in the z-direction
    #                             simu.model.dx, # Spatial sampling rate in the x-direction
    #                             simu.model.dx # Spatial sampling rate in the z-direction
    #                              )

    # rsd_norm, _ = regularization.cost_regularization(m_tmp,
    #                                               tv_properties=inpa['tv'],
    #                                               tikhonov_properties=None
    #                                               )

    # rsd = rsd + rsd_norm

    # Method2
    mx, my = grad_de(m)
    rsd_norm = np.sum(np.abs(mx) + np.abs(my))
    rsd = rsd - rsd_norm

    return rsd

def regular_StepLR(itera,step_size,alpha,gamma=0.8):
    n = itera//step_size
    return alpha*np.power(gamma,n)

def cross_correlate_max(obs, syn):
    ''' calculate the cross-correlation lag between two traces
    '''

    obs_data = su2array(obs)
    syn_data = su2array(syn)
    nt = np.size(obs_data, -1)

    a =   fftpack.fft(obs_data)
    b = - fftpack.fft(syn_data).conjugate()

    cc = np.argmax(np.abs(fftpack.ifft(a*b)), -1) - 1
    cc[np.where(cc<nt//2)] = cc[np.where(cc<nt//2)] + nt

    return cc
