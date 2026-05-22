###############################################################################
#
# SWIT: Seismic Waveform Inversion Toolbox
#
# by Haipeng Li at USTC, haipengl@mail.ustc.edu.cn
# 
# June, 2021  
#
# Postprocess module
#
###############################################################################

import numpy as np
import scipy.signal

from plot import plot_model2D
from tools import array2vector, smooth2d, vector2array,vector2array,array2vector1
from base import Regularization
# from PyFWI.fwi_tools import Regularization
from tv_norm import TV_projection

def shrink(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)

def grad_de(m):
    gx = np.zeros_like(m)
    gy = np.zeros_like(m)

    gx[:-1, :] = m[1:, :] - m[:-1, :]
    gy[:, :-1] = m[:, 1:] - m[:, :-1]

    return gx, gy

def div(px, py):
    out = np.zeros_like(px)

    # out[1:, :] += px[1:, :] - px[:-1, :]
    # out[:, 1:] += py[:, 1:] - py[:, :-1]
    out[0, :] += px[0, :]
    out[1:-1, :] += px[1:-1, :] - px[0:-2, :]
    out[-1, :] += -px[-2, :]

    out[:, 0] += py[:, 0]
    out[:, 1:-1] += py[:, 1:-1] - py[:, 0:-2]
    out[:, -1] += -py[:, -2]

    return out

def grad_precond(simu, optim, grad, forw, back):
    ''' Gradient preconditioning
    '''
    nx = simu.model.nx
    nz = simu.model.nz
    vp = simu.model.vp
    vpmax = optim.vpmax
    marine_or_land = optim.marine_or_land
    grad_mute = optim.grad_mute
    grad_thred = optim.grad_thred
    grad_smooth = optim.grad_smooth
    grad_mask = optim.grad_mask
    grad = vector2array(grad, nx, nz)
    forw = vector2array(forw, nx, nz)
    back = vector2array(back, nx, nz)
    
    # plot forward and backward illumination
    # plot_model2D(simu, forw.T, np.min(forw), 10 * np.median(forw), 'forw-illum-%03d' % optim.iter, 'my_seismic_cmap')
    # plot_model2D(simu, back.T, np.min(back), 10 * np.median(back), 'back-illum-%03d' % optim.iter, 'my_seismic_cmap')
    # grad[::-1] = grad[::-2]
    # grad[nx-1,:] = 0
    # grad[nx-2,:] = 0
    # grad[nx-1,:] = grad[nx-3,:]
    # grad[nx-2,:] = grad[nx-3,:]

    if grad_mute > 0:
        grad *= grad_taper(nx, nz, tapersize = grad_mute, thred = grad_thred, marine_or_land=marine_or_land)
    #apply the inverse Hessian
    if min(nx, nz) > 40:      # set 40 grids in default
        span = 40
    else:                     # in case the grid number is less than 40
        span = int(min(nx, nz)/2)
    forw = smooth2d(forw, span)
    back = smooth2d(back, span)
    
    epsilon = 0.0001    
    forw = forw / np.max(forw)
    back = back / np.max(back)
    precond = forw + back
    precond = precond / np.max(precond)
    precond[precond < epsilon] = epsilon
    grad = grad / precond

    # smooth the gradient
    if grad_smooth > 0:
        # exclude water-layer
        if marine_or_land in ['Marine', 'Offshore']: 
            grad[:,grad_mute:] = smooth2d(grad[:,grad_mute:], span=grad_smooth)
        # land gradient smooth
        else:
            grad = smooth2d(grad, span=grad_smooth)
    # apply taper mask, land daming or water-layer masking

    if np.any(grad_mask == None):
        pass
    else:
        if np.shape(grad_mask) != np.shape(grad):
            raise('Wrong size of grad mask: the size of the mask should be identical to the size of vp model')
        else:
            grad *= grad_mask

    # gradient with respect to the velocity
    grad = - 2 * grad   #  / np.power(simu.model.vp, 3)

    # scale the gradient properly
    # mask = np.loadtxt(simu.system.homepath + '/model/mask.dat')
    # grad = grad * (1-mask)
    grad *= vpmax / abs(grad).max()

  

    return array2vector(grad)

def grad_precond_tv(simu, optim, grad, forw, back,iter):
    ''' Gradient preconditioning
    '''
    nx = simu.model.nx
    nz = simu.model.nz
    vp = simu.model.vp
    vpmax = optim.vpmax
    marine_or_land = optim.marine_or_land
    grad_mute = optim.grad_mute
    grad_thred = optim.grad_thred
    grad_smooth = optim.grad_smooth
    grad = vector2array(grad, nx, nz)
    forw = vector2array(forw, nx, nz)
    back = vector2array(back, nx, nz)
    grad_mask = optim.grad_mask
    grad_fwi_tmp = np.copy(grad)

    lmbda = simu.model.lmbda 
    print('Current gradient misfit lmbda: ',lmbda)
    grad = TV_projection(grad_fwi_tmp, simu.model.lmbda, iterMax=simu.model.norm_iter, tau=simu.model.tau)
    # grad = PDHG_ROF(grad_fwi_tmp, simu.model.lmbda, iterMax=simu.model.norm_iter, tau=simu.model.tau,sigma=None)
    
    # plot forward and backward illumination
    plot_model2D(simu, forw.T, np.min(forw), 10 * np.median(forw), 'forw-illum-%03d' % optim.iter, 'my_seismic_cmap')
    plot_model2D(simu, back.T, np.min(back), 10 * np.median(back), 'back-illum-%03d' % optim.iter, 'my_seismic_cmap')

    # apply taper mask, land daming or water-layer masking
    if grad_mute > 0:
        grad *= grad_taper(nx, nz, tapersize = grad_mute, thred = grad_thred, marine_or_land=marine_or_land)

    #apply the inverse Hessian
    if min(nx, nz) > 40:      # set 40 grids in default
        span = 40
    else:                     # in case the grid number is less than 40
        span = int(min(nx, nz)/2)
    forw = smooth2d(forw, span)
    back = smooth2d(back, span)
    
    epsilon = 0.0001
    forw = forw / np.max(forw)
    back = back / np.max(back)
    precond = forw + back
    precond = precond / np.max(precond)
    precond[precond < epsilon] = epsilon
    grad = grad / precond

    # smooth the gradient
    if grad_smooth > 0:
        # exclude water-layer
        if marine_or_land in ['Marine', 'Offshore']: 
            grad[:,grad_mute:] = smooth2d(grad[:,grad_mute:], span=grad_smooth)
        # land gradient smooth
        else:
            grad = smooth2d(grad, span=grad_smooth)
    
    if np.any(grad_mask == None):
        pass
    else:
        if np.shape(grad_mask) != np.shape(grad):
            raise('Wrong size of grad mask: the size of the mask should be identical to the size of vp model')
        else:
            grad *= grad_mask

    # gradient with respect to the velocity
    grad = - 2 * grad   #  / np.power(simu.model.vp, 3)
    # # grad = np.where(np.abs(grad) < 0.02*np.max(np.abs(grad) ),0,grad)
    # for iz in range(nz):
        # grad[:,iz] = grad[:,iz] /((iz+1)*(iz+1))
    # grad = np.where(np.abs(grad) < 0.001*np.max(np.abs(grad) ),0,grad)
    # scale the gradient properly
    # mask = np.loadtxt('/home/guangyuan/下载/mask_test.dat')
    # grad = grad * mask
    grad *= vpmax / abs(grad).max()
  

    return array2vector(grad)

def grad_precond_tv1(simu, optim, grad, forw, back,px,py,bx,by):
    ''' Gradient preconditioning
    '''
    nx = simu.model.nx
    nz = simu.model.nz
    vp = simu.model.vp
    vpmax = optim.vpmax
    marine_or_land = optim.marine_or_land
    grad_mute = optim.grad_mute
    grad_thred = optim.grad_thred
    grad_smooth = optim.grad_smooth
    grad = vector2array(grad, nx, nz)
    forw = vector2array(forw, nx, nz)
    back = vector2array(back, nx, nz)
    

    # plot forward and backward illumination
    plot_model2D(simu, forw.T, np.min(forw), 10 * np.median(forw), 'forw-illum-%03d' % optim.iter, 'my_seismic_cmap')
    plot_model2D(simu, back.T, np.min(back), 10 * np.median(back), 'back-illum-%03d' % optim.iter, 'my_seismic_cmap')

    # apply taper mask, land daming or water-layer masking
    if grad_mute > 0:
        grad *= grad_taper(nx, nz, tapersize = grad_mute, thred = grad_thred, marine_or_land=marine_or_land)
    
    m = simu.model.vp

    # # Method1
    # m_tmp = array2vector1(m)
    # # m_tmp = np.copy(m)
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
    # print(np.max(m_tmp))
    # print(inpa['tv'])
    # # print(m_tmp.shape)
    # _, grad_norm = regularization.cost_regularization(m_tmp,
    #                                               tv_properties=inpa['tv'],
    #                                               tikhonov_properties=None
    #                                               )
    # grad_norm = vector2array(grad_norm, nx, nz)
    # grad += grad_norm

    #Method2

    mx, my = grad_de(m)
    mx_tmp = smooth2d(mx - px + bx, 10)
    my_tmp = smooth2d(my - py + by, 10)
    tv_term = div(mx_tmp, my_tmp)
    tv_term[:,0:2] = 0
    tv_term[:,simu.model.nz-2:simu.model.nz] = 0
    tv_term = smooth2d(tv_term, 10)
    max_grad = np.max(np.abs(grad))
    max_tv = np.max(np.abs(tv_term)) + 1e-12
    scale_factor = max_grad / max_tv
    tv_scaled = tv_term * scale_factor
    # np.savetxt('/data/guangyuan/SWIT-1.0/examples/Review/TV-norm/Marm_TV2/model/tv_term.dat',tv_scaled)
    grad += simu.model.gamma  * tv_scaled


    #apply the inverse Hessian
    if min(nx, nz) > 40:      # set 40 grids in default
        span = 40
    else:                     # in case the grid number is less than 40
        span = int(min(nx, nz)/2)
    forw = smooth2d(forw, span)
    back = smooth2d(back, span)
    
    epsilon = 0.0001
    forw = forw / np.max(forw)
    back = back / np.max(back)
    precond = forw + back
    precond = precond / np.max(precond)
    precond[precond < epsilon] = epsilon
    grad = grad / precond

    # smooth the gradient
    if grad_smooth > 0:
        # exclude water-layer
        if marine_or_land in ['Marine', 'Offshore']: 
            grad[:,grad_mute:] = smooth2d(grad[:,grad_mute:], span=grad_smooth)
        # land gradient smooth
        else:
            grad = smooth2d(grad, span=grad_smooth)

    # gradient with respect to the velocity
    grad = - 2 * grad   #  / np.power(simu.model.vp, 3)
    grad *= vpmax / abs(grad).max()
  

    return array2vector(grad)


def grad_taper(nx, nz, tapersize=20, thred=0.05, marine_or_land='Marine'):
    ''' Gradient taper
    '''

    # for masking the water layer, use the zero threds
    if marine_or_land in ['Marine', 'Offshore']: 
        taper = np.ones((nx, nz))
        for ix in range(nx):
            taper[ix, :tapersize] = 0.0
            
    # for the land gradient damping, use the small threds
    else:
        H = scipy.signal.windows.hamming(tapersize*2)  # gaussian window
        H = H[tapersize:]
        taper = np.zeros((nx, nz))
        for ix in range(nx):
            taper[ix, :tapersize] = H
        taper = smooth2d(taper, span=tapersize//2)
        taper /= taper.max()
        taper *= (1 - thred)
        taper = - taper + 1
        taper = taper * taper      # taper^2 is better than taper^1

    return taper
