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
from tools import array2vector, smooth2d, vector2array
from base import Regularization

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
    plot_model2D(simu, forw.T, np.min(forw), 10 * np.median(forw), 'forw-illum-%03d' % optim.iter, 'my_seismic_cmap')
    plot_model2D(simu, back.T, np.min(back), 10 * np.median(back), 'back-illum-%03d' % optim.iter, 'my_seismic_cmap')
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

def grad_precond_tv(simu, optim, grad, forw, back):
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
    m_tmp = array2vector(m)
    inpa = {}
    inpa['tv'] = {
    'az': simu.model.alphaz,  # regularizatin in the z-direction
    'ax': simu.model.alphax, # regularizatin in the x-direction
    'lambda_weight': simu.model.lmbda # regularizatin weight
    }

    regularization = Regularization(simu.model.nx, # number of samples in the x-direction
                                simu.model.nz, # number of samples in the z-direction
                                simu.model.dx, # Spatial sampling rate in the x-direction
                                simu.model.dx # Spatial sampling rate in the z-direction
                                 )
    print(np.max(m_tmp))
    print(inpa['tv'])
    _, grad_norm = regularization.cost_regularization(m_tmp,
                                                  tv_properties=inpa['tv'],
                                                  tikhonov_properties=None
                                                  )
    grad_norm = vector2array(grad_norm, nx, nz)
    grad += grad_norm

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
    # return grad_norm


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
