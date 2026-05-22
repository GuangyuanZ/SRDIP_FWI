###############################################################################
#
# Total variation for Seismic Waveform Inversion Toolbox
#
# by Guangyuan Zou at USTC, guangyuan@mail.ustc.edu.cn
# 
# March, 2025 
#
# TV operator
#
###############################################################################

import numpy as np


def div_op(f):

    Nx, Ny = f.shape[:2]
    div_f = np.zeros((Nx, Ny))

    div_f[1:-1,:] += f[1:-1,:,0] - f[:-2,:,0]
    div_f[0, :] += f[0, :, 0]
    div_f[-1, :] += -f[-2, :, 0]
    
    div_f[:, 1:-1] += f[:, 1:-1, 1] - f[:, :-2, 1]
    div_f[:, 0] += f[:, 0, 1]
    div_f[:, -1] += -f[:, -2, 1]

    return div_f

def abs_op(f):

    Nx, Ny = f.shape[:2]
    abs_f = np.zeros((Nx, Ny))
    abs_f = np.sqrt(f[:, :, 0]**2 + f[:, :, 1]**2)

    return abs_f



def nabla_op(f):

    Nx, Ny = f.shape[:2]
    nabla_f = np.zeros((Nx, Ny, 2))

    nabla_f[:-1, :, 0] = f[1:, :] - f[:-1, :]
    nabla_f[:, :-1, 1] = f[:, 1:] - f[:, :-1]
    
    return nabla_f


def TV_projection(g, lbda, iterMax=1000, tau=1/8):
    
    Nx, Ny = g.shape
    p0 = np.zeros((Nx, Ny, 2))
    p1 = np.zeros((Nx, Ny, 2))

    for iter in range(iterMax):
        temp = div_op(p0) - g / lbda
        nabla_temp = nabla_op(temp)
        abs_nabla_temp = abs_op(nabla_temp)
        p1 = (p0 + tau * nabla_temp)  / (1 + tau*abs_nabla_temp)[..., np.newaxis]  
        p0 = np.copy(p1)
    
    pi_g = div_op(p1)
    u = g - lbda * pi_g
    return u

def TV_projection(g, lam, iterMax=100, tau=0.1, sigma=None):
    """
    PDHG solver for the ROF denoising model.

    Parameters
    ----------
    g : ndarray
        Noisy image, shape (Nx, Ny)
    lam : float
        Regularization parameter
    iterMax : int
        Number of iterations
    tau : float
        Primal step size
    sigma : float or None
        Dual step size. If None, uses 1 / (tau * 8)

    Returns
    -------
    u0 : ndarray
        Denoised image
    """

    if sigma is None:
        sigma = 1 / (tau * 8)

    Nx, Ny = g.shape

    u0 = np.ones((Nx, Ny))
    p0 = np.ones((Nx, Ny, 2))

    for _ in range(iterMax):

        # primal update
        u0_bar = u0 + tau * div_op(p0)
        u1 = (u0_bar + tau * g) / (1 + tau)

        # dual update
        p0_bar = p0 + sigma * nabla_op(u1)

        norm = np.sqrt(
            p0_bar[:, :, 0]**2 + p0_bar[:, :, 1]**2
        )

        p1 = p0_bar / np.maximum(1, norm[..., np.newaxis] / lam)

        # update variables
        u0 = u1.copy()
        p0 = p1.copy()

    return u0

def TV_projection_dual(g, lbda, iterMax=1000, tau=1/8):
    
    Nx, Ny = g.shape
    p0 = np.zeros((Nx, Ny, 2))
    p1 = np.zeros((Nx, Ny, 2))

    for iter in range(iterMax):
        temp = div_op(p0) - g / lbda
        nabla_temp = nabla_op(temp)
        abs_nabla_temp = abs_op(nabla_temp)
        p1 = (p0 + tau * nabla_temp)  / (1 + tau*abs_nabla_temp)[..., np.newaxis]  
        p0 = np.copy(p1)
    
    pi_g = div_op(p1)
    # u = g - lbda * pi_g
    return pi_g
    

def PDHG_ROF(g, lambd, iterMax=100, tau=0.1, sigma=None):
    """Primal-Dual Hybrid Gradient (PDHG) 方法求解 ROF 问题
    
    Parameters
    ----------
    g : ndarray
        Noisy image, shape (Nx, Ny)
    lambd : float
        simu.model.lambda
    iterMax : int
        simu.model.norm_iter
    tau : float
        simu.model.tau
    """
    if sigma is None:
        sigma = 1 / (tau * 8)
    
    Nx, Ny = g.shape
    u0 = np.ones((Nx, Ny))
    p0 = np.ones((Nx, Ny, 2))

    for _ in range(iterMax):
        u0_bar = u0 + tau * div_op(p0)
        u1 = (u0_bar + tau * g) / (1 + tau)

        p0_bar = p0 + sigma * nabla_op(u1)
        norm_p0_bar = np.maximum(1, lambd**-1 * np.sqrt(p0_bar[..., 0]**2 + p0_bar[..., 1]**2))
        p1 = p0_bar / norm_p0_bar[..., np.newaxis]  # 保持形状一致

        u0 = u1.copy()
        p0 = p1.copy()

    return u0