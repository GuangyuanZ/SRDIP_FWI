###############################################################################
#
# SWIT: Seismic Waveform Inversion Toolbox
#
# by Haipeng Li at USTC, haipengl@mail.ustc.edu.cn
# 
# June, 2021  
#
# Base module
#
###############################################################################

import os
from multiprocessing import cpu_count

import numpy as np
import scipy.sparse as sp
from tools import vector2array,array2vector1

class simulate(object):
    ''' simulate class defining all parameters for seismic wavefield modeling
    '''

    def __init__(self, model, source, receiver, system):
        ''' define all parameters here
        '''
        self.model = model
        self.source = source
        self.receiver = receiver
        self.system = system
        self.initialize()      # initialize


    def __check(self):
        ''' check all parameters are acceptable
        '''
        dx = self.model.dx
        dt = self.model.dt
        f0 = self.source.f0

        vpmin = np.min(self.model.vp)
        vpmax = np.max(self.model.vp)
        dt_req = np.sqrt(3.0/8.0) * dx / vpmax
        dx_req = vpmin / f0 / 10.
        f0_req = vpmin / dx / 10.

        # check path format
        if self.system.homepath[-1] != '/':
            self.system.homepath += '/'

        # Check the stable condition: 4-th order FD: dt <= sqrt(3/8) * dx / vmax
        if dt_req <= dt:
            raise ValueError('modeling stability: dt = %.4f ms > dt_required = %.4f ms: ' % (dt*1000, dt_req*1000))

        # Check the dispersion condition: dx <= vmin/(10*f0)
        if dx_req < dx:
            print('Warning: modeling dispersion, dx = %.2f m > dx_required =  %.2f m' %(dx, dx_req))
            print('Warning: modeling dispersion, f0 = %.2f Hz > f0_required = %.2f Hz' %(f0, f0_req))
  
        if (self.source.xz[:,0].min() < self.model.xx.min() or 
            self.source.xz[:,0].max() > self.model.xx.max() or  
            self.receiver.xz[:,0].min() < self.model.xx.min() or 
            self.receiver.xz[:,0].max() > self.model.xx.max()):
        
            raise ValueError('source or receiver coordinates are out of range')
        
        # Check the system and set the number of CPUs
        self.system.mpiproc = min([self.system.mpiproc, self.source.n, cpu_count() // 2])
        
        # Use one thread in calling scipy to do the filtering
        os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1


    def __builddir(self):
        ''' Build and clean the working folders
        '''

        homepath = self.system.homepath
        # clean the previous data if exists
        folder_list = [homepath+'data/syn',
                       homepath+'data/tempdata',
                       homepath+'parfile',
                       homepath+'outputs',
                       homepath+'figures',
                    ]
        for _, ifolder in enumerate(folder_list):
            if os.path.exists(ifolder):
                os.system('rm  -r %s' % ifolder)

        # creat the working folders
        folder_list = ['-p ' + homepath,                      # Home folder
                       homepath+'data',                      # Data folder
                       homepath+'data/obs/',                 # Observed data
                       homepath+'data/syn/',                 # Synthetic data
                       homepath+'data/tempdata/',            # Tempdata data
                       homepath+'parfile',                   # Parfile
                       homepath+'parfile/forward_parfile/',  # Forward parfile
                       homepath+'parfile/forward_source/',   # Forward source
                       homepath+'parfile/adjoint_parfile/',  # Adjoint parfile
                       homepath+'parfile/adjoint_source/',   # Adjoint source
                       homepath+'parfile/model/',            # Model
                       homepath+'outputs',                   # Outputs
                       homepath+'outputs/velocity/',         # Outputs: velocity
                       homepath+'outputs/gradient/',         # Outputs: gradient
                       homepath+'outputs/direction/',        # Outputs: direction
                       homepath+'outputs/LBFGS_memory/',     # Outputs: LBFGS_memory
                       homepath+'figures',                   # Figures
                       homepath+'figures/model/',            # Figures: gradient and velocity
                       homepath+'figures/waveform/',         # Figures: waveform
                       ]
        for _, ifolder in enumerate(folder_list):
            if not os.path.exists(ifolder):
                os.system('mkdir %s' % ifolder)


    def __screenprint(self):
        ''' Print information to screen
        '''

        # print the primary modeling parameters
        print('*****************************************************')
        print('\n        Seismic Waveform Inversion Toolbox         \n')
        print('*****************************************************\n')
        print('Forward modeling : nx, nz = %d, %d' %(self.model.nx, self.model.nz) )
        print('Forward modeling : dx = %.1f m'             %(self.model.dx))
        print('Forward modeling : dt = %.2f ms, %d steps'  %(self.model.dt * 1000, self.model.nt))
        print('Forward modeling : %d shots run in mpi, %d CPU available'%(self.system.mpiproc, cpu_count() // 2))


    def initialize(self):
        ''' Initialize the simulation
        '''

        self.__check()              # Check simulation parameter
        self.__builddir()           # Build and clean working folder
        self.__screenprint()        # Print to screen


class model(object):
    ''' model parameters for wavefiled simulation (2D or 3D)
    '''

    def __init__(self, nx, nz, dx, dt, nt, fs, pml, vp, rho,step_size,norm_iter,lmbda,tau,gamma,alphax,alphaz):

        # basic model
        self.step_size = step_size
        self.norm_iter = norm_iter
        self.alphax = alphax
        self.alphaz = alphaz
        self.lmbda = lmbda
        self.tau = tau
        self.gamma = gamma
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dt = dt
        self.nt = nt
        self.fs = fs
        self.pml = pml
        self.nx_pml = self.nx + self.pml * 2
        self.nz_pml = self.nz + self.pml * (2 - self.fs)
        # coordinate points (x, z), and the time array
        self.xx = np.arange(0, self.nx * self.dx, self.dx)
        self.zz = np.arange(0, self.nz * self.dx, self.dx)
        self.t  = np.linspace(0, self.dt*self.nt, num=self.nt, endpoint=False)
        # velocity, density, etc.
        self.rho = rho                                         # density in kg/m^3
        self.vp = vp                                           # p-wave velocity in m/s
        self.vpmax = vp.max()                                  # maximum vp
        self.vpmin = vp.min()                                  # mininum vp
        # output
        self.savesnap = 0
        self.savestep = 1


class source(object):
    ''' source parameters for forward wavefield simulation (2D or 3D)
    '''

    def __init__(self, f0, n, xz, wavelet):
        self.f0 = f0
        self.n = n
        self.xz = xz
        self.wavelet = wavelet


class receiver(object):
    ''' receiver parameters for forward wavefield simulation (2D or 3D)
    '''

    def __init__(self, n, xz):
        self.n = n
        self.xz = xz


class system(object):
    ''' system setting
    '''

    def __init__(self, homepath, mpiproc, figaspect=1):
        self.homepath = homepath
        self.mpiproc = mpiproc
        self.figaspect = figaspect


# optimize class
class optimize(object):
    ''' FWI optimation parameters
    '''
 
    def __init__(self, misfit_type, scheme, maxiter, step_length, vpmax, vpmin, marine_or_land,
                 grad_mute, grad_smooth,
                 fre_filter, fre_low, fre_high, 
                 mute_late_arrival, mute_late_window, normalize,
                 mute_offset_short, mute_offset_long, 
                 mute_offset_short_dis, mute_offset_long_dis,grad_mask=None):
        ''' Define all parameters
        '''

        # basic inversion parameters
        self.iter = 0
        self.misfit_type = misfit_type
        self.scheme = scheme
        self.maxiter = maxiter
        self.step_length = step_length
        self.vpmax = vpmax
        self.vpmin = vpmin
        self.marine_or_land = marine_or_land
        # gradient preconditioning
        self.grad_mute = grad_mute
        self.grad_smooth = grad_smooth
        self.grad_mask = grad_mask
        # data filter
        self.fre_filter = fre_filter
        self.fre_low = fre_low
        self.fre_high = fre_high
        # pick first break and mute later arrivals
        self.mute_late_arrival = mute_late_arrival
        self.mute_late_window = mute_late_window
        # data normalization
        self.normalize = normalize
        # data offset mute
        self.mute_offset_short = mute_offset_short
        self.mute_offset_long  = mute_offset_long
        self.mute_offset_short_dis = mute_offset_short_dis           # (units: m)
        self.mute_offset_long_dis  = mute_offset_long_dis            # (units: m)

        # set the taper for muting the gradient around the source 
        if self.marine_or_land.lower() in ['marine', 'offshore']:
            self.grad_thred = 0.0
        elif self.marine_or_land.lower() in ['land', 'onshore']:
            self.grad_thred = 0.001
        else:
            raise ValueError('not supported modeling marine_or_land: %s'%(self.marine_or_land))

        # initilize
        self.initialize()


    def __check(self):
        ''' check parameters.
        '''

        if self.scheme not in ['NLCG', 'LBFGS']:
            raise ValueError('not supported inversion scheme: %s' % self.scheme)

        if self.misfit_type not in ['Waveform', 'Envelope', 'Traveltime', 'Globalcorrelation','RTM','Globalcorrelation_tv','Globalcorrelation_tv1']:
            raise ValueError('not supported misfit function: %s' % self.misfit_type)

        if ('Max-Trace' not in self.normalize and 
            'L1-Event' not in self.normalize and 
            'L2-Event' not in self.normalize and 
            'L1-Trace' not in self.normalize and 
            'L2-Trace' not in self.normalize and 
            'None' not in self.normalize):
            raise ValueError('not supported normalization:', self.normalize)

        if self.fre_filter not in ['Lowpass', 'Bandpass', 'Highpass', 'None']:
            raise ValueError('not supported frequency filter: %s' %self.fre_filter)

        if self.vpmax < self.vpmin:
            raise ValueError('vpmax=%f m/s is less than vpmin=%f m/s\n' %(self.vpmax, self.vpmin))

        if self.fre_low > self.fre_high:
            raise ValueError('fre_low > fre_high')

        if self.mute_offset_short_dis > self.mute_offset_long_dis:
            raise ValueError('mute_offset_short_dis > mute_offset_long_dis')


    def __screenprint(self):
        ''' Print information to screen.
        '''
        # basic inversion parameter
        print('Inversion scheme : %s' % self.scheme)
        print('Inversion misfit : %s' % self.misfit_type)
        print('Inversion maxiter: %d' % self.maxiter)
        print('Inversion step   : %.3f'      % self.step_length)
        print('Inversion vpmin  : %.1f m/s'  % self.vpmin)
        print('Inversion vpmax  : %.1f m/s'  % self.vpmax)
        print('Gradient  mute   : %d grids on top' % self.grad_mute)
        print('Gradient  smooth : %d grids Gaussian smooth' % self.grad_smooth)

        # filtering 
        if self.fre_filter in ['None']:
            print('Data processing  : no filtering')
        elif self.fre_filter in ['Bandpass']:
            print('Data processing  : %s, %.2f ~ %.2f Hz' %(self.fre_filter, self.fre_low, self.fre_high))
        elif self.fre_filter in ['Lowpass']:
            print('Data processing  : %s, < %.2f Hz' %(self.fre_filter, self.fre_low))
        elif self.fre_filter in ['Highpass']:
            print('Data processing  : %s, > %.2f Hz' %(self.fre_filter, self.fre_high))

        # pick and mute
        if self.mute_late_arrival:
            print('Data processing  : time window, %.2f s after the first break' % self.mute_late_window)
        else:
            print('Data processing  : no time window')

        # mute offset
        if self.mute_offset_short:
                print('Data processing  : mute short offset, %.1f m' % self.mute_offset_short_dis)
        else:
            print('Data processing  : mute short offset, none')
        if self.mute_offset_long:
                print('Data processing  : mute long offset, %.1f m' % self.mute_offset_long_dis)
        else:
            print('Data processing  : mute long offset, none')
        
        # normalize
        if self.normalize in ['None']:
            print('Data processing  : no normalization (keep AVO effect)')
        else:
            #print('Data processing  : normalization, %s' % (', '.join(self.normalize)))
            print('Data processing  : normalization, %s' % (self.normalize))
        print('Data processing  : OMP Threads = %s' %os.environ["OMP_NUM_THREADS"])
        
        print('\nsee more in json-parameter files under parfile folder\n')
        
        print('*****************************************************\n')


    def initialize(self):
        ''' Initialize the optimazation.
        '''

        self.__check()              # Check simulation parameter
        self.__screenprint()





class Regularization:
    """
    Regularization Prepares tools for regularizing FWI problem

    Parameters
    ----------
    nx : int scalar
        Number of samples in x-direction
    nz : int scalar
        Number of samples in z-direction
    dx : float scalar
        Spatial sampling rate in x-direction
    dz : float scalar
        Spatial sampling rate in z-direction
    """
    def __init__(self, nx, nz, dx, dz):

        self.idx = 1 / dx
        self.idz = 1 / dz

        self.idx2 = 1 / (dx * dx)
        self.idz2 = 1 / (dz * dz)

        self.dx = dx
        self.dz = dz

        self.nx = nx
        self.nz = nz
        self.n_elements = nz * nx

        self.Bx2, self.Bz2 = derivative(nx, nz, dx, dz, 2)
        self.Bx1, self.Bz1 = derivative(nx, nz, dx, dz, 1)

        self.D2 = self.Bx2.T @ self.Bx2 + self.Bz2.T @ self.Bz2

    def cost_regularization(self, x0,
                            tv_properties=None,
                            tikhonov_properties=None,
                            tikhonov0_properties=None):
        x = np.copy(x0)
        rms = 0
        grad = np.zeros(x.shape)
        # Because we may provide the properties but ask for regularization in some special frequencies
        if tv_properties:
            f_tv, g_tv = self.tv(x, 1e-7, alpha_z=tv_properties['az'], alpha_x=tv_properties['ax'])

            rms += tv_properties['lambda_weight'] * f_tv
            grad += tv_properties['lambda_weight'] * g_tv


        if tikhonov_properties:
            f_tikh, g_tikh = self.tikhonov(x, alpha_z=tikhonov_properties['az'], alpha_x=tikhonov_properties['ax'])

            rms += tikhonov_properties['lambda_weight'] * f_tikh
            grad += tikhonov_properties['lambda_weight'] * g_tikh

        if tikhonov0_properties:
            f_tikh, g_tikh = self.tikhonov_0(x)

            rms += tikhonov0_properties['lambda_weight'] * f_tikh
            grad += tikhonov0_properties['lambda_weight']  # No gradient

        return rms, grad



    def tv(self, x0, eps, alpha_z, alpha_x):
        """
        Parameters
        ----------
        x0 : float
            Data
        eps : scalar float
            small value for make it deffrintiable at zero
        alpha_z : scalar float
            coefficient of Dz
        alpha_x : scalar float
            coefficient of Dx
            
        Returns
        -------
        rms : scalar float
            loss
        grad : scalar float
            Gradient of loss w.r.t. model parameters
        """
        x = np.copy(x0)

        ln = (self.nx*self.nz)
        ln_x = len(x)
        n = ln_x//ln  # NOT self.n_parameter

        x1 = np.zeros(ln_x,)
        for i in range(n):
            mx1 = self.Bx1 @ x[i*ln:(i+1)*ln]
            mz1 = self.Bz1 @ x[i*ln:(i+1)*ln]

            # To ignore the effect of sharp change after first 15 samples
            mz1[:17*self.nx] = 0.0

            x1[i*ln:(i+1)*ln] = alpha_x * mx1 + alpha_z * mz1
        rms, grad = self.l1(x1, eps)
        grad_tmp = vector2array(grad,self.nz,self.nx)
        grad_new = array2vector1(grad_tmp)
        return rms, grad_new

    @staticmethod
    def l1(x0, eps=1e-7):
        x = np.copy(x0)

        x1 = np.copy(x)
        len_x = len(x)
        x1[np.abs(x1) <= eps] = eps
        w_1 = np.sqrt(np.abs(x1))
        w = sp.spdiags(1/w_1, diags=0, m=len_x,  n=len_x)

        wx = w@x
        rms = wx.T @ wx  # np.sum(np.abs(x))

        grad = 2 * wx.T @ w

        return rms, grad

    @staticmethod
    def l2(x0):
        x = np.copy(x0)
        rms = x.T @ x

        return rms

    def tikhonov(self, x0, alpha_z, alpha_x):
        """
        A method to implement Tikhonov regularization with order of 2

        Parameters
        ----------
            x0 : 1D ndarray
                Data
            alpha_z : float
                coefficient of Dz
            alpha_x : float
                coefficient of Dx

        Returns
        -------
        rms : scalar float
            loss
        grad : scalar float
            Gradient of loss w.r.t. model parameters
        """
        x = np.copy(x0)
        ln = (self.nx * self.nz)
        ln_x = len(x)
        n = ln_x // ln  # NOT self.n_parameter

        x1 = np.zeros(ln_x,)
        grad = np.zeros(x.shape)
        for i in range(n):
            m = np.copy(x[i * ln:(i + 1) * ln])
            mx1 = self.Bx1 @ m
            mz1 = self.Bz1 @ m

            # To ignore the effect of sharp change after first 15 samples
            mz1[:17 * self.nx] = 0.0

            x1[i * ln:(i + 1) * ln] = alpha_x * mx1 + alpha_z * mz1
            grad[i * ln:(i + 1) * ln] = alpha_x * (self.Bx1.T @ self.Bx1) @ m +\
                                        alpha_z * (self.Bz1.T @ self.Bz1) @ m

        rms = self.l2(x1)

        return rms, grad

    def tikhonov_0(self, x0):
        """
        A method to implement Tikhonov regularization with order of 0

        Parameters
        ----------
            x0 : 1D ndarray
                Data
            alpha_z : float
                coefficient of Dz
            alpha_x : float
                coefficient of Dx

        Returns
        -------
            rms : float
                error
            grad : 1D ndarray
                gradient of the regularization
        """
        x = np.copy(x0)

        ln = (self.nx * self.nz)
        ln_x = len(x)
        n = ln_x // ln  # NOT self.n_parameter

        x1 = np.zeros(ln_x, )
        for i in range(n):
            mx1 = x[i * ln:(i + 1) * ln]

            x1[i * ln:(i + 1) * ln] = mx1
        rms, grad = self.l1(x1)
        
        return rms, grad

    def parameter_relation(self, m0, models, k0, kend, freq):
        """
        parameter_relation considers regularization for the
        relation between parameters.


        Parameters
        ----------
        m0 : ndarray
            Vector of parameters
        models : dict
            A dictionary containing couple of dictionaries which includes a numpy
            polyfit model and regularization parameter.
        k0 : int
            Index of the first parameter in m0
        kend : int
            Index of the last parameter in m0


        Returns
        -------
        rms : float
            rms of regularization
        grad: ndarray
            Vector of gradient od the regularization
        """
        rms = 0
        grad = np.zeros(m0.shape)

        for param in models:
            par = [char for char in param]
            model = models[param]['model']
            lam = models[param]['lam']

            desired_freq  = models[param]['freqs']

            if freq not in np.array(desired_freq).reshape(-1):
                # has to be written like that to work eaither if freq is given as int or list
                return 0.0, grad

            par_int = np.int32(par)
            if par_int[1] in np.arange(k0+1, kend+1):
                pre21 = model(m0[(par_int[0]-1) * self.n_elements:par_int[0] * self.n_elements])

                dm21 = m0[(par_int[1]-1)  * self.n_elements:par_int[1] * self.n_elements] - pre21

                rms += 0.5 * lam * np.dot(dm21.T, dm21)
                grad[(par_int[1]-1)  * self.n_elements: par_int[1] *self.n_elements] = gaussian_filter(lam * dm21 * 1, 1)

        return rms, grad

    def priori_regularization(self, m0, regularization_dict, k0, kend, freq):
        """
        priori_regularization consider the priori information regularization.


        Parameters
        ----------
        m0 : float
            Vector of parameters
        regularization_dict : dict
            A dictionary containing couple of priori model and regularization hyperparameter
        k0 : int
            Index of the first parameter in m0
        kend : int
            Index of the last parameter in m0

        Returns
        -------
        rms : float
            rms of regularization
        grad: ndarray
            Vector of gradient od the regularization

        References
        ----------
        Asnaashari et al., 2013, Regularized seismic full waveform inversion with prior model information, Geophysics, 78(2), R25-R36, eq. 5.
        """
        if regularization_dict is None:
            return 0.0, np.zeros(m0.shape, np.float64)

        m0 = np.copy(m0[: kend * self.n_elements])
        mp = np.zeros(m0.shape)
        desired_freq  = regularization_dict['freqs']

        if freq not in np.array(desired_freq).reshape(-1):
            return 0.0, np.zeros(m0.shape, np.float64)

        lam = regularization_dict['lam']

        mp_dict = regularization_dict['mp']

        for i in range(kend - k0):
            mp[i * self.n_elements: (i + 1) * self.n_elements] = mp_dict[[*mp_dict][k0 + i]].reshape(-1)

        ii = jj = np.arange((kend - k0) * self.n_elements)
        v = np.ones((ii.shape)) / np.var(mp)

        W = sp.csr_matrix((v, (ii, jj)))

        diff = (m0 - mp).reshape(-1, 1)

        rms = lam * 0.5 * (diff.T @ W) @ diff

        grad = lam * W.T @ diff

        return rms.item(), grad.reshape(-1)
    

def derivative(nx, nz, dx, dz, order):
    """
    Compute spatial derivative operators for grid _cells_
    For 1st order: \n
    \tforward operator is (u_{i+1} - u_i)/dx\n
    \tcentered operator is (u_{i+1} - u_{i-1})/(2dx)\n
    \tbackward operator is (u_i - u_{i-1})/dx \n

    For 2nd order: \n
    \tforward operator is (u_i - 2u_{i+1} + u_{i+2})/dx^2 \n
    \tcentered operator is (u_{i-1} - 2u_i + u_{i+1})/dx^2 \n
    \tbackward operator is (u_{i-2} - 2u_{i-1} + u_i)/dx^2 \n

    Parameters
    ----------
        nx : int
            Number of samples in X-direction
        nz : int
            Number of samples in Z-direction
        dx : float
            Samplikng rate in X-direction
        dz : float
            Samplikng rate in Z-direction
        order : int
            Order of derivative

    Returns
    -------
        Dx : Dispersed matrix
            Derivative matrix in X-direction
        Dz : Dispersed matrix
            Derivative matrix in Z-direction

    """

    if order == 1:

        # forward operator is (u_{i+1} - u_i)/dx
        # centered operator is (u_{i+1} - u_{i-1})/(2dx)
        # backward operator is (u_i - u_{i-1})/dx

        idx = 1 / dx
        idz = 1 / dz

        i = np.kron(np.arange(nz * nx), np.ones((2,), dtype=np.int64))
        j = np.zeros((nz * nx * 2,), dtype=np.int64)
        v = np.zeros((nz * nx * 2,))

        jj = np.vstack((np.arange(nx), nx + np.arange(nx))).T
        jj = jj.flatten()
        j[:2 * nx] = jj

        vd = idx * np.tile(np.array([-1, 1]), (nx,))
        v[:2 * nx] = vd

        jj = np.vstack((-nx + np.arange(nx), nx + np.arange(nx))).T
        jj = jj.flatten()
        for n in range(1, nz - 1):
            j[n * 2 * nx:(n + 1) * 2 * nx] = n * nx + jj
            v[n * 2 * nx:(n + 1) * 2 * nx] = 0.5 * vd

        jj = np.vstack((-nx + np.arange(nx), np.arange(nx))).T
        jj = jj.flatten()
        j[(nz - 1) * 2 * nx:nz * 2 * nx] = (nz - 1) * nx + jj
        v[(nz - 1) * 2 * nx:nz * 2 * nx] = vd

        Dz = sp.csr_matrix((v, (i, j)))

        jj = np.vstack((np.hstack((0, np.arange(nx - 1))),
                        np.hstack((np.arange(1, nx), nx - 1)))).T
        jj = jj.flatten()
        vd = idz * np.hstack((np.array([-1, 1]),
                              np.tile(np.array([-0.5, 0.5]), (nx - 2,)),
                              np.array([-1, 1])))

        for n in range(nz):
            j[n * 2 * nx:(n + 1) * 2 * nx] = n * nx + jj
            v[n * 2 * nx:(n + 1) * 2 * nx] = vd

        Dx = sp.csr_matrix((v, (i, j)))
    else:  # 2nd order
        idx2 = 1 / (dx * dx)
        idz2 = 1 / (dz * dz)

        i = np.kron(np.arange(nz * nx), np.ones((3,), dtype=np.int64))
        j = np.zeros((nz * nx * 3,), dtype=np.int64)
        v = np.zeros((nz * nx * 3,))

        jj = np.vstack((np.arange(nx), nx + np.arange(nx),
                        2 * nx + np.arange(nx))).T
        jj = jj.flatten()
        j[:3 * nx] = jj
        vd = idx2 * np.tile(np.array([1.0, -2.0, 1.0]), (nx,))
        v[:3 * nx] = vd

        for n in range(1, nz - 1):
            j[n * 3 * nx:(n + 1) * 3 * nx] = (n - 1) * nx + jj
            v[n * 3 * nx:(n + 1) * 3 * nx] = vd

        j[(nz - 1) * 3 * nx:nz * 3 * nx] = (nz - 3) * nx + jj
        v[(nz - 1) * 3 * nx:nz * 3 * nx] = vd

        Dz = sp.csr_matrix((v, (i, j)))

        jj = np.vstack((np.hstack((0, np.arange(nx - 2), nx - 3)),
                        np.hstack((1, np.arange(1, nx - 1), nx - 2)),
                        np.hstack((2, np.arange(2, nx), nx - 1)))).T
        jj = jj.flatten()
        vd = vd * idz2 / idx2

        for n in range(nz):
            j[n * 3 * nx:(n + 1) * 3 * nx] = n * nx + jj
            v[n * 3 * nx:(n + 1) * 3 * nx] = vd

        Dx = sp.csr_matrix((v, (i, j)))

    return Dx, Dz
