###############################################################################
#
# SWIT: Seismic Waveform Inversion Toolbox
#
# by Haipeng Li at USTC, haipengl@mail.ustc.edu.cn
# 
# June, 2021  
#
# Forward and adjoint Solver module
#
###############################################################################

# SWIT Model
import numpy as np
import torch
import math
import os
from misfit import misfit
from preprocess import process_workflow
from solver import adjoint, forward
from tools import array2vector
from inversion import optimize_init
from tools import array2vector, save_inv_scheme,smooth2d
from plot import plot_inv_scheme
# ADFWI Model
from ADFWI.model       import *
from ADFWI.dip import *
from ADFWI.utils       import *
from tqdm import tqdm
import time
# from ADFWI import DIP_CNN
import warnings
warnings.filterwarnings("ignore")
import pkg_resources as pkg
import os, random
from torch.autograd.functional import hessian

def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    return result

def set_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    # if deterministic and check_version(torch.__version__, '1.11.1'):  # https://github.com/ultralytics/yolov5/pull/8213
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def Unet_prepare(simu,vp_init,layer_num,load_pretrained = True):

    save_path = simu.system.homepath + 'model/DIP_model.pt'
    load_path = simu.system.homepath + 'model/DIP_model.pt'

    # -----------------------------------
    #    Model parameters
    # -----------------------------------

    nz = simu.model.nz
    nx = simu.model.nx
    vp_true = simu.model.vp.T
    device = "cuda:0"
    base_channel = 64
    dtype = torch.float32

    model_shape = [nz,nx]
    DIP_model = DIP_Unet(model_shape,
                        n_layers= layer_num,
                        vmin=vp_true.min()/1000,
                        vmax=vp_true.max()/1000,
                        base_channel=base_channel,
                        device=device)
    DIP_model.to(device)

    print('Prepare Unet model')
    # -----------------------------------
    #     Pretrain DIP model
    # -----------------------------------
    pretrain        = True
    # load_pretrained = True
    if pretrain:
        if load_pretrained:
            # load the model parameters
            DIP_model.load_state_dict(torch.load(load_path))
        else:
            lr          = 0.005
            iteration   = 10000
            step_size   = 2000
            # iteration = 5000
            # step_size = 1
            gamma       = 0.5
            optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
            vp_init = numpy2tensor(vp_init,dtype=dtype).to(device)
            pbar = tqdm(range(iteration+1))
            for i in pbar:  
                vp_nn = DIP_model()
                loss = torch.sqrt(torch.sum((vp_nn - vp_init)**2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_description(f'Pretrain Iter:{i}, Misfit:{loss.cpu().detach().numpy()}')
            torch.save(DIP_model.state_dict(),os.path.join(save_path))

    return DIP_model

def CNN_prepare(simu,optim,vp_init,layer_num,load_pretrained = True):

    save_path = simu.system.homepath + 'model/DIP_model.pt'
    load_path = simu.system.homepath + 'model/DIP_model.pt'
    # set_seeds(seed=1234, deterministic=True)
    # -----------------------------------
    #    Model parameters
    # -----------------------------------

    nz = simu.model.nz
    nx = simu.model.nx
    # vp_true = simu.model.vp.T
    vmin = optim.vpmin
    vmax = optim.vpmax

    device = "cuda:0"
    # device = 'cpu'
    base_channel = 64
    dtype = torch.float32
    print('Check vmin Value: ',vmin)
    print('Check vmax Value: ',vmax)
    # loss_pretrain = np.zeros(10000)
    model_shape = [nz,nx]
    DIP_model = DIP_CNN(model_shape,
            in_channels=[32,32],
            vmin=vmin/1000,
            vmax=vmax/1000,
            # vmin=vmin,
            # vmax=vmax,
            device=device)
    DIP_model.to(device)

    print('Prepare CNN model')
    # -----------------------------------
    #     Pretrain DIP model
    # -----------------------------------
    pretrain        = True
    # load_pretrained = True
    start_time = time.time()
    print('Start time: ', start_time)
    if pretrain:
        if load_pretrained:
            # load the model parameters
            DIP_model.load_state_dict(torch.load(load_path))
        else:
            lr          = 0.0005
            # lr = 0.0001
            # lr = 0.001
            iteration   = 15000
            step_size   = 1000
            # iteration = 5000
            # step_size = 1
            # gamma       = 0.5
            gamma = 0.6
            optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
            vp_init = numpy2tensor(vp_init,dtype=dtype).to(device)
            pbar = tqdm(range(iteration+1))
            for i in pbar:  
                vp_nn,_ = DIP_model()
                loss = torch.sqrt(torch.sum((vp_nn - vp_init)**2))
                # loss_pretrain[i] = loss.cpu().detach().numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_description(f'Pretrain Iter:{i}, Misfit:{loss.cpu().detach().numpy()}')
            torch.save(DIP_model.state_dict(),os.path.join(save_path))
            # np.savetxt('/home/guangyuan/桌面/SWIT-1.0/examples/case9_ti_fwi_CNN/model/pre_train_loss.dat',loss_pretrain)
    print('Pretrain time: ',time.time() - start_time)
    return DIP_model

def CNN_adv_prepare(simu,optim,vp_init,layer_num,load_pretrained = True):

    save_path = simu.system.homepath + 'model/DIP_model.pt'
    load_path = simu.system.homepath + 'model/DIP_model.pt'
    # set_seeds(seed=1234, deterministic=True)
    # -----------------------------------
    #    Model parameters
    # -----------------------------------

    nz = simu.model.nz
    nx = simu.model.nx
    # vp_true = simu.model.vp.T
    vmin = optim.vpmin
    vmax = optim.vpmax

    device = "cuda:0"
    # device = 'cpu'
    base_channel = 64
    dtype = torch.float32
    print('Check vmin Value: ',vmin)
    print('Check vmax Value: ',vmax)
    # loss_pretrain = np.zeros(10000)
    model_shape = [nz,nx]
    DIP_model = DIP_CNN_advanced(model_shape,
            in_channels=[16,16],
            vmin=vmin/1000,
            vmax=vmax/1000,
            # vmin=vmin,
            # vmax=vmax,
            device=device)
    DIP_model.to(device)

    print('Prepare CNN model')
    # -----------------------------------
    #     Pretrain DIP model
    # -----------------------------------
    pretrain        = True
    # load_pretrained = True
    start_time = time.time()
    print('Start time: ', start_time)
    if pretrain:
        if load_pretrained:
            # load the model parameters
            DIP_model.load_state_dict(torch.load(load_path))
        else:
            lr          = 0.005
            # lr = 0.0005
            iteration   = 10000
            step_size   = 1000
            # iteration = 5000
            # step_size = 1
            gamma       = 0.5
            optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
            vp_init = numpy2tensor(vp_init,dtype=dtype).to(device)
            pbar = tqdm(range(iteration+1))
            for i in pbar:  
                vp_nn = DIP_model()
                loss = torch.sqrt(torch.sum((vp_nn - vp_init)**2))
                # loss_pretrain[i] = loss.cpu().detach().numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_description(f'Pretrain Iter:{i}, Misfit:{loss.cpu().detach().numpy()}')
            torch.save(DIP_model.state_dict(),os.path.join(save_path))
            # np.savetxt('/home/guangyuan/桌面/SWIT-1.0/examples/case9_ti_fwi_CNN/model/pre_train_loss.dat',loss_pretrain)
    print('Pretrain time: ',time.time() - start_time)
    return DIP_model


class Unet_update:

    def __init__(self,simu,DIP_model,adj_grad,lr,optimizer,scheduler):

        self.simu = simu
        self.DIP_model = DIP_model
        self.adj_grad = adj_grad
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
    

    def inversion(self):
        output = self.DIP_model()

        def grad_post_process(grad):
            grad = grad.cpu().detach().numpy()
            with torch.no_grad():
                grad = grad*self.adj_grad
                grad  = numpy2tensor(grad,dtype=torch.float32).to("cuda:0")
            return grad
        
        # Adam optimizor
        output.register_hook(grad_post_process)
        loss = torch.sum(output)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # def closure():
        #     self.optimizer.zero_grad()
        #     output = self.DIP_model()
        #     output.register_hook(grad_post_process)
        #     loss = torch.sum(output)
        #     loss.backward()
        #     return loss
        
        # self.optimizer.step(closure)

class CNN_update:

    def __init__(self,simu,DIP_model,adj_grad,lr,optimizer,scheduler):

        self.simu = simu
        self.DIP_model = DIP_model
        self.adj_grad = adj_grad
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
    

    def inversion(self):
    
        # def grad_post_process(grad):
        #     grad = grad.cpu().detach().numpy()
        #     with torch.no_grad():
        #         grad = grad*(self.adj_grad+1e-6)
        #         grad  = numpy2tensor(grad,dtype=torch.float32).to("cuda:0")
        #     return grad
        def grad_post_process(grad):
            if isinstance(self.adj_grad, np.ndarray):
                adj_grad_tensor = torch.from_numpy(self.adj_grad).to(grad.device, dtype=grad.dtype)
            else:
                adj_grad_tensor = self.adj_grad.to(grad.device)
            adj_grad_tensor = torch.ones_like(grad) 
            return grad * (adj_grad_tensor + 1e-6)
            
        
                
        def zero_small_gradients(model, threshold=1e-6):
            for param in model.parameters():
                if param.grad is not None:
                    param.grad[torch.abs(param.grad) < threshold] = 0
        

        
        # # return grad_post_process
        # Adam optimizor
        output,_ = self.DIP_model()

        output.register_hook(grad_post_process)
        loss = torch.sum(output)
        # np.save("Hessian.npy", H.detach().cpu().numpy())
        self.optimizer.zero_grad()
        loss.backward()

        for name, param in self.DIP_model.named_parameters():
            if param.grad is not None:
                fname = '/data/guangyuan/SWIT-1.0/examples/Hessian_DIP/model/' + f"{name}" + ".npy"
                grad = param.grad.cpu().numpy()
                np.save(os.path.join(fname), grad)

        # torch.nn.utils.clip_grad_norm_(self.DIP_model.parameters(), max_norm=1.0)
        # max_grad = max(param.grad.abs().max().item() for param in self.DIP_model.parameters() if param.grad is not None)
        # print(max_grad)
        # zero_small_gradients(self.DIP_model, threshold=0.0005*max_grad) 
        self.optimizer.step()
        self.scheduler.step()

        # # LBFGS
        # def closure():
        #     self.optimizer.zero_grad()
        #     output = self.DIP_model()
        #     output.register_hook(grad_post_process)
        #     loss = torch.sum(output)
        #     loss.backward()
        #     return loss
        # torch.nn.utils.clip_grad_norm_(self.DIP_model.parameters(), max_norm=1.0)
        # self.optimizer.step(closure)



def DIP_inversion(simu, optim,layer_num,inv_model,load_pretrained = True):
    ''' inversion workflow
    '''
    # Prepare Unet
    vp_init = inv_model['vp'] 
    # DIP_model = Unet_prepare(simu,vp_init.T,layer_num,load_pretrained)
    DIP_model = CNN_prepare(simu,optim,vp_init.T,layer_num,load_pretrained)
    # DIP_model = CNN_adv_prepare(simu,optim,vp_init.T,layer_num,load_pretrained)
    
    # set the initial model
    simu.model.vp = inv_model['vp'] 
    simu.model.rho = inv_model['rho'] 

    # initialize the inversion
    inv_scheme = optimize_init(simu, optim)
    inv_scheme['v_now'] = array2vector(simu.model.vp)
    inv_scheme['v_old'] = array2vector(simu.model.vp)
    pbar_epoch = tqdm(range(0,optim.maxiter),colour='green')
    # while optim.iter < optim.maxiter:
    print('Start inversion')

    ######################################### Adam optim #########################################
    # lr = 0.0004
    # lr = 0.0001
    lr = 0.00004
    # lr  = 0.00001
    ### Adam ####
    optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=25,gamma=0.8)


    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.8)

    # ## LBFGS ####
    # # optimizer = torch.optim.LBFGS(DIP_model.parameters(), line_search_fn='strong_wolfe')
    # optimizer = torch.optim.LBFGS(DIP_model.parameters(), lr = lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5,last_epoch=-1)
    # # SGD ####
    # optimizer = torch.optim.SGD(DIP_model.parameters(), lr= lr,momentum=0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.8)
    ### RMSprop ####
    # optimizer = torch.optim.RMSprop(DIP_model.parameters(), lr= lr,weight_decay=1e-8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.8)

    start_time = time.time()
    print('Start time: ', start_time)
    for i in pbar_epoch:
        optim.iter += 1
        # print('\n-----------  iteration %d  -----------\n'%optim.iter)

        # synthetic data from the current model
        forward(simu, simu_type='syn', savesnap=1)
        
        # process the synthetic data
        process_workflow(simu, optim, simu_type='syn')

        # evaluate the misfit
        inv_scheme['f_now'] = misfit(simu, optim.misfit_type)   
        print('Misfit:',inv_scheme['f_now'])

        # evaluate the gradient (with preconditioning and scaling)
        inv_scheme['g_now'] = adjoint(simu, optim)
        adj_grad = inv_scheme['g_now'].reshape(simu.model.nx,simu.model.nz)

        # if grad_smooth[math.floor(i/iter)] > 0:
        #     print('Smooth parameters: ',grad_smooth[math.floor(i/iter)])
        #     adj_grad = smooth2d(adj_grad, span=grad_smooth[math.floor(i/iter)])
        #     inv_scheme['g_now']  = adj_grad.flatten()
        # # Previous optim
        # lr = 0.001 * 0.8**(math.floor(i/10))
        # optimizer = torch.optim.Adam(DIP_model.parameters(), lr =lr)

        # # Unet DIP
        # Unet_u = Unet_update(simu,DIP_model,adj_grad.T,lr,optimizer,scheduler)
        # Unet_u.inversion()
        # output = DIP_model()

        # CNN DIP
        CNN_u = CNN_update(simu,DIP_model,adj_grad.T,lr,optimizer,scheduler)
        CNN_u.inversion()
        # output = DIP_model()
        
        output,hidden_outputs = DIP_model()
        print(output.mean())
        save_hidden_outputs(hidden_outputs,'/data/guangyuan/SWIT-1.0/examples/Hessian_DIP/model/')


        # update v
        v_tmp = output.cpu().detach().numpy()
        simu.model.vp = v_tmp.T

        # save v
        inv_scheme['v_now'] = simu.model.vp.flatten()
        # save and plot current outputs
        save_inv_scheme(simu, optim, inv_scheme)
        plot_inv_scheme(simu, optim, inv_scheme)

    print('DIPFWI time: ',time.time() - start_time)

    # # ######################################### LBFGS optim #########################################
    # # lr = 0.01
    # lr = 0.01
    # optimizer = torch.optim.LBFGS(DIP_model.parameters(), lr=lr,max_iter=1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.8)
    # # lr = 0.0005
    # # #### Adam ####
    # # optimizer = torch.optim.Adam(DIP_model.parameters())
    # # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.8)
    # for i in pbar_epoch:
    #     optim.iter += 1
    #     def closure():

    #         # optim.iter += 1
    #         # print('\n-----------  iteration %d  -----------\n'%optim.iter)
    #         # synthetic data from the current model
    #         forward(simu, simu_type='syn', savesnap=1)

    #         # process the synthetic data
    #         process_workflow(simu, optim, simu_type='syn')

    #         # evaluate the misfit
    #         inv_scheme['f_now'] = misfit(simu, optim.misfit_type)   
    #         print('Misfit:',inv_scheme['f_now'])

    #         # evaluate the gradient (with preconditioning and scaling)
    #         inv_scheme['g_now'] = adjoint(simu, optim)
    #         adj_grad = inv_scheme['g_now'].reshape(simu.model.nx,simu.model.nz)

    #         # CNN DIP
    #         CNN_u = CNN_update(simu,DIP_model,adj_grad.T,lr,optimizer,scheduler)
    #         grad = CNN_u.inversion()

    #         output = DIP_model()
    #         output.register_hook(grad)
    #         loss = torch.sum(output)
    #         optimizer.zero_grad()
    #         loss.backward()

    #         return loss
        
    #     optimizer.step(closure)
    #     # optimizer.step()
    #     scheduler.step()

    #     output = DIP_model()

    #     # update v
    #     v_tmp = output.cpu().detach().numpy()
    #     simu.model.vp = v_tmp.T

    #     # save v
    #     inv_scheme['v_now'] = simu.model.vp.flatten()
    #     # save and plot current outputs
    #     save_inv_scheme(simu, optim, inv_scheme)
    #     plot_inv_scheme(simu, optim, inv_scheme)

    save_path = simu.system.homepath + 'model/DIP_model_final.pt'
    torch.save(DIP_model.state_dict(),os.path.join(save_path))
    print('\n-----------  iteration end  -----------\n')


def save_hidden_outputs(hidden_outputs, save_dir="hidden_outputs"):

    os.makedirs(save_dir, exist_ok=True)
    
    for i, h in enumerate(hidden_outputs):
        h_np = h.cpu().numpy()
        np.save(os.path.join(save_dir, f"layer_{i}.npy"), h_np)
