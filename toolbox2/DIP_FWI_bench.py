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
    # vp_init = np.loadtxt('/home/guangyuan/桌面/SWIT-1.0/examples/多尺度测试/DIP_32_32/model/grad.dat').T
    # vp_init = vp_init/np.max(vp_init)
    nz = simu.model.nz
    nx = simu.model.nx
    vp_true = simu.model.vp.T
    device = "cuda:0"
    base_channel = 32
    dtype = torch.float32

    model_shape = [nz,nx]
    DIP_model = DIP_Unet(model_shape,
                        n_layers= layer_num,
                        vmin=vp_true.min()/1000,
                        vmax=vp_true.max()/1000,
                        base_channel=base_channel,
                        bilinear=True,
                        device=device)
    DIP_model.to(device)

    print('Prepare Unet model')
    # -----------------------------------
    #     Pretrain DIP model
    # -----------------------------------
    pretrain        = False
    # load_pretrained = True
    if pretrain:
        if load_pretrained:
            # load the model parameters
            DIP_model.load_state_dict(torch.load(load_path))
        else:
            lr          = 0.005
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
                vp_nn,_ = DIP_model()
                # print(vp_nn)
                loss = torch.sqrt(torch.sum((vp_nn - vp_init)**2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_description(f'Pretrain Iter:{i}, Misfit:{loss.cpu().detach().numpy()}')
            torch.save(DIP_model.state_dict(),os.path.join(save_path))

    return DIP_model

def Unet_sg_prepare(simu,vp_init,layer_num,load_pretrained = True):

    save_path = simu.system.homepath + 'model/DIP_model.pt'
    load_path = simu.system.homepath + 'model/DIP_model.pt'

    nz = simu.model.nz
    nx = simu.model.nx
    vp_true = simu.model.vp.T
    device = "cuda:0"
    base_channel = 32
    dtype = torch.float32

    model_shape = [nz,nx]
    DIP_model = DIP_Unet_sg(model_shape,
                        n_layers= layer_num,
                        vmin=vp_true.min()/1000,
                        vmax=vp_true.max()/1000,
                        base_channel=base_channel,
                        bilinear=True,
                        device=device)
    DIP_model.to(device)

    print('Prepare Unet model')

    return DIP_model


def CNN_prepare_dv(simu,optim,vp_init,layer_num,load_pretrained = True):

    save_path = simu.system.homepath + 'model/DIP_model.pt'
    load_path = simu.system.homepath + 'model/DIP_model.pt'

    # Fix optimizor or not
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
    # device = "cpu"
    base_channel = 64
    dtype = torch.float32
    print('Check vmin Value: ',vmin)
    print('Check vmax Value: ',vmax)
    # loss_pretrain = np.zeros(10000)
    model_shape = [nz,nx]
    h_in  = math.ceil(model_shape[0]/(2**4))
    w_in  = math.ceil(model_shape[1]/(2**4))
    random_state_num = 32 * h_in * w_in
    DIP_model = DIP_CNN1(model_shape,
            in_channels=[32,32],
            vmin=vmin/1000,
            vmax=vmax/1000,
            # vmin=vmin,
            # vmax=vmax,
            random_state_num = 100,
            device=device)
    DIP_model.to(device)

    print('Prepare CNN model')
    # -----------------------------------
    #     Pretrain DIP model
    # -----------------------------------
    pretrain        = False
    # load_pretrained = True
    start_time = time.time()
    if pretrain:
        if load_pretrained:
            # load the model parameters
            DIP_model.load_state_dict(torch.load(load_path))
        else:
            # lr          = 0.005
            lr = 0.005
            iteration   = 10000
            step_size   = 1000
            # iteration = 5000
            # step_size = 1
            gamma       = 0.5
            # gamma = 0.8
            
            optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
            vp_init = numpy2tensor(vp_init,dtype=dtype).to(device)
            pbar = tqdm(range(iteration+1))
            for i in pbar:  
                # vp_nn = DIP_model()
                vp_nn,hidden_outputs = DIP_model()
                # save_hidden_outputs(hidden_outputs,'/home/guangyuan/桌面/SWIT-1.0/examples/多尺度测试/case7_CNN_FWI/model/')
                # print('********************************')
                # print('\n')
                # print(grad_hidden_outputs)
                # print('\n')
                # print(hidden_outputs)
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
            random_state_num = 5000,
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
            # lr = 0.001
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
    # _,hidden_outputs = DIP_model()
    # save_hidden_outputs(hidden_outputs,'/data/guangyuan/SWIT-1.0/examples/Hessian_DIP/model/')
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
            # adj_grad_tensor = torch.ones_like(grad) 
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

        # for name, param in self.DIP_model.named_parameters():
        #     if param.grad is not None:
        #         # print('ss')
        #         fname = '/data/guangyuan/SWIT-1.0/examples/Hessian_DIP/model/' + f"{name}" + ".npy"
        #         grad = param.grad.cpu().numpy()
        #         np.save(os.path.join(fname), grad)


        self.optimizer.step()
        self.scheduler.step()


class Unet_sg_update:

    def __init__(self,simu,DIP_model,adj_grad,output_tmp,lr,optimizer,scheduler,optimizer2,scheduler2):

        self.simu = simu
        self.DIP_model = DIP_model
        self.adj_grad = adj_grad
        self.output_tmp = output_tmp
        self.lr = lr
        self.optimizer = optimizer
        self.optimizer2 = optimizer2
        self.scheduler = scheduler
        self.scheduler2 = scheduler2
    

    def inversion(self):
    

        def grad_post_process(grad):
            if isinstance(self.adj_grad, np.ndarray):
                adj_grad_tensor = torch.from_numpy(self.adj_grad).to(grad.device, dtype=grad.dtype)
            else:
                adj_grad_tensor = self.adj_grad.to(grad.device)
            # diff_tmp = self.DIP_model.random_latent_vector.squeeze(0).squeeze(0)-self.output_tmp.squeeze(0).squeeze(0)
            # adj_grad_tensor = adj_grad_tensor +adj_grad_tensor.max()*diff_tmp/diff_tmp.max()
            # adj_grad_tensor = torch.ones_like(grad) 
            
            return grad * (adj_grad_tensor + 1e-6)
                
        def zero_small_gradients(model, threshold=1e-6):
            for param in model.parameters():
                if param.grad is not None:
                    param.grad[torch.abs(param.grad) < threshold] = 0
        
        output,_ = self.DIP_model()

        output.register_hook(grad_post_process)
        loss = torch.sum(output)
        
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        loss.backward()
        # print(self.DIP_model.random_latent_vector.grad)


        self.optimizer.step()
        self.optimizer2.step()
        self.scheduler.step()
        self.scheduler2.step()


def DIP_inversion(simu, optim,layer_num,inv_model,load_pretrained = True):
    ''' inversion workflow
    '''
    # Prepare Unet
    vp_init = inv_model['vp'] 
    # DIP_model = Unet_prepare(simu,vp_init.T,layer_num,load_pretrained)
    # DIP_model = Unet_prepare(simu,vp_init.T,layer_num,load_pretrained)
    DIP_model = CNN_prepare(simu,optim,vp_init.T,layer_num,load_pretrained)
    # DIP_model = CNN_adv_prepare(simu,optim,vp_init.T,layer_num,load_pretrained)
    
    # set the initial model
    vp_init_tmp,_ = DIP_model()
    inv_model['vp']  = vp_init_tmp.detach().cpu().numpy().T

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
    lr = 0.0004
    # lr = 0.0004
    # lr = 0.01
    # lr = 0.0002
    # lr  = 0.00001
    # ### Adam ####
    optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.8)
     # # SGD ####
    # optimizer = torch.optim.SGD(DIP_model.parameters(), lr= 1e-9,momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.8)

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

        # CNN DIP
    
        CNN_u = CNN_update(simu,DIP_model,adj_grad.T,output_tmp,lr,optimizer,scheduler)
        CNN_u.inversion()
        # output = DIP_model()
        
        output,hidden_outputs = DIP_model()
        # save_hidden_outputs(hidden_outputs,'/data/guangyuan/SWIT-1.0/examples/Hessian_DIP/model/')


        # update v
        v_tmp = output.cpu().detach().numpy()
        simu.model.vp = v_tmp.T

        # save v
        inv_scheme['v_now'] = simu.model.vp.flatten()
        # save and plot current outputs
        save_inv_scheme(simu, optim, inv_scheme)
        plot_inv_scheme(simu, optim, inv_scheme)
        save_path = simu.system.homepath + 'model/DIP_model_inv.pt'
        torch.save(DIP_model.state_dict(),os.path.join(save_path))

    print('DIPFWI time: ',time.time() - start_time)



    # save_path = simu.system.homepath + 'model/DIP_model_final.pt'
    # torch.save(DIP_model.state_dict(),os.path.join(save_path))
    print('\n-----------  iteration end  -----------\n')


def save_hidden_outputs(hidden_outputs, save_dir="hidden_outputs"):

    os.makedirs(save_dir, exist_ok=True)
    
    for i, h in enumerate(hidden_outputs):
        h_np = h.detach().cpu().numpy()
        np.save(os.path.join(save_dir, f"layer_{i}.npy"), h_np)


def DIP_inversion_unet(simu, optim,layer_num,inv_model,load_pretrained = True):
    ''' inversion workflow
    '''
    # Prepare Unet
    vp_init = inv_model['vp'] 
    DIP_model = Unet_prepare(simu,vp_init.T,layer_num,load_pretrained)
    # DIP_model = CNN_prepare(simu,optim,vp_init.T,layer_num,load_pretrained)
    
    ##### set the initial model #####
    # vp_tmp,_ = DIP_model()
    # inv_model['vp'] = vp_tmp.detach().cpu().numpy().T
    v_tmp = torch.tensor(vp_init)
    device='cuda:0'
    v_tmp = v_tmp.to(device)

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
    # v cnn lr
    lr = 0.01
    # lr = 0.5
    # dv cnn lr
    # lr = 0.001
    # dv Unet lr
    # lr = 0.005
    # ### Adam ####
    optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=150,gamma=0.8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.8)


    start_time = time.time()
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

        # CNN DIP
        CNN_u = CNN_update(simu,DIP_model,adj_grad.T,lr,optimizer,scheduler)
        CNN_u.inversion()

        ##### Update model #####
        # output,_ = DIP_model()
        output,[] = DIP_model()
        print(output.max())
        output = output + v_tmp.T
        output = torch.clamp(output, min=1200, max=5800)
        output = output.cpu().detach().numpy()

        ##### update v #####
        # v_tmp = output.cpu().detach().numpy()
        # simu.model.vp = v_tmp.T
        simu.model.vp = output.T

        # save v
        inv_scheme['v_now'] = simu.model.vp.flatten()
        # save and plot current outputs
        if i % 10 == 0:
            save_inv_scheme(simu, optim, inv_scheme)
            plot_inv_scheme(simu, optim, inv_scheme)
        # np.savetxt('/data/guangyuan/SWIT-1.0/examples/case7_CNN_FWI/model/vp_random.dat',DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
        # print('Update random max:',DIP_model.random_latent_vector[0,0,50,50])
        save_path = simu.system.homepath + 'model/DIP_model_inv.pt'
        torch.save(DIP_model.state_dict(),os.path.join(save_path))

    print('DIPFWI time: ',time.time() - start_time)


    # save_path = simu.system.homepath + 'model/DIP_model_final.pt'
    # torch.save(DIP_model.state_dict(),os.path.join(save_path))
    print('\n-----------  iteration end  -----------\n')


def DIP_inversion_cnndv(simu, optim,layer_num,inv_model,load_pretrained = True):
    ''' inversion workflow
    '''
    # Prepare Unet
    vp_init = inv_model['vp'] 
    DIP_model = CNN_prepare_dv(simu,optim,vp_init.T,layer_num,load_pretrained)
    
    ###### set the initial model ######
    # vp_tmp,_ = DIP_model()
    # inv_model['vp'] = vp_tmp.detach().cpu().numpy().T
    v_tmp = torch.tensor(vp_init)
    device='cuda:0'
    v_tmp = v_tmp.to(device)

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
    # v cnn lr
    # lr = 0.0004
    # lr = 0.0008
    # v unet lr
    # lr = 0.0004
    # dv cnn lr
    # lr = 0.004
    lr = 0.01
    # dv Unet lr
    # lr = 0.0004
    # lr = 0.01
    # ### Adam ####
    optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=150,gamma=0.8)

    start_time = time.time()
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

        # CNN DIP
        CNN_u = CNN_update(simu,DIP_model,adj_grad.T,lr,optimizer,scheduler)
        CNN_u.inversion()
        
        ###### update model ######
        # output,_ = DIP_model()
        output,hidden_outputs = DIP_model()
        print(output.max())
        output = output + v_tmp.T
        output = torch.clamp(output, min=optim.vpmin, max=optim.vpmax)
        output = output.cpu().detach().numpy()

        ###### update v ######
        # v_tmp = output.cpu().detach().numpy()
        # simu.model.vp = v_tmp.T
        simu.model.vp= output.T


        # save v
        inv_scheme['v_now'] = simu.model.vp.flatten()
        # save and plot current outputs
        save_inv_scheme(simu, optim, inv_scheme)
        plot_inv_scheme(simu, optim, inv_scheme)

    print('DIPFWI time: ',time.time() - start_time)

    save_path = simu.system.homepath + 'model/DIP_model_final.pt'
    torch.save(DIP_model.state_dict(),os.path.join(save_path))
    print('\n-----------  iteration end  -----------\n')





def DIP_inversion_unet_sg(simu, optim,layer_num,inv_model,load_pretrained = True):
    ''' inversion workflow
    '''
    # Prepare Unet
    vp_init = inv_model['vp'] 
    DIP_model = Unet_sg_prepare(simu,vp_init.T,layer_num,load_pretrained)
    # print(DIP_model.random_latent_vector)
    ##### set the initial model #####
    v_tmp = torch.tensor(vp_init)
    device='cuda:0'
    v_tmp = v_tmp.to(device)

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
    lr = 0.01
    # optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
    # optimizer2 = torch.optim.Adam(DIP_model.random_latent_vector(),lr=lr)
    optimizer = torch.optim.Adam([param for name, param in DIP_model.named_parameters() if name != 'random_latent_vector'],lr=lr)
    optimizer2 = torch.optim.Adam([DIP_model.random_latent_vector], lr=0.01) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=150,gamma=0.8)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,step_size=150,gamma=0.8)

    start_time = time.time()
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

        # CNN DIP
        output_tmp,hidden_outputs = DIP_model()
        CNN_u = Unet_sg_update(simu,DIP_model,adj_grad.T,output_tmp,lr,optimizer,scheduler,optimizer2,scheduler2)
        CNN_u.inversion()

        ##### Update model #####
        # output,_ = DIP_model()
        output,[] = DIP_model()
        # print(output.max())
        output = output + v_tmp.T
        output = torch.clamp(output, min=1200, max=5800)
        output = output.cpu().detach().numpy()

        ##### update v #####
        simu.model.vp = output.T

        # save v
        inv_scheme['v_now'] = simu.model.vp.flatten()
        # save and plot current outputs
        if i % 10 == 0:
            save_inv_scheme(simu, optim, inv_scheme)
            plot_inv_scheme(simu, optim, inv_scheme)
        np.savetxt('/data/guangyuan/SWIT-1.0/examples/case7_CNN_FWI/model/vp_random.dat',DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
        print('Update random max:',DIP_model.random_latent_vector[0,0,50,50])
    print('DIPFWI time: ',time.time() - start_time)

    print('\n-----------  iteration end  -----------\n')

def Unet_dn_prepare(simu,vp_init,layer_num,load_pretrained = True):

    save_path = simu.system.homepath + 'model/DIP_model_init.pt'
    load_path = simu.system.homepath + 'model/DIP_model_init.pt'
    save_path1 = simu.system.homepath + 'model/input_init_pre.dat'
    save_path2 = simu.system.homepath + 'model/input_inv_pre.dat'

    # -----------------------------------
    #    Model parameters
    # -----------------------------------
    # vp_init = np.loadtxt('/home/guangyuan/桌面/SWIT-1.0/examples/多尺度测试/DIP_32_32/model/grad.dat').T
    # vp_init = vp_init/np.max(vp_init)
    nz = simu.model.nz
    nx = simu.model.nx
    vp_true = simu.model.vp.T
    device = "cuda:0"
    base_channel = 32
    # base_channel = 8
    dtype = torch.float32

    model_shape = [nz,nx]
    DIP_model = DIP_Unet_dn(model_shape,
                        n_layers= layer_num,
                        vmin=vp_true.min()/1000,
                        vmax=vp_true.max()/1000,
                        base_channel=base_channel,
                        bilinear=True,
                        device=device)
    DIP_model.to(device)
    torch.save(DIP_model.state_dict(),os.path.join(save_path))
    print('Prepare Unet model')
    # -----------------------------------
    #     Pretrain DIP model
    # -----------------------------------
    # load_pretrained = True
    pretrain        = False
    # load_pretrained = True
    if pretrain:
            lr          = 0.005
            iteration   = 200
            step_size   = 250
            # iteration = 5000
            # step_size = 1
            gamma       = 0.5
            # optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
            optimizer = torch.optim.Adam([DIP_model.random_latent_vector], lr=20) # Success
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
            vp_init = numpy2tensor(vp_init,dtype=dtype).to(device)
            # vp_init = vp_init * 0
            pbar = tqdm(range(iteration+1))
            for i in pbar:  
                if i == 0:
                    np.savetxt(save_path1,DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
                else:
                    np.savetxt(save_path2,DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
                vp_nn,_ = DIP_model()
                # print(vp_nn)
                loss = torch.sqrt(torch.sum((vp_nn - vp_init)**2))
                # loss = torch.sqrt(torch.sum((DIP_model.random_latent_vector - vp_init)**2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_description(f'Pretrain Iter:{i}, Misfit:{loss.cpu().detach().numpy()}')
            # torch.save(DIP_model.state_dict(),os.path.join(save_path))

    if load_pretrained:
        DIP_model.load_state_dict(torch.load(load_path))
        print('Load previous Unet work')

    return DIP_model

class Unet_dn_update:

    def __init__(self,simu,x,DIP_model,adj_grad,lr,optimizer,scheduler,optimizer2,scheduler2):

        # x = x/x.max()
        self.simu = simu
        self.DIP_model = DIP_model
        self.adj_grad = adj_grad
        self.x = x
        # self.output_tmp = output_tmp
        self.lr = lr
        self.optimizer = optimizer
        self.optimizer2 = optimizer2
        self.scheduler = scheduler
        self.scheduler2 = scheduler2
    

    def inversion(self):
    

        def grad_post_process(grad):
            if isinstance(self.adj_grad, np.ndarray):
                adj_grad_tensor = torch.from_numpy(self.adj_grad).to(grad.device, dtype=grad.dtype)
            else:
                adj_grad_tensor = self.adj_grad.to(grad.device)
            return grad * (adj_grad_tensor + 1e-6)
                
        def zero_small_gradients(model, threshold=1e-6):
            for param in model.parameters():
                if param.grad is not None:
                    param.grad[torch.abs(param.grad) < threshold] = 0
        output = 0
        output,_ =  self.DIP_model()

        output.register_hook(grad_post_process)
        loss = torch.sum(output) 
        
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()
    
    def inversion1(self):


        loss = torch.sqrt(torch.sum((self.DIP_model.random_latent_vector - self.x)**2))
        # print(self.x.mean())
        self.optimizer2.zero_grad()
        loss.backward()


        self.optimizer2.step()
        self.scheduler2.step()

def DIP_inversion_unet_dn(simu, optim,layer_num,inv_model,load_pretrained = True):
    ''' inversion workflow
    '''
    save_path1 = simu.system.homepath + 'model/input_init.dat'
    # save_path2 = simu.system.homepath + 'model/input_inv.dat'
    # Prepare Unet
    vp_init = inv_model['vp'] 
    DIP_model = Unet_dn_prepare(simu,vp_init.T,layer_num,load_pretrained)
    # print(DIP_model.random_latent_vector)
    ##### set the initial model #####
    v_tmp = torch.tensor(vp_init)
    device='cuda:0'
    v_tmp = v_tmp.to(device)

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
    lr = 0.01
    # lr = 100
    optimizer = torch.optim.Adam([param for name, param in DIP_model.named_parameters() if name != 'random_latent_vector'],lr=lr)
    # optimizer2 = torch.optim.Adam([DIP_model.random_latent_vector], lr=10) # Success
    optimizer2 = torch.optim.Adam([DIP_model.random_latent_vector], lr=0.001*1000*1) # Success
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=250,gamma=0.8)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,step_size=250,gamma=0.8)

    start_time = time.time()
    output = v_tmp.T
    output = output.cpu().detach().numpy()

    for i in pbar_epoch:
        if i == 0 :
            np.savetxt(save_path1,DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
        if i%1 == 0:
            # save_path2 = simu.system.homepath + 'model/input_inv'+str(i)+'.dat'
            save_path2 = simu.system.homepath + 'model/input_inv.dat'
            np.savetxt(save_path2,DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
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


        if i == 0:
            output0,_ =  DIP_model()
            output_tmp = output0.detach()
            output_tmp = output_tmp + v_tmp.T
            x = output_tmp.unsqueeze(0).unsqueeze(0).float().to('cuda:0')  # 添加批次和通道维度
        else:
            x = output_tmp.unsqueeze(0).unsqueeze(0).float().to('cuda:0')
        # print(x.shape)
        CNN_u = Unet_dn_update(simu,x,DIP_model,adj_grad.T,lr,optimizer,scheduler,optimizer2,scheduler2)
        CNN_u.inversion()
        CNN_u.inversion1()
        # print('Inv net para:',x.mean())

        output,_ =  DIP_model()
        rounded = torch.round(v_tmp.T * 1000) / 1000
        mask = rounded == 1486.000
        output[mask] = 0
        output_tmp = output.detach()
        output = output + v_tmp.T
        output_tmp = output_tmp + v_tmp.T
        output = torch.clamp(output,optim.vpmin, max=optim.vpmax)
        output = output.cpu().detach().numpy()
        
        ##### update v #####
        simu.model.vp = output.T

        # save v
        inv_scheme['v_now'] = simu.model.vp.flatten()
        # save and plot current outputs
        if i % 10 == 0:
            save_inv_scheme(simu, optim, inv_scheme)
            plot_inv_scheme(simu, optim, inv_scheme)
    print('DIPFWI time: ',time.time() - start_time)

    print('\n-----------  iteration end  -----------\n')