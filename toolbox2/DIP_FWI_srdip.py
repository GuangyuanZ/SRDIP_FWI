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
from tools import array2vector, save_inv_scheme,smooth2d,loadbinfloat32
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
    # if deterministic and check_version(torch.__version__, '1.12.0'):  # https://github.com/ultralytics/yolov5/pull/8213
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
    DIP_model = DIP_Unet1(model_shape,
                        n_layers= layer_num,
                        vmin=vp_true.min()/1000,
                        vmax=vp_true.max()/1000,
                        base_channel=base_channel,
                        bilinear=True,
                        device=device)
    DIP_model.to(device)
    torch.save(DIP_model.state_dict(),os.path.join(save_path))
    # print(pretrain)
    np.savetxt('/home/guangyuan/桌面/SWIT-1.0/examples/Different_model/DIP_32_32/model/vp_random2.dat',DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
        # print('Update random max:',DIP_model.random_latent_vector[0,0,50,50])

    print('Prepare Unet model')
    # -----------------------------------
    #     Pretrain DIP model
    # -----------------------------------
    pretrain    = False
    # load_pretrained = True
    if pretrain:
        if load_pretrained:
            # load the model parameters
            DIP_model.load_state_dict(torch.load(load_path))
        else:
            # lr          = 0.005
            lr = 0.0005
            iteration   = 10000
            step_size   = 1000
            # iteration = 5000
            # step_size = 1
            gamma       = 0.5
            optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
            optimizer2 = torch.optim.Adam([DIP_model.random_latent_vector],lr = lr*1000)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
            scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,step_size=step_size,gamma=0.8)
            vp_init = numpy2tensor(vp_init,dtype=dtype).to(device)
            pbar = tqdm(range(iteration+1))
            for i in pbar:  
                vp_nn,_ = DIP_model()
                # print(vp_nn)
                # loss = torch.sqrt(torch.sum((vp_nn - vp_init)**2)) + torch.sqrt(torch.sum((dvp_nn - 0)**2)) 
                loss = torch.sqrt(torch.sum((vp_nn - 0)**2))
                optimizer.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer2.step()
                scheduler.step()
                scheduler2.step()
                pbar.set_description(f'Pretrain Iter:{i}, Misfit:{loss.cpu().detach().numpy()}')
            torch.save(DIP_model.state_dict(),os.path.join(save_path))
    np.savetxt('/home/guangyuan/桌面/SWIT-1.0/examples/DIP_32_32/model/vp_random1.dat',DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
        # print('Update random max:',DIP_model.random_latent_vector[0,0,50,50])

    return DIP_model

def Unet_prepare_bench(simu,vp_init,layer_num,load_pretrained = True):

    save_path = simu.system.homepath + 'model/DIP_model.pt'
    load_path = simu.system.homepath + 'model/DIP_model.pt'
        # Prepare Unet
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
    pretrain        = True
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
            vp_init = vp_init
            pbar = tqdm(range(iteration+1))
            for i in pbar:  
                vp_nn,_ = DIP_model()
                # print(vp_nn)
                loss = torch.sqrt(torch.sum((vp_nn - vp_init)**2))
                # loss = torch.sqrt(torch.sum((DIP_model.random_latent_vector - vp_init)**2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_description(f'Pretrain Iter:{i}, Misfit:{loss.cpu().detach().numpy()}')

            torch.save(DIP_model.state_dict(),os.path.join(save_path))

    return DIP_model

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
    # device = "cuda:0"
    device = "cpu"
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
                # loss = torch.sqrt(torch.sum((vp_nn - vp_init)**2))
                loss = torch.sqrt(torch.sum((DIP_model.random_latent_vector - vp_init)**2))
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

def Unet_sg_prepare(simu,vp_init,layer_num,load_pretrained = True):

    save_path = simu.system.homepath + 'model/DIP_model_init.pt'
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
    torch.save(DIP_model.state_dict(),os.path.join(save_path))
    np.savetxt('/home/guangyuan/桌面/SWIT-1.0/examples/多尺度测试/case_BP_sg/model/vp_random_initial.dat',DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
    print('Prepare Unet model')
    

    return DIP_model

def Unet_prepare_v(simu,vp_init,layer_num,load_pretrained = True):

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
    DIP_model = DIP_Unetv(model_shape,
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
    pretrain        = True
    # load_pretrained = True
    if pretrain:
        if load_pretrained:
            # load the model parameters
            DIP_model.load_state_dict(torch.load(load_path))
        else:
            lr          = 0.005
            iteration   = 3000
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




def CNN_prepare(simu,optim,vp_init,layer_num,load_pretrained = True):

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
    # vp_init = np.loadtxt('/home/guangyuan/桌面/SWIT-1.0/examples/多尺度测试/DIP_32_32/model/grad.dat').T
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
    print(h_in)
    print(w_in)
    random_state_num = 32 * h_in * w_in
    DIP_model = DIP_CNN(model_shape,
            # init_model = vp_init,
            in_channels=[32,32],
            vmin=vmin/1000,
            vmax=vmax/1000,
            # vmin=vmin,
            # vmax=vmax,
            random_state_num = 100,
            device=device)
    DIP_model.to(device)
    # DIP_model.freeze_network()
    # DIP_model.random_latent_vector.requires_grad = True

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

def MLP_prepare(simu,optim,vp_init,layer_num,load_pretrained = True):

    save_path = simu.system.homepath + 'model/DIP_model.pt'
    load_path = simu.system.homepath + 'model/DIP_model.pt'

    # -----------------------------------
    #    Model parameters
    # -----------------------------------

    nz = simu.model.nz
    nx = simu.model.nx
    # vp_true = simu.model.vp.T
    vmin = optim.vpmin
    vmax = optim.vpmax

    device = "cuda:0"
    base_channel = 64
    dtype = torch.float32
    print('Check vmin Value: ',vmin)
    print('Check vmax Value: ',vmax)

    model_shape = [nz,nx]
    DIP_model = DIP_MLP(model_shape,
                random_state_num = 100,
                hidden_layer_number = [1000,1000],
                vmin=vmin/1000,
                vmax=vmax/1000,
                device=device)
    DIP_model.to(device)

    print('Prepare CNN model')
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
            gamma       = 0.5
            optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
            # optimizer = torch.optim.LBFGS(DIP_model.parameters(), lr = lr, max_iter=5)
            # optimizer = torch.optim.SGD(DIP_model.parameters(), lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
            vp_init = numpy2tensor(vp_init,dtype=dtype).to(device)
            pbar = tqdm(range(iteration+1))
            for i in pbar:  
                # def closure():
                vp_nn = DIP_model()
                loss = torch.sqrt(torch.sum((vp_nn - vp_init)**2))
                optimizer.zero_grad()
                loss.backward()
                    
                # return loss
                
                # optimizer.step(closure)
                # scheduler.step()
                
                vp_nn1 = DIP_model()
                loss1 = torch.sqrt(torch.sum((vp_nn1 - vp_init)**2))
                
                optimizer.step()
                scheduler.step()
                pbar.set_description(f'Pretrain Iter:{i}, Misfit:{loss1.cpu().detach().numpy()}')
            torch.save(DIP_model.state_dict(),os.path.join(save_path))

    return DIP_model

class Unet_sg_update:

    def __init__(self,simu,x,DIP_model,adj_grad,lr,optimizer,scheduler,optimizer2,scheduler2):

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
        vp_tmp =  np.loadtxt('/home/guangyuan/桌面/SWIT-1.0/examples/Different_model/case_32_32_sg/model/vp_initial.dat').T
        vp_tensor = torch.from_numpy(vp_tmp).unsqueeze(0).unsqueeze(0).float().to('cuda:0')
        self.vp_init = vp_tensor
    

    def inversion(self):
    

        def grad_post_process(grad):
            if isinstance(self.adj_grad, np.ndarray):
                adj_grad_tensor = torch.from_numpy(self.adj_grad).to(grad.device, dtype=grad.dtype)
            else:
                adj_grad_tensor = self.adj_grad.to(grad.device)
            # diff_tmp = self.DIP_model.random_latent_vector.squeeze(0).squeeze(0)-self.output_tmp.squeeze(0).squeeze(0)
            # adj_grad_tensor = adj_grad_tensor + self.output_tmp.squeeze(0).squeeze(0)
            # adj_grad_tensor = torch.ones_like(grad) 
            
            return grad * (adj_grad_tensor + 1e-6)
                
        def zero_small_gradients(model, threshold=1e-6):
            for param in model.parameters():
                if param.grad is not None:
                    param.grad[torch.abs(param.grad) < threshold] = 0
        output = 0
        # for j in range(3):
        output,_ =  self.DIP_model(self.DIP_model.random_latent_vector)
            # output += output_tmp
        # output = output/3
        # output,_ = self.DIP_model()

        output.register_hook(grad_post_process)
        # print(torch.sum(output) )
        # print(0.01*torch.linalg.norm(self.DIP_model.random_latent_vector-self.vp_init))
        # loss = torch.sum(output)  + 0.01*torch.linalg.norm(self.DIP_model.random_latent_vector-self.vp_init)
        loss = torch.sum(output) 
        
        self.optimizer.zero_grad()
        # self.optimizer2.zero_grad()
        loss.backward()
        # print(self.DIP_model.random_latent_vector.grad)
        # print(self.adj_grad)
        #### Important ####
        grad_tmp = self.adj_grad / np.linalg.norm(self.adj_grad,ord=2)**2
        self.DIP_model.random_latent_vector = self.DIP_model.random_latent_vector - 1000/(1+np.exp((self.x-1000)/500))* torch.tensor(grad_tmp).unsqueeze(0).unsqueeze(0).float().to('cuda:0')
        self.optimizer.step()
        # self.optimizer2.step()
        self.scheduler.step()
        # self.scheduler2.step()

class Unet_dn_update:

    def __init__(self,simu,x,DIP_model,adj_grad,lr,optimizer,scheduler,optimizer2,scheduler2):
        
        # x /= x.max()
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
        # with torch.no_grad():
        #     # self.DIP_model.random_latent_vector.data /= torch.max(self.DIP_model.random_latent_vector.data)
        #     self.DIP_model.random_latent_vector.data = self.DIP_model.random_latent_vector.data/(torch.norm(self.DIP_model.random_latent_vector.data, p=2) ** 2)

class Unet_update:

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
        #         grad = grad*self.adj_grad
        #         grad  = numpy2tensor(grad,dtype=torch.float32).to("cuda:0")
        #     return grad
        def grad_post_process(grad):
            if isinstance(self.adj_grad, np.ndarray):
                adj_grad_tensor = torch.from_numpy(self.adj_grad).to(grad.device, dtype=grad.dtype)
            else:
                adj_grad_tensor = self.adj_grad.to(grad.device)
            # adj_grad_tensor = torch.ones_like(grad) 
            return grad * adj_grad_tensor 
        
        # Adam optimizor
        output,_ = self.DIP_model()
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

    def __init__(self,simu,x,DIP_model,adj_grad,lr,optimizer,scheduler,optimizer2,scheduler2):
        
        self.x = x
        self.simu = simu
        self.DIP_model = DIP_model
        self.adj_grad = adj_grad
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer2 = optimizer2
        self.scheduler2 = scheduler2


    def inversion(self):
        # np.savetxt('/home/guangyuan/桌面/SWIT-1.0/examples/Different_model/DIP_32_32/model/noise.dat',self.x.squeeze(0).squeeze(0).cpu().detach().numpy())
        # print('Noise.dat success')
        # def grad_post_process(grad):
        #     grad = grad.cpu().detach().numpy()
        #     with torch.no_grad():
        #         grad = grad*(self.adj_grad+1e-6)
        #         grad  = numpy2tensor(grad,dtype=torch.float32).to("cuda:0")
        #     return grad
        # print('Inversion max:')
        # print(x.max())
        output,hidden_outputs = self.DIP_model(self.x)
        def grad_post_process(grad):
            if isinstance(self.adj_grad, np.ndarray):
                adj_grad_tensor = torch.from_numpy(self.adj_grad).to(grad.device, dtype=grad.dtype)
            else:
                adj_grad_tensor = self.adj_grad.to(grad.device)
            adj_grad_tensor = adj_grad_tensor + output
            # adj_grad_tensor = torch.ones_like(grad) 
            return grad * (adj_grad_tensor + 1e-6)

        
        def zero_small_gradients(model, threshold=1e-6):
            for param in model.parameters():
                if param.grad is not None:
                    param.grad[torch.abs(param.grad) < threshold] = 0

        
        # # return grad_post_process
        # Adam optimizor
        # output = self.DIP_model()
        output,hidden_outputs = self.DIP_model(self.x)
        output.register_hook(grad_post_process)
        loss = torch.sum(output)
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        loss.backward()
        # print(self.DIP_model().hidden_outputs[0].grad)
        # torch.nn.utils.clip_grad_norm_(self.DIP_model.parameters(), max_norm=1.0)
        # max_grad = max(param.grad.abs().max().item() for param in self.DIP_model.parameters() if param.grad is not None)
        # print(max_grad)
        # zero_small_gradients(self.DIP_model, threshold=0.0005*max_grad)
 
        self.optimizer.step()
        self.optimizer2.step()
        for param_group in self.optimizer.param_groups:
            if param_group['lr'] < 0.001:
                param_group['lr'] = 0.001
            # 打印当前学习率
            print(f"Current learning rate: {param_group['lr']}")

        self.scheduler.step()
        self.scheduler2.step()
        

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

class CNN_update_bench:

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
class DD_update:

    def __init__(self,simu,DIP_model,adj_grad,lr,optimizer,scheduler,net_input):

        self.simu = simu
        self.DIP_model = DIP_model
        self.adj_grad = adj_grad.T
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net_input = net_input
    

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

            
            return grad * (adj_grad_tensor + 1e-6)


        

        # output = self.DIP_model(self.net_input.to('cpu')).mean(dim=[0,1], keepdim=False)
        # print(output.size)
        # output.register_hook(grad_post_process)
        # loss = torch.sum(output)
        # self.optimizer.zero_grad()
        # loss.backward()

        output,hidden_outputs = self.DIP_model()
        print(output.size)
        output.register_hook(grad_post_process)
        loss = torch.sum(output)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

class ADD_update:

    def __init__(self,simu,DIP_model,adj_grad,lr,optimizer,scheduler,net_input):

        self.simu = simu
        self.DIP_model = DIP_model
        self.adj_grad = adj_grad.T
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net_input = net_input
    

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

            
            return grad * (adj_grad_tensor + 1e-6)


        
        # # return grad_post_process
        # Adam optimizor
        # output = self.DIP_model()
        output = self.DIP_model(self.net_input.to('cpu')).mean(dim=[0,1], keepdim=False)
        output.register_hook(grad_post_process)
        loss = torch.sum(output)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        # self.scheduler.step()

def DIP_inversion(simu, optim,layer_num,inv_model,load_pretrained = True):
    ''' inversion workflow
    '''
    # Prepare Unet
    vp_init = inv_model['vp'] 
    # DIP_model = Unet_prepare(simu,vp_init.T,layer_num,load_pretrained)
    DIP_model = CNN_prepare(simu,optim,vp_init.T,layer_num,load_pretrained)
    
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
    # lr = 100
    lr = 0.01
    # dv Unet lr
    # lr = 0.0004
    # lr = 0.01
    # ### Adam ####
    optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
    # optimizer = torch.optim.Adam([DIP_model.random_latent_vector], lr=1000)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=150,gamma=0.85)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.8)

    # ## LBFGS ####
    # # optimizer = torch.optim.LBFGS(DIP_model.parameters(), line_search_fn='strong_wolfe')
    # optimizer = torch.optim.LBFGS(DIP_model.parameters(), lr = lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5,last_epoch=-1)
    # SGD ####
    # optimizer = torch.optim.SGD(DIP_model.parameters(), lr= 1e-11,momentum=0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.8)
    ### RMSprop ####
    # optimizer = torch.optim.RMSprop(DIP_model.parameters(), lr= lr,weight_decay=1e-8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.8)

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
        # for name, param in DIP_model.named_parameters():
        #     if param.grad is not None:
        #         # print(f"{name} gradient:\n{param.grad}")
        #         # print('/home/guangyuan/桌面/SWIT-1.0/examples/Inverse_model/model/'+f"{name}"+'.dat')
        #         np.save('/home/guangyuan/桌面/SWIT-1.0/examples/Inverse_model/model/'+f"{name}"+'.npy',param.grad.cpu().detach().numpy())
        #         # save_gradhidden_outputs(param.grad,'/home/guangyuan/桌面/SWIT-1.0/examples/Inverse_model/model/'+f"{name}")
        #     else:
        #         print(f"{name} has no gradient")

        ###### update model ######
        # output,_ = DIP_model()
        output,hidden_outputs = DIP_model()
        print(output.max())
        output = output + v_tmp.T
        # v_tmp = output.T
        output = torch.clamp(output, min=1500, max=4800)
        output = output.cpu().detach().numpy()
        
        simu.model.vp = output.T


        # save v
        inv_scheme['v_now'] = simu.model.vp.flatten()
        # save and plot current outputs
        if i % 1 == 0:
            save_inv_scheme(simu, optim, inv_scheme)
            plot_inv_scheme(simu, optim, inv_scheme)

        # save_inv_scheme(simu, optim, inv_scheme)
        # plot_inv_scheme(simu, optim, inv_scheme)

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

def lr_lambda(initial_lr,min_lr,epoch):
    lr = initial_lr * (0.8 ** (epoch // 40))
    return max(lr / initial_lr, min_lr / initial_lr)

def get_parameter_number(net):

    # calculating the parameters of network

    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    return {'Total': total_num, 'Trainable': trainable_num}

def DD_inversion(simu, optim,layer_num,inv_model,load_pretrained = True):
    ''' inversion workflow
    '''
    # Prepare Unet
    vp_init = inv_model['vp'] 
    DIP_model = Unet_prepare(simu,vp_init.T,layer_num,load_pretrained)
    v_mig = torch.tensor(vp_init)
    v_mig = v_mig.to('cuda:0')
    # DIP_model = CNN_prepare(simu,optim,vp_init.T,layer_num,load_pretrained)
    # DIP_model = CNN_adv_prepare(simu,optim,vp_init.T,layer_num,load_pretrained)
    # DIP_model = DD_prepare(simu,optim,vp_init.T,layer_num,load_pretrained)
    # DIP_model = MLP_prepare(simu,optim,vp_init.T,layer_num,load_pretrained)
    
    # set the initial model
    vp_tmp,_ = DIP_model()
    # inv_model['vp'] = vp_tmp.detach().cpu().numpy().T
    inv_model['vp'] = vp_init
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
    lr = 0.0008
    # lr = 0.002
    # lr = 0.00008
    # ### Adam ####
    optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.8)

    # ## LBFGS ####
    # # optimizer = torch.optim.LBFGS(DIP_model.parameters(), line_search_fn='strong_wolfe')
    # optimizer = torch.optim.LBFGS(DIP_model.parameters(), lr = lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5,last_epoch=-1)
    # SGD ####
    # optimizer = torch.optim.SGD(DIP_model.parameters(), lr= 1e-11,momentum=0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.8)
    ### RMSprop ####
    # optimizer = torch.optim.RMSprop(DIP_model.parameters(), lr= lr,weight_decay=1e-8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.8)

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
        for name, param in DIP_model.named_parameters():
            if param.grad is not None:
                # print(f"{name} gradient:\n{param.grad}")
                # print('/home/guangyuan/桌面/SWIT-1.0/examples/Inverse_model/model/'+f"{name}"+'.dat')
                np.save('/home/guangyuan/桌面/SWIT-1.0/examples/Inverse_model/model/'+f"{name}"+'.npy',param.grad.cpu().detach().numpy())
                # save_gradhidden_outputs(param.grad,'/home/guangyuan/桌面/SWIT-1.0/examples/Inverse_model/model/'+f"{name}")
            else:
                print(f"{name} has no gradient")
        # output = DIP_model()
        output = DIP_model()
        print(output.max())
        output = output + v_mig.T
        # print(grad_hidden_outputs.shape)
        # save_hidden_outputs(hidden_outputs,'/home/guangyuan/桌面/SWIT-1.0/examples/Inverse_model/model/')
        # save_gradhidden_outputs(grad_hidden_outputs,'/home/guangyuan/桌面/SWIT-1.0/examples/Inverse_model/model/')

        # update v
        v_tmp = output.cpu().detach().numpy()
        simu.model.vp = v_tmp.T

        # save v
        inv_scheme['v_now'] = simu.model.vp.flatten()
        # save and plot current outputs
        save_inv_scheme(simu, optim, inv_scheme)
        plot_inv_scheme(simu, optim, inv_scheme)

    print('DIPFWI time: ',time.time() - start_time)



    save_path = simu.system.homepath + 'model/DIP_model_final.pt'
    torch.save(DIP_model.state_dict(),os.path.join(save_path))
    print('\n-----------  iteration end  -----------\n')

def ADD_inversion(simu, optim,layer_num,inv_model,load_pretrained = True):
    ''' inversion workflow
    '''
    # Prepare DD network
    vp_init = inv_model['vp'] 

    device='cpu'
    DIP_model = ADD()

    save_path = simu.system.homepath + 'model/DIP_model.pt'
    print(get_parameter_number(DIP_model))
    torch.save( DIP_model.state_dict(),os.path.join(save_path))
    DIP_model.to(device)

    v_mig = torch.tensor(vp_init)
    v_mig = v_mig.to(device)
    
    
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
    
    lr = 0.008
    ### Adam ####
    optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5000,gamma=0.8)

    start_time = time.time()
    net_input = torch.randn(1,64,11,5)
    print('net_input.shape', net_input.shape)

    for i in pbar_epoch:

        DIP_model.train()

        # save v

        inv_scheme['v_now'] = simu.model.vp.flatten()

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

        DD_u = ADD_update(simu,DIP_model,adj_grad.T,lr,optimizer,scheduler,net_input)
        DD_u.inversion()

        output = DIP_model(net_input.to(device)).mean(dim=[0,1], keepdim=False).T + v_mig.T

        # output = v_mig.T

        output = torch.clamp(output, min=1500, max=4800)
        v_tmp = output.cpu().detach().numpy()
        # # save and plot current outputs
        if i % 1 == 0:
            save_inv_scheme(simu, optim, inv_scheme)
            plot_inv_scheme(simu, optim, inv_scheme)

        simu.model.vp = v_tmp.T


        # # save and plot current outputs
        # save_inv_scheme(simu, optim, inv_scheme)
        # plot_inv_scheme(simu, optim, inv_scheme)

    print('DIPFWI time: ',time.time() - start_time)



    save_path = simu.system.homepath + 'model/DIP_model_final.pt'
    torch.save(DIP_model.state_dict(),os.path.join(save_path))
    print('\n-----------  iteration end  -----------\n')


def save_hidden_outputs(iter,hidden_outputs, save_dir="hidden_outputs"):

    os.makedirs(save_dir, exist_ok=True)
    
    for i, h in enumerate(hidden_outputs):
        h_np = h.cpu().numpy()
        np.save(os.path.join(save_dir, f"layer_iter{iter}_{i}.npy"), h_np)


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
    # lr = 0.001
    # lr = 0.5
    # dv cnn lr
    # lr = 0.001
    # dv Unet lr
    # lr = 0.005
    # ### Adam ####
    optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=250,gamma=0.8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.8)
    optimizer2 = torch.optim.Adam([DIP_model.random_latent_vector], lr=lr*1000) 
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,step_size=1500,gamma=0.8)


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
        # x = 2500*torch.rand(1, 1, 77, 200).to('cuda:0')
        # np.savetxt('/home/guangyuan/桌面/SWIT-1.0/examples/Different_model/DIP_32_32/model/vp_latent.dat',DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
        # print('save vp_latent.dat success')
        CNN_u = CNN_update(simu,x,DIP_model,adj_grad.T,lr,optimizer,scheduler,optimizer2,scheduler2)
        CNN_u.inversion()

        ##### Update model #####
        output,_ = DIP_model()
        # output,[] = DIP_model(x)
        # output = DIP_model.random_latent_vector.squeeze(0).squeeze(0)
        # print(output.max())
        output = output + v_tmp.T
        # output = output + output1
        output = torch.clamp(output, min=optim.vpmin, max=optim.vpmax)
        output = output.cpu().detach().numpy()

        ##### update v #####
        # v_tmp = output.cpu().detach().numpy()
        # simu.model.vp = v_tmp.T
        simu.model.vp = output.T

        # save v
        inv_scheme['v_now'] = simu.model.vp.flatten()
        # save and plot current outputs
        if i % 1 == 0:
            save_inv_scheme(simu, optim, inv_scheme)
            plot_inv_scheme(simu, optim, inv_scheme)

        
        print('Update random max:',DIP_model.random_latent_vector[0,0,50,50])

    print('DIPFWI time: ',time.time() - start_time)


    # save_path = simu.system.homepath + 'model/DIP_model_final.pt'
    # torch.save(DIP_model.state_dict(),os.path.join(save_path))
    print('\n-----------  iteration end  -----------\n')


def DIP_inversion_unetv(simu, optim,layer_num,inv_model,load_pretrained = True):
    ''' inversion workflow
    '''
    # Prepare Unet
    vp_init = inv_model['vp'] 
    DIP_model = Unet_prepare_v(simu,vp_init.T,layer_num,load_pretrained)
    # DIP_model = CNN_prepare(simu,optim,vp_init.T,layer_num,load_pretrained)
    
    ##### set the initial model #####
    vp_tmp,_ = DIP_model()
    inv_model['vp'] = vp_tmp.detach().cpu().numpy().T
    # v_tmp = torch.tensor(vp_init)
    # device='cuda:0'
    # v_tmp = v_tmp.to(device)

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
    lr = 0.0005
    # lr = 0.5
    # dv cnn lr
    # lr = 0.001
    # dv Unet lr
    # lr = 0.005
    # ### Adam ####
    optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.8)
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
        CNN_u = Unet_update(simu,DIP_model,adj_grad.T,lr,optimizer,scheduler)
        CNN_u.inversion()

        ##### Update model #####
        output,_ = DIP_model()
        # output,[] = DIP_model()
        # print(output.max())
        # output = output + v_tmp.T
        # output = torch.clamp(output, min=optim.vpmin, max=optim.vpmax)
        # output = output.cpu().detach().numpy()

        ##### update v #####
        v_tmp = output.cpu().detach().numpy()
        simu.model.vp = v_tmp.T
        # simu.model.vp = output.T

        # save v
        inv_scheme['v_now'] = simu.model.vp.flatten()
        # save and plot current outputs
        if i % 1 == 0:
            save_inv_scheme(simu, optim, inv_scheme)
            plot_inv_scheme(simu, optim, inv_scheme)

    print('DIPFWI time: ',time.time() - start_time)


    # save_path = simu.system.homepath + 'model/DIP_model_final.pt'
    # torch.save(DIP_model.state_dict(),os.path.join(save_path))
    print('\n-----------  iteration end  -----------\n')


def DIP_inversion_unet_bench(simu, optim,layer_num,inv_model,load_pretrained = True):
    ''' inversion workflow
    '''
    # Prepare Unet
    vp_init = inv_model['vp'] 
    DIP_model = Unet_prepare_bench(simu,vp_init.T,layer_num,load_pretrained)
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=250,gamma=0.8)
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
        CNN_u = CNN_update_bench(simu,DIP_model,adj_grad.T,lr,optimizer,scheduler)
        CNN_u.inversion()

        ##### Update model #####
        # output,_ = DIP_model()
        output,[] = DIP_model()
        print(output.max())
        output = output + v_tmp.T
        output = torch.clamp(output, min=optim.vpmin, max=optim.vpmax)
        output = output.cpu().detach().numpy()

        ##### update v #####
        # v_tmp = output.cpu().detach().numpy()
        # simu.model.vp = v_tmp.T
        simu.model.vp = output.T

        # save v
        inv_scheme['v_now'] = simu.model.vp.flatten()
        # save and plot current outputs
        if i % 1 == 0:
            save_inv_scheme(simu, optim, inv_scheme)
            plot_inv_scheme(simu, optim, inv_scheme)
        # np.savetxt('/data/guangyuan/SWIT-1.0/examples/case7_CNN_FWI/model/vp_random.dat',DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
        print('Update random max:',DIP_model.random_latent_vector[0,0,50,50])

    print('DIPFWI time: ',time.time() - start_time)


    # save_path = simu.system.homepath + 'model/DIP_model_final.pt'
    # torch.save(DIP_model.state_dict(),os.path.join(save_path))
    print('\n-----------  iteration end  -----------\n')

def DIP_inversion_unet_sg(simu, optim,layer_num,inv_model,load_pretrained = True):
    ''' inversion workflow
    '''
    # Prepare Unet
    save_path1 = simu.system.homepath + 'model/input_init.dat'
    save_path2 = simu.system.homepath + 'model/input_inv.dat'
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
    # optimizer2 = torch.optim.Adam([DIP_model.random_latent_vector], lr=0.01*100) # Success
    optimizer2 = torch.optim.Adam([DIP_model.random_latent_vector], lr=0.01*80) # Success
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=300,gamma=0.8)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,step_size=300,gamma=0.8)

    start_time = time.time()
    for i in pbar_epoch:
        if i == 0:
            np.savetxt(save_path1,DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
        else:
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

        # CNN DIP
        # x = []
        output = 0
        # for j in range(3):
        x = 1000*(torch.rand(1, 1, 77, 200).to('cuda:0')-0.5)
            # x.append(noise)

        # output_tmp,hidden_outputs = DIP_model(x)
        CNN_u = Unet_sg_update(simu,i,DIP_model,adj_grad.T,lr,optimizer,scheduler,optimizer2,scheduler2)
        CNN_u.inversion()

        ##### Update model #####
        # output,_ = DIP_model()
        # for j in range(3):
        output,_ =  DIP_model(DIP_model.random_latent_vector)
        # output += output_tmp
        # output = output/3
        # output,[] = DIP_model(x)
        # print(output.max())
        rounded = torch.round(v_tmp.T * 1000) / 1000
        mask = rounded == 1499.616
        output[mask] = 0
        output = output + v_tmp.T
        output = torch.clamp(output, optim.vpmin, max=optim.vpmax)
        output = output.cpu().detach().numpy()

        ##### update v #####
        simu.model.vp = output.T

        # save v
        inv_scheme['v_now'] = simu.model.vp.flatten()
        # save and plot current outputs
        if i % 20 == 0:
            save_inv_scheme(simu, optim, inv_scheme)
            plot_inv_scheme(simu, optim, inv_scheme)

    print('DIPFWI time: ',time.time() - start_time)

    print('\n-----------  iteration end  -----------\n')


def sigmoid_lr(epoch):
    return 1.0 / (1.0 + math.exp(-(epoch - 700) / 251))

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
    # device='cuda:0'
    device = "cpu"
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
    # lr = 0.01
    lr = 0.01
    lr_base = 1
    # optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
    # optimizer2 = torch.optim.Adam(DIP_model.random_latent_vector(),lr=lr)
    optimizer = torch.optim.Adam([param for name, param in DIP_model.named_parameters() if name != 'random_latent_vector'],lr=lr)
    # optimizer2 = torch.optim.Adam([DIP_model.random_latent_vector], lr=10) # Success
    optimizer2 = torch.optim.Adam([DIP_model.random_latent_vector], lr=0.001*1000*25)#uccess
    # optimizer2 = torch.optim.Adam([DIP_model.random_latent_vector], lr=0.001*1000*4)#uccess
    # optimizer2 = torch.optim.Adam([DIP_model.random_latent_vector], lr=1*0.0001) # Success
    # optimizer2 = torch.optim.Adam([DIP_model.random_latent_vector], lr=lr_base)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=150,gamma=0.8)
    # scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sigmoid_lr)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,step_size=150,gamma=0.8)
    # scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,step_size=500,gamma=0.8)

    start_time = time.time()
    output = v_tmp.T
    output = output.cpu().detach().numpy()

    for i in pbar_epoch:
        if i == 0 :
            np.savetxt(save_path1,DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
        if i%10 == 0:
            save_path2 = simu.system.homepath + 'model/input_inv'+str(i)+'.dat'
            # save_path2 = simu.system.homepath + 'model/input_inv.dat'
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

        # CNN DIP
        # x = torch.from_numpy(output).unsqueeze(0).unsqueeze(0).float().to('cuda:0')
        # vp_tmp = loadbinfloat32('/home/guangyuan/桌面/SWIT-1.0/examples/FWI_str/model/vp-50.bin').reshape(200,77).T
        # vp_tmp = np.loadtxt('/home/guangyuan/桌面/SWIT-1.0/examples/多尺度测试/case_BP_true/model/vp_init.dat').T*0
        # x = torch.from_numpy(vp_tmp).unsqueeze(0).unsqueeze(0).float().to('cuda:0')  # 添加批次和通道维度
        # x = v_tmp.T.unsqueeze(0).unsqueeze(0).float().to('cuda:0')  # 添加批次和通道维度

        if i == 0:
            output0,_ =  DIP_model()
            output_tmp = output0.detach()
            # output_tmp = output_tmp + v_tmp.T
            output_tmp = v_tmp.T
            x = output_tmp.unsqueeze(0).unsqueeze(0).float().to('cpu')  # 添加批次和通道维度
        else:
            x = output_tmp.unsqueeze(0).unsqueeze(0).float().to('cpu')
        # print(x.shape)
        CNN_u = Unet_dn_update(simu,x,DIP_model,adj_grad.T,lr,optimizer,scheduler,optimizer2,scheduler2)
        CNN_u.inversion()
        CNN_u.inversion1()
        # print('Inv net para:',x.mean())

        output,_ =  DIP_model()
        rounded = torch.round(v_tmp.T * 1000) / 1000
        mask = rounded == 1486.000
        # mask = rounded == 1
        # mask = rounded == 1496.000
        output[mask] = 0
        output_tmp = output.detach()
        output = output + v_tmp.T
        # output_tmp = output_tmp + v_tmp.T
        output_tmp = v_tmp.T
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