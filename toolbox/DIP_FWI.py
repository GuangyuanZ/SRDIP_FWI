'''
* Author: Guangyuan Zou(USTC) : guangyuan@mail.ustc.edu.cn
* Date: 2025-10-10 09:08:12
* LastEditors: Guangyuan Zou
* LastEditTime: 2026-05-21 10:41:48
* Description: 
'''

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
# from ADFWI.model       import *
# from ADFWI.dip import *
from ADFWI.utils       import *
from Unet import UNet as DIP_Unet
from tqdm import tqdm
import time
# from ADFWI import DIP_CNN
import warnings
warnings.filterwarnings("ignore")
import pkg_resources as pkg
import os, random

def Unet_dn_prepare(simu,optim,vp_init,layer_num,load_pretrained = True):

    save_path = simu.system.homepath + 'model/DIP_model_init.pt'
    load_path = simu.system.homepath + 'model/DIP_model_init.pt'

    nz = simu.model.nz
    nx = simu.model.nx
    vp_true = simu.model.vp.T
    device = optim.device
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

    if load_pretrained:
        DIP_model.load_state_dict(torch.load(load_path))
        print('Load previous Unet work')

    return DIP_model

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

def DIP_inversion_unet_dn(simu, optim,layer_num,inv_model,load_pretrained = True):
    ''' inversion workflow
    '''
    save_path1 = simu.system.homepath + 'model/input_init.dat'
    # Prepare Unet
    vp_init = inv_model['vp'] 
    DIP_model = Unet_dn_prepare(simu,optim,vp_init.T,layer_num,load_pretrained)
    # print(DIP_model.random_latent_vector)
    ##### set the initial model #####
    v_tmp = torch.tensor(vp_init)
    device=optim.device
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
    optimizer = torch.optim.Adam([param for name, param in DIP_model.named_parameters() if name != 'random_latent_vector'],lr=optim.step_length1)
    optimizer2 = torch.optim.Adam([DIP_model.random_latent_vector], lr=optim.step_length2)#uccess
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=optim.decay_step1,gamma=0.8)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,step_size=optim.decay_step2,gamma=0.8)


    start_time = time.time()
    output = v_tmp.T
    output = output.cpu().detach().numpy()

    for i in pbar_epoch:
        if i == 0 :
            np.savetxt(save_path1,DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
        if i%10 == 0:
            # pass
            save_path2 = simu.system.homepath + 'model/input_inv'+str(i)+'.dat'
            save_pt_path = simu.system.homepath + 'model/itera'+str(i)+'.pt'
            # save_path2 = simu.system.homepath + 'model/input_inv.dat'
            np.savetxt(save_path2,DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
            torch.save(DIP_model.state_dict(),os.path.join(save_pt_path))
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
     
            x = output_tmp.unsqueeze(0).unsqueeze(0).float().to(device)  # 添加批次和通道维度
        else:
            x = output_tmp.unsqueeze(0).unsqueeze(0).float().to(device)
        # print(x.shape)
        CNN_u = Unet_dn_update(simu,x,DIP_model,adj_grad.T,lr,optimizer,scheduler,optimizer2,scheduler2)
        CNN_u.inversion()
        CNN_u.inversion1()
        # print('Inv net para:',x.mean())

        output,_ =  DIP_model()
        rounded = torch.round(v_tmp.T * 1000) / 1000
        mask = rounded == 1486.000  # mask water velocity

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


def DIP_inversion_unet(simu, optim,layer_num,inv_model,load_pretrained = True):
    ''' inversion workflow
    '''
    save_path1 = simu.system.homepath + 'model/input_init.dat'
    # Prepare Unet
    vp_init = inv_model['vp'] 
    DIP_model = Unet_dn_prepare(simu,optim,vp_init.T,layer_num,load_pretrained)
    # print(DIP_model.random_latent_vector)
    ##### set the initial model #####
    v_tmp = torch.tensor(vp_init)
    device=optim.device
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
    optimizer = torch.optim.Adam([param for name, param in DIP_model.named_parameters() if name != 'random_latent_vector'],lr=optim.step_length1)
    # optimizer = torch.optim.Adam(DIP_model.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam([DIP_model.random_latent_vector], lr=optim.step_length2)#uccess
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=optim.decay_step1,gamma=0.8)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,step_size=optim.decay_step2,gamma=0.8)


    start_time = time.time()
    output = v_tmp.T
    output = output.cpu().detach().numpy()

    for i in pbar_epoch:
        if i == 0 :
            np.savetxt(save_path1,DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
        if i%10 == 0:
            # pass
            save_path2 = simu.system.homepath + 'model/input_inv'+str(i)+'.dat'
            save_pt_path = simu.system.homepath + 'model/itera'+str(i)+'.pt'
            # save_path2 = simu.system.homepath + 'model/input_inv.dat'
            np.savetxt(save_path2,DIP_model.random_latent_vector.squeeze(0).squeeze(0).cpu().detach().numpy())
            torch.save(DIP_model.state_dict(),os.path.join(save_pt_path))
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
     
            x = output_tmp.unsqueeze(0).unsqueeze(0).float().to(device)  # 添加批次和通道维度
        else:
            x = output_tmp.unsqueeze(0).unsqueeze(0).float().to(device)
        # print(x.shape)
        CNN_u = Unet_dn_update(simu,x,DIP_model,adj_grad.T,lr,optimizer,scheduler,optimizer2,scheduler2)
        CNN_u.inversion()
        # CNN_u.inversion1()
        # print('Inv net para:',x.mean())

        output,_ =  DIP_model()
        rounded = torch.round(v_tmp.T * 1000) / 1000
        mask = rounded == 1486.000  # mask water velocity

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