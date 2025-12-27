import torch
import torch.nn  as nn
import numpy as np
import math
from utils import build_scaling_rotation, inverse_sigmoid,strip_symmetric
from sh_utils import RGB2SH
from scipy.spatial import KDTree
from torch.autograd import Function


def row_max_to_one(matrix):
    max_values, _ = torch.max(matrix, dim=1, keepdim=True)
    result = (matrix == max_values)
    return result.float()
    
def distCUDA2(points):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    meanDists = np.clip(meanDists, 1e-8, np.inf)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

class GaussModel(nn.Module):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid

        self.bias_activation = torch.tanh

        self.inverse_T_activation = inverse_sigmoid

        self.T_activation = nn.Softmax(dim=1)

        self.inverse_opacity_activation = inverse_sigmoid
        self.projection_activation=torch.sigmoid

        self.rotation_activation = torch.nn.functional.normalize
    
    def __init__(self, sh_degree : int=4,debug=False,mode='train'):
        super(GaussModel, self).__init__()
        if mode=='train':
            self.active_sh_degree=0
        elif mode=='test':
            print('==Test==')
            self.active_sh_degree=sh_degree ####test set =4 train =0
        self.max_sh_degree = sh_degree  

        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self._T = torch.empty(0)
        self._bias = torch.empty(0)

        self._phi=torch.empty(0)
        self._phi_o=torch.empty(0)
    
        self.linear1=nn.Sequential(
                        nn.Linear(3,6),
                        nn.Sigmoid(),
                        nn.BatchNorm1d(6),
                        nn.Linear(6,3),
                        nn.Sigmoid(),
                        nn.BatchNorm1d(3),
                        nn.Linear(3,2),
                        nn.Sigmoid()
        )
 
        self.linear2=nn.Sequential(
                        nn.Linear(9,6),
                        nn.Linear(6,3),
                        nn.Linear(3,4),
                        nn.ReLU()
        )

        self.linear3=nn.Sequential(
                        nn.Linear(63,30),
                        nn.ReLU(),
                        nn.BatchNorm1d(30),
                        nn.Linear(30,15),
                        nn.ReLU(),
                        nn.BatchNorm1d(15),
                        nn.Linear(15,1),
                        nn.ReLU()
        )
        self.setup_functions()
        self.debug = debug


    def create_from_pcd(self, pcd,M,P_BS):
       
        points = pcd #the number of points*3 (x,y,z), tensor
        fused_point_cloud= points.float().cuda()
        N=pcd.shape[0]
        print("Number of points at initialisation : ", N)
        print("Number of points after selection : ", M)
        # 
        distances = torch.norm(fused_point_cloud - P_BS, dim=1)  # Compute distances to the base station
        _, indices = torch.topk(-distances, M) 
     
        T_init_ = torch.zeros(M, N)
        T_init_[torch.arange(M), indices] = 1

        T_init=T_init_

        rsrps= torch.zeros(M,32) #the number of points*the number of beams, tensor

        b_init=torch.zeros(M,3)

        phi_init=torch.zeros(M,32)

        phi_o_init=torch.zeros(M,32)

        fused_rsrps=  RGB2SH(rsrps.float().cuda())

        features = torch.zeros((fused_rsrps.shape[0], 32, (self.max_sh_degree + 1) ** 2)).float().cuda()

        features[:, :32, 0 ] = fused_rsrps
        features[:, 32:, 1:] = 0.0

        point=points[indices,:].detach().cpu().numpy()
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(point)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    
        rots = torch.zeros((M, 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((M, 1), dtype=torch.float, device="cuda"))


        print("Number of points after selection : ", M)


        if self.debug:
            # easy for visualization
            colors = np.zeros_like(colors)
            opacities = inverse_sigmoid(0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        self._init_xyz=fused_point_cloud
        self._T=nn.Parameter((T_init).contiguous().float().cuda().requires_grad_(True))
        self._b=nn.Parameter(b_init.contiguous().float().cuda().requires_grad_(True))
        self._phi=nn.Parameter(phi_init.contiguous().float().cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._phi_o=nn.Parameter(phi_o_init.contiguous().float().cuda().requires_grad_(True))
       
        return self

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)+0.000001

    @property
    def get_bias(self):
        return self._b,self._b
   
    @property
    def get_selection_matrix(self):
        selection_matrix =self.T_activation(self._T)
        return selection_matrix, self._T
    
  
    
    @property
    def get_selection_matrix_eval(self):
        selection_matrix =self.T_activation(self._T)
        selection_matrix_=row_max_to_one(selection_matrix)
        return selection_matrix_, self._T

    @property
    def get_xyz(self):
        T,_=self.get_selection_matrix
        # ll=800
        b,bb=self.get_bias
        _xyz=T@self._init_xyz
        _xyz_=_xyz.detach().clone()
        _xyz_[:,2]= torch.maximum(_xyz_[:, 2], torch.tensor(0.0))
        _xyz_bias=  T@self._init_xyz+b
        _xyz_bias_=_xyz_bias.detach().clone()
        _xyz_bias_[:,2]= torch.maximum(_xyz_bias[:, 2], torch.tensor(0.0))
        return _xyz, _xyz_bias,b,bb
    
    @property
    def get_xyz_eval(self):
        T,_=self.get_selection_matrix_eval
        # ll=800
        b,bb=self.get_bias
        _xyz=T@self._init_xyz
        _xyz_=_xyz.detach().clone()
        _xyz_[:,2]= torch.maximum(_xyz_[:, 2], torch.tensor(0.0))
        _xyz_bias=  T@self._init_xyz+b
        _xyz_bias_=_xyz_bias.detach().clone()
        _xyz_bias_[:,2]= torch.maximum(_xyz_bias[:, 2], torch.tensor(0.0))
        return _xyz, _xyz_bias,b,bb
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1),self._phi
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity),self._phi_o
    
    @property
    def get_project(self):
        return  self.linear1,self.linear2,self.linear3
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            print('==========')
            print('SH degree plus 1')
            print('==========')
            self.active_sh_degree += 1
    
    def training_setup(self, training_args):

        l = [
            {'params': [self._features_dc], 'lr': training_args['feature_lr'], "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args['feature_lr']/ 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args['opacity_lr'], "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args['scaling_lr'], "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args['rotation_lr'], "name": "rotation"},
            {'params': self.linear1.parameters(), 'lr': training_args['linear1'], "name": "linear1"},
            {'params': self.linear2.parameters(), 'lr': training_args['linear2'], "name": "linear2"},
            {'params': self.linear3.parameters(), 'lr': training_args['linear3'], "name": "linear3"},
            {'params': [self._T], 'lr': training_args['T_lr'], "name": "selection"},
            {'params': [self._b], 'lr': training_args['bias_lr'], "name": "bias"},
            {'params': [self._phi], 'lr': training_args['phi_lr'], "name": "phi"},
             {'params': [self._phi_o], 'lr': training_args['phi_o_lr'], "name": "phi"},
 
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args['position_lr_init'],
                                                    lr_final=training_args['position_lr_final'],
                                                    lr_delay_mult=training_args['position_lr_delay_mult'],
                                                    max_steps=training_args['position_lr_max_steps'])
        return  self.optimizer 


