import pdb
import torch
import torch.nn as nn
import math
from einops import reduce
from sh_utils import eval_sh
import torch.autograd.profiler as profiler
USE_PROFILE = False
import contextlib
import pdb

class Embedder():
    """positional encoding
    """
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']    # input dimension of gamma
        out_dim = 0

        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']    # L-1, 10-1 by default
        N_freqs = self.kwargs['num_freqs']         # L


        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)  #2^[0,1,...,L-1]
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """return: gamma(input)
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)




def get_embedder(multires, is_embeded=True, input_dims=1):
    """get positional encoding function

    Parameters
    ----------
    multires : log2 of max freq for positional encoding, i.e., (L-1)
    i : set 1 for default positional encoding, 0 for none
    input_dims : input dimension of gamma


    Returns
    -------
        embedding function; output_dims
    """
    if is_embeded == False:
        return nn.Identity(), input_dims

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])+0.0001

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    if torch.any(torch.isnan(x)):
        print(r)
        print(q)
        pdb.set_trace()

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    if torch.any(torch.isnan(L)):
        print(s)
        pdb.set_trace()
    return L

def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    if torch.any(torch.isnan(actual_covariance)):
        pdb.set_trace()
    return actual_covariance

@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max

class GaussRenderer(nn.Module):

    def __init__(self, bs_location,white_bkgd=True, **kwargs):
        super(GaussRenderer, self).__init__()
        self.debug = False
        self.white_bkgd = white_bkgd
        self.bs=bs_location
        self.rsrp_coord=torch.arange(0,32).unsqueeze(-1).to('cuda')
        self.embed_depth_bs_fn, self.input_depth_bs_dim = get_embedder(10, True, 3)
    
        
    def build_rsrp(self, means3D, shs,degree, position_grids):
        rays_o = position_grids
        rays_d = means3D[None,:,:] - rays_o[:,None,:]
        rays_d_normarlized=rays_d/rays_d.norm(dim=2,keepdim=True)
        rsrp = eval_sh(degree, shs.permute(0,2,1), rays_d_normarlized)
        rsrp = (rsrp + 0.5).clip(min=0.0)
        return rsrp

    
    def render(self,position_grid, means2D, cov2d, rsrp, opacity,phi_o, depths,embedded):
        batch_size,_=position_grid.shape
        self.render_rsrp = torch.empty(0).to('cuda')
        self.render_depth = torch.empty(0).to('cuda')
        self.rsrp_coord=torch.stack(torch.meshgrid(torch.arange(4), torch.arange(8), indexing='xy'), dim=-1).to('cuda')

        sorted_depths, index = torch.sort(depths.squeeze(-1),dim=1)

        

        means2D=means2D.expand(batch_size,2000,2).cuda()
        cov2d=cov2d.expand(batch_size,2000,2,2).cuda()
    
        opacity=opacity.squeeze().expand(batch_size,-1).cuda()
        phi_o=phi_o.squeeze().expand(batch_size,-1,-1).cuda()
      
        index_unsqueezed = index.unsqueeze(-1).expand(-1, -1, 2)
        sorted_means2D = torch.gather(means2D, 1, index_unsqueezed)
        index_expanded = index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2)
        sorted_conic = torch.gather(cov2d, 1, index_expanded) 
    
     
        sorted_opacity = torch.gather(opacity, 1, index) 
     

        if rsrp.shape[0]!=batch_size:
            rsrp_=rsrp.unsqueeze(0).expand(batch_size,-1,-1).cuda()
        else:
            print('SH Degree>=1')
            rsrp_=rsrp

      
        
        index_ = index.unsqueeze(-1).expand(-1, -1, rsrp_.size(2)).cuda()
        sorted_rsrp = torch.gather(rsrp_, 1, index_)
        sorted_phi_o_ = torch.gather(phi_o, 1, index_) 
        

        dx = (self.rsrp_coord[None,None,:,:].flatten(0,-2) - sorted_means2D[:,:,None,:]) 
        
      
        gauss_weight = torch.exp(-0.5 * (
                    dx[:,:, :, 0]**2 * sorted_conic[:,:, 0, 0].unsqueeze(-1) 
                    + dx[:,:, :, 1]**2 * sorted_conic[:,:, 1, 1].unsqueeze(-1) 
                    + dx[:,:,:,0]*dx[:,:,:,1] * sorted_conic[:,:, 0, 1].unsqueeze(-1) 
                    + dx[:,:,:,0]*dx[:,:,:,1] * sorted_conic[:,:, 1, 0].unsqueeze(-1))*0.001)
      

        sorted_opacity_=sorted_opacity.unsqueeze(-1).expand(-1,-1,32).cuda()
   
        embedded_=embedded.unsqueeze(0).expand(batch_size,-1,32).cuda()
       
        embedded_sorted=torch.gather(embedded_, 1, index_) 

        sorted_opacity__=sorted_opacity_*embedded_sorted

        alpha=(sorted_opacity__*gauss_weight**2).clip(max=0.99)*torch.exp(1j * sorted_phi_o_)

        T=torch.cat([torch.ones_like(alpha[:,:1,:]),1-alpha[:,:-1,:]],dim=1).cumprod(dim=1)

        final_acc_alpha=(alpha*T).sum(dim=1)

        self.render_rsrp=(T * alpha * sorted_rsrp).sum(dim=1).reshape(batch_size,32)+ (1-final_acc_alpha) * (1 if self.white_bkgd else 0)

        self.render_rsrp_=torch.abs(self.render_rsrp)**2
 
        if torch.any(torch.isnan(self.render_rsrp)):
            pdb.set_trace()


        return   self.render_rsrp_
        
   
    def forward(self, model,position_grid,eval=False, **kwargs):
        batchsize, _ = position_grid.shape
        opacity,phi_o = model.get_opacity
        scales = model.get_scaling
        rotations = model.get_rotation
        shs,phi = model.get_features
        linear1,linear2,linear3=model.get_project
        # bias = model.get_bias
        if eval:
            T,TT=model.get_selection_matrix_eval
            means3d,means3D,bias,bb = model.get_xyz_eval
        else:
            T,TT=model.get_selection_matrix
            means3d,means3D,bias,bb = model.get_xyz
        
        if USE_PROFILE:
            prof = profiler.record_function
        else:
            prof = contextlib.nullcontext

        direction=self.bs[None,:]-means3D
        embedded_depths_bs=self.embed_depth_bs_fn(direction)

        embedded_feature=linear3(embedded_depths_bs)

        depths = torch.norm(means3D[None,:,:] - position_grid[:,None,:], dim=2, keepdim=True) #bactch_size*the number of cloud point*1
        
       
        with prof("build rsrp"):
            rsrp_ = self.build_rsrp(means3D=means3D, shs=shs,degree=model.active_sh_degree,  position_grids=position_grid)
            rsrp=rsrp_*torch.exp(1j * phi)


        
        with prof("build cov3d"):
            cov3d = build_covariance_3d(scales, rotations) #the number of cloud point*3*3
            
        with prof("build cov1d"):####Different from CV tasks
            cov3d_=cov3d.reshape(2000,-1)
            cov2d=linear2(cov3d_).reshape(2000,2,2)

            means2D_=linear1(means3D).squeeze()
            means2D=((means2D_)*torch.tensor([3,7]).cuda())
 
  

        
        with prof("render"):
            rets = self.render(
                position_grid = position_grid, 
                means2D=means2D,
                cov2d=cov2d,
                rsrp=rsrp,
                opacity=opacity, 
                phi_o=phi_o,
                depths=depths,
                embedded=embedded_feature,
            )

        return rets,means3D,means3d,bias,bb,T,TT
