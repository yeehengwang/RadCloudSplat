import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
from shutil import copyfile
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from gaussian_model_radcloudsplat import GaussModel 
from gaussian_render_radcloudsplat import GaussRenderer
from datasets import RSRP_dataset
import utils as utils
import loss_utils as loss_utils
import time
import torch.nn as nn
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class SmoothLoss(nn.Module):
    def __init__(self, lambda_tv=1.0):
        super(SmoothLoss, self).__init__()
        self.lambda_tv = lambda_tv
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target, coords):
        l2_loss = self.mse_loss(pred, target)
        tv_loss = self.total_variation(pred, coords)
        loss = l2_loss + self.lambda_tv * tv_loss
        return loss

    def total_variation(self, pred, coords):
        batch_size = coords.size(0)
        dist_matrix = torch.cdist(coords, coords)  
        knn_idx = dist_matrix.topk(2, largest=False).indices[:, 1]  
        # pdb.set_trace()
        diff = pred - pred[knn_idx]
        tv_loss = torch.mean(torch.norm(diff, p=1, dim=-1))
        return tv_loss
    

class DGS_Runner():
    def __init__(self,mode,**kwargs):

        kwargs_path = kwargs['path']

        kwargs_train = kwargs['train']

        kargs_newtrain=kwargs['newtrain']

        self.expname = kwargs_path['expname']

        if mode != 'test':
            self.batch_size = kwargs_train['batch_size']
        else:
            self.batch_size=1
        # self.datadir = kwargs_path['datadir']
        self.logdir = kwargs_path['logdir']
        self.devices = torch.device('cuda')
        self.points_path='./demo_data/pc6k_farthest_downsampled_points.txt'
        self.bs=torch.tensor([5.0, 0.26, 0]).float().to(self.devices)
        self.world_size=30
        self.bs=self.bs/self.world_size
       
        self.train_index_path='./demo_data/train_index_indoor.txt'
        self.test_index_path='./demo_data/test_index_indoor.txt'
        print("Loading training set...")
        self.train_set = RSRP_dataset(self.train_index_path, self.world_size)
        print("Loading test set...")
        self.test_set = RSRP_dataset(self.test_index_path, self.world_size)
        # pdb.set_trace()
        self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        print(f"Train set size:%d, Test set size:%d", len(self.train_set), len(self.test_set))

        self.points=torch.tensor(np.loadtxt(self.points_path)).float().to(self.devices)/self.world_size

        self.points_=torch.tensor(np.loadtxt(self.points_path)).float().to(self.devices)/self.world_size



        self.gaussianmodel=GaussModel(debug=False,mode=mode).to(self.devices)
        self.M=2000
        self.gaussianmodel.create_from_pcd(self.points,self.M,self.bs)


        for name, param in self.gaussianmodel.named_parameters():
            if param.requires_grad:
                print(f"Updating {name} with size {param.size()}")

        # self.scaler = GradScaler()

        self.optimizer=self.gaussianmodel.training_setup(kargs_newtrain)


        self.GaussRenderer=GaussRenderer(self.bs)
        self.current_iteration = 0
  
        if kwargs_train['load_ckpt'] or mode == 'test':
            self.load_best_checkpoints()  
        self.batch_size = kwargs_train['batch_size']
        self.total_iterations = kwargs_train['total_iterations']
        self.save_freq = kwargs_train['save_freq']

    def load_checkpoints(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts_radcloudsplat')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        print('Found ckpts %s', ckpts)

        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Loading ckpt %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.devices)

            self.gaussianmodel.load_state_dict(ckpt['gaussianmodel_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.current_iteration = ckpt['current_iteration']
        
    def save_checkpoint(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts_radcloudsplat')
        model_lst = [x for x in sorted(os.listdir(ckptsdir)) if x.endswith('.tar')]
        if len(model_lst) > 2:
            print(model_lst)
            os.remove(ckptsdir + '/%s' % model_lst[0])

        ckptname = os.path.join(ckptsdir, '{:06d}.tar'.format(self.current_iteration))
    
        torch.save({
            'current_iteration': self.current_iteration,
            'gaussianmodel_state_dict': self.gaussianmodel.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, ckptname)
        return ckptname
    def save_best_checkpoint(self,degree):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts_radcloudsplat')
        model_lst = [x for x in sorted(os.listdir(ckptsdir)) if x.endswith('.pth')]
        if len(model_lst) > 0:
            print(model_lst)
            os.remove(ckptsdir + '/%s' % model_lst[0])

        ckptname = os.path.join(ckptsdir, 'best_model_{:06d}_SH_{:d}.pth'.format(self.current_iteration,degree))
        torch.save({
            'current_iteration': self.current_iteration,
            'gaussianmodel_state_dict': self.gaussianmodel.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, ckptname)
        return ckptname
    def load_best_checkpoints(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts_radcloudsplat')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'pth' in f]
        print('Found ckpts %s', ckpts)


        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Loading ckpt %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.devices)

            self.gaussianmodel.load_state_dict(ckpt['gaussianmodel_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.current_iteration = ckpt['current_iteration']

    def eval_network_rsrp(self):
        MAE=0
        ii=0
        times=[]
        flag=-1
        for test_input, test_label in tqdm(self.test_iter):
            # print('/')
            
            ii+=1
            position_grid=test_input.to(self.devices)
            self.gaussianmodel.eval()
            t1=time.time()
            out,_,_,_,_,_,_=self.GaussRenderer(model=self.gaussianmodel,position_grid=position_grid,eval=True)
            t2=time.time()
            t_gap=(t2-t1)*1000
            times.append(t_gap)
            rsrp_dB=out*30-50
            rsrp_GT_dB=test_label
            mae=np.mean(abs(rsrp_dB.detach().cpu().numpy() - rsrp_GT_dB.detach().cpu().numpy()))
            print('==================Evaluating MAE(dB):',mae)
            MAE+=mae
        
        MAE_mean=MAE/ii
        print('==================================')
        print('Total Test Mean MAE(dB):',MAE_mean)
        print('Total Test Mean Time(ms):',np.mean(times))
        print('==================================')
        return MAE_mean


    def train(self):
        Losss=[]
        MAE=[]
        bestb_loss=1e10
        maeb_loss=1e10
        patient_i=0
        mae_=1e10
        flag=0
        indictor=0
        sh_degree=0

        while self.current_iteration <= self.total_iterations:
            loss_saved=0
            mae_saved=0
            ll=0
            l=0
            
            for train_input, train_label in tqdm(self.train_iter):
                flag=flag+1

                ll+=self.batch_size
                l+=1
                position_grid=train_input.to(self.devices)
                self.gaussianmodel.train()
                rsrp_GT=(train_label.to(self.devices)+50)/30
                self.optimizer.zero_grad()
                out,pcs,pcs_,bias,bb,T,TT=self.GaussRenderer(model=self.gaussianmodel,position_grid=position_grid)
            
                bs_regulariztion_term=torch.mean(torch.norm(pcs_-self.bs,dim=1, keepdim=True))
                bias_regulariztion_term=torch.norm(bias)
             
                l1_T=torch.norm(T,p=1)
                term4=torch.norm(T.max(1).values-1)

                criterion = SmoothLoss(lambda_tv=0.2)
                
                loss = criterion(out, rsrp_GT, position_grid)*15+0.03*bs_regulariztion_term+bias_regulariztion_term*0.01+l1_T+term4*17
                total_loss=loss

             
                self.optimizer.zero_grad() 
                total_loss.backward()
                self.optimizer.step()

                self.current_iteration += 1
                # self.gaussianmodel.update_learning_rate(self.current_iteration)
                # Every 500 s we increase the levels of SH up to a maximum degree
                # if self.current_iteration % 1000 == 0:
                if self.current_iteration % 500 == 0:
                    if sh_degree<4:
                        sh_degree+=1
                    self.gaussianmodel.oneupSHdegree()
             
                loss_saved+=total_loss
                print('SH Degree=',sh_degree)
                rsrp_dB=out*30-50
                rsrp_GT_dB=train_label
                mae=np.mean(abs(rsrp_dB.detach().cpu().numpy() - rsrp_GT_dB.detach().cpu().numpy()))
  
                mae_saved+=mae
           
                
            loss_saved=loss_saved/l
            print('Total Mean LOSS:',loss_saved.detach().cpu().numpy())
            Losss.append(loss_saved.detach().cpu().numpy())
            mae_saved=mae_saved/l
            MAE.append(mae_saved)
            print('Total MAE(dB):',mae_saved)
  

            if loss_saved>=bestb_loss:
                patient_i+=1
    
                if patient_i>=50:
                     print('Early Stop')
                     break
            temp_=False
            
            

            if mae_saved<maeb_loss and temp_==False:
                    if patient_i>0:
                         patient_i=0
                    maeb_loss=mae_saved
                    ckptname = self.save_checkpoint()
                    print('Save at'+ ckptname)

            if True:
                flag=0
                m=self.eval_network_rsrp()
                if m<=mae_:
                    indictor=0
                    mae_=m
                    ckptname = self.save_best_checkpoint(sh_degree)
                    print('Save at'+ ckptname)
                else:
                    print('Continue')
                    if self.current_iteration>=3500:
                        indictor+=1
                        if indictor>=30:
                            print('Overfitting STOP at:',self.current_iteration)
                            break
                    


        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./radcloudsplat_setting.yml', help='config file path')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train') ###If you wanna test, please using 'test'

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    with open(args.config) as f:
        kwargs = yaml.safe_load(f)
        f.close()


    worker = DGS_Runner(mode=args.mode, **kwargs)
    if args.mode == 'train':
        worker.train()
    elif args.mode == 'test':
        worker.eval_network_rsrp()
