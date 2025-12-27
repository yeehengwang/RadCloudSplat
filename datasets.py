import os
import random
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm
import pdb

class RSRP_dataset(Dataset):

    def __init__(self, indexdir, scale_worldsize=1):
        super().__init__()

        self.rsrpdata_dir='./demo_data/RSRP_data.npy'
        self.location='./demo_data/location.npy'

        self.dataset_index = np.loadtxt(indexdir)-1



        self.rx_poses = torch.from_numpy(np.load(
            self.location))  
        self.rx_poses = self.rx_poses / scale_worldsize

        self.RSRPs = (torch.from_numpy(np.load(
            self.rsrpdata_dir)))
       


        self.nn_inputs, self.nn_labels = self.load_data()

    def load_data(self):
        """load data from datadir to memory for training

        Returns
        --------
        nn_inputs : tensor. [n_samples, 3]. The inputs for training
                    position_grid:3

        nn_labels : tensor. [n_samples, n_rsrp = 32]. The RSRP as labels
        """
        ## NOTE! Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 3)), dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((len(self), 32)), dtype=torch.float32)

    
        data_counter = 0
        for idx in tqdm(self.dataset_index, total=len(self.dataset_index)):  # sample from dataset_index
            idx=int(idx)

         
            nn_inputs[data_counter] = self.rx_poses[idx]
            nn_labels[data_counter] = self.RSRPs[idx]  # 40
            data_counter += 1

        return nn_inputs, (nn_labels)

      

    def __len__(self):

        return len(self.dataset_index)  

    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index] 
