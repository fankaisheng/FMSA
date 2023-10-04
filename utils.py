import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader,Dataset
import numpy as np
import scipy.io as sio
import torch.optim as optim
from torch.autograd import grad
import traceback
import pandas as pd
from tqdm import tqdm
import random


label_list = {'BENIGN': 0,
              'PortScan': 1,
                'Web': 2,
                'SSH-Patator': 3,
                'FTP-Patator': 4,
                'Bot': 5,
                'DoS Hulk': 6,
                'DoS GoldenEye': 7,
                'DDoS': 8}

# Victim dataloader
class NIDS_dataset(Dataset):
    def __init__(self, file):
        df = pd.read_csv(file, index_col=False)
        self.items = np.array(df.iloc[:,:-1])
        self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
                    np.std(self.items, axis=1, keepdims=True) + 0.00001)
        self.label = np.array(df.iloc[:,-1])

    def __len__(self):
        # print('item的长度是',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print(self.items[idx,:],self.label[idx])
        return self.items[idx, :], self.label[idx]
    
class Victim_dataset(Dataset):
    def __init__(self, file, malware_class):
        df = pd.read_csv(file, index_col=False)
        self.items = np.array(df.iloc[:,:-1])
        self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
                    np.std(self.items, axis=1, keepdims=True) + 0.00001)
        self.label = np.array(df.iloc[:,-1])

        benign_index = np.random.choice(self.label[self.label==0].index, 500, False).tolist()
        malware_index = np.random.choice(self.label[self.label==malware_class].index, 500, False).tolist()
        index = benign_index + malware_index
        self.items = self.items[index]
        self.label = self.label[index]
    
    def __len__(self):
        # print('item的长度是',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print(self.items[idx,:],self.label[idx])
        return self.items[idx, :], self.label[idx]
    

# Victim model
class DNN_NIDS(nn.Module):
    def __init__(self):
        super(DNN_NIDS, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(59, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 9)
        )

    def forward(self, x):
        return self.model(x)

# E
class Extractor(nn.Module):
    def __init__(self):
        super(DNN_NIDS, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(59, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 64),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.model(x)

# G
class Generator(nn.Module):
    def __init__(self, noise_dim=64, attri_dim=64, input_shape=59):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.attri_dim = attri_dim
        self.input_shape = input_shape
        self.modelG = nn.Sequential(
            nn.Linear(self.noise_dim+self.attri_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, self.input_shape)
        )
    
    def forward(self, noise, attri):
        gen_input = torch.cat((attri,noise), -1)
        sample = self.modelG(gen_input)
        return sample

# D
class Discriminator(nn.Module):
    def __init__(self, input_shape=59):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.modelD = nn.Sequential(
            nn.Linear(self.input_shape, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, sample):
        result = self.modelD(sample).mean(0)
        return result


### MAML
class BaseLearner(nn.Module):
    def __init__(self, module=None):
        super(BaseLearner, self).__init__()
        self.module = module
    
    def __getattr__(self, attr):
        try:
            return super(BaseLearner, self).__getattr__(attr)
        except AttributeError:
            return getattr(self.__dict__['_modules']['module'], attr)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def clone_module(module, memo=None):

    if memo is None:
        memo = {}
    if not isinstance(module, torch.nn.Module):
        print("======error!=======")
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()
    
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[buff_ptr] = cloned

    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
            
    return clone


def update_module(module, updates=None, memo=None):
    if memo is None:
        memo = {}
        
    if updates is not None:
        params = list(module.parameters())
        if not len(updates) == len(params):
            msg = 'WARNING:update_module(): Parameters and updates have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(updates)) + ')'
            print(msg)
        for p, g in zip(params, updates):
            p.update = g 
            
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p in memo:
            module._parameters[param_key] = memo[p]
        else:
            if p is not None and hasattr(p, 'update') and p.update is not None:
                updated = p + p.update
                p.update = None
                memo[p] = updated
                module._parameters[param_key] = updated
                
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff in memo:
            module._buffers[buffer_key] = memo[buff]
        else:
            if buff is not None and hasattr(buff, 'update') and buff.update is not None:
                updated = buff + buff.update
                buff.update = None
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    for module_key in module._modules:
        module._modules[module_key] = update_module(module._modules[module_key],
                      updates=None,
                      memo=memo,
        )
        
    if hasattr(module, 'flatten_parameters'):
        module._apply(lambda x: x)
    return module

def maml_update(model, lr, grads=None):
    i = 0
    if grads is not None:
        params = list(model.parameters())
        if not len(grads) == len(list(params)):
            msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
            print(msg)
        for p,g in zip(params, grads):
            if g is not None:
                p.update = -lr *g
    return update_module(model) 


class MAML(BaseLearner):
    def __init__(self, model, lr=1e-3, 
                 first_order=False,
                 allow_unused=None,
                 allow_nograd=False):
        super(MAML, self).__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def adapt(self, loss, 
              first_order=None,
              allow_unused=None,
              allow_nograd=None):
        global gradients

        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order
        
        
        if allow_nograd:
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            grad_params = grad(loss,
                               diff_params,
                               retain_graph=second_order,
                               create_graph=second_order,
                               allow_unused=allow_unused)
            gradients = []
            grad_counter = 0
            
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(loss,
                             self.module.parameters(),
                             retain_graph=second_order,
                             create_graph=second_order,
                             allow_unused=allow_unused)
            except RuntimeError:
                traceback.print_exc()
                print('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')
            # Update the module
        self.module = maml_update(self.module, self.lr, gradients)
    
    def clone(self, lr=None, first_order=None, allow_unused=None, allow_nograd=None):
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        if lr is None:
            lr = self.lr
        return MAML(clone_module(self.module),
                    lr=lr,
                    first_order=first_order,
                    allow_unused=allow_unused,
                    allow_nograd=allow_nograd)
    

# FMSA
class FMSA_dataset(Dataset):

    def __init__(self, data_dir, batchsz=100, n_way=2, k_shot=5, k_query=15):
        self.dataset = pd.read_csv(data_dir)
        self.batchsz = batchsz
        self.n_way = n_way  
        self.k_shot = k_shot  
        self.k_query = k_query  
        self.data_cache = self.load_data_cache()

        self.items = np.array(self.dataset.iloc[:,:-1])
        self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
                    np.std(self.items, axis=1, keepdims=True) + 0.00001)
        self.label = np.array(self.dataset.iloc[:,-1])

    def load_data_cache(self):
        data_index = {}
        for key in label_list.keys():
            label_name = label_list[key]
            label_index = self.dataset.loc[self.dataset[' Label']==label_name].index
            data_index[label_name] = label_index
        labels = np.array(range(0,9))
        data_cache = []
        for batch in range(self.batchsz):
            support_index, query_index = [], []
            malwar_label = np.random.choice(labels[1:], 1, False)[0]
            choice_label = [0, malwar_label]
            for choice in choice_label:
                selected_index = np.random.choice(data_index[choice], self.k_shot+self.k_query, False).tolist()
                support_index += selected_index[:self.k_shot]
                query_index += selected_index[self.k_shot:]
            random.shuffle(support_index)
            random.shuffle(query_index)
            data_cache.append([support_index, query_index])
        return data_cache

    def __len__(self):
        return self.batchsz

    def __getitem__(self, index):
        support_index, query_index = self.data_cache[index]
        support_x,support_y = self.items[support_index],self.label[support_index]
        query_x,query_y = self.items[query_index],self.label[query_index]
        return (support_x, support_y, query_x, query_y)

class FMSA(nn.Module):
    def __init__(self,meta_E, meta_G, meta_D, meta_C, victim_dir):
        super().__init__()
        self.update_lr = 0.01
        self.meta_lr = 1e-3
        self.victim_dir = victim_dir
        self.meta_E = meta_E
        self.meta_G = meta_G
        self.meta_D = meta_D
        self.meta_C = meta_C
        self.meta_optimD = optim.RMSprop(self.meta_D.parameters(), lr = self.meta_lr)
        gc_para=list(self.meta_E.parameters())+list(self.meta_G.parameters())
        self.meta_optimEG = optim.Adam(gc_para, lr=self.meta_lr,betas=(0.9, 0.99))
        self.meta_optimC = optim.Adam(self.meta_C, lr=self.meta_lr,betas=(0.9, 0.99))
        self.criterion = nn.BCELoss()
        self.n_way = 2
        self.noise_dim = 64
        self.attri_dim = 64
        self.input_shape = 59
    

    def train_victim_model(self, label1, label2):
        if label1 == 0:
            victim_data = Victim_dataset(self.victim_dir, label1)
        else:
            victim_data = Victim_dataset(self.victim_dir, label2)
        victim_loader = DataLoader(victim_data, batch_size=32, shuffle=True)
        victim_model=DNN_NIDS()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(victim_model.parameters(), lr=0.001)
        for data, label in victim_loader:
            pred = victim_model(data.float())
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return victim_model
    
    def forward(self, x_spt, y_spt, x_qry, y_qry, meta_traing=False):
        batchsz, supportsz, input_shape = x_spt.shape
        querysz = x_qry.shape[1]
        

        for batch in range(batchsz):
            label1, label2 = torch.unique(y_spt.view(-1))
            victim_model = self.train_victim_model(label1, label2)
            # criterion = nn.CrossEntropyLoss()
            # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            D_loss, E_loss, G_loss1, G_loss2 = 0,0,0,0
            task_E = self.meta_G.clone(lr=self.update_lr)
            task_G = self.meta_G.clone(lr=self.update_lr)
            task_D = self.meta_D.clone(lr=self.update_lr)
            task_C = self.meta_C.clone(lr=self.update_lr)
            support_x = x_spt[batch]
            support_y = y_spt[batch]
            query_x = x_qry[batch]
            query_y = y_qry[batch]

            # inner loop
            real_data = support_x.to(torch.float32)
            errD_real = task_D(real_data)
            extract_feature = task_E(real_data)
            noise = torch.FloatTensor(supportsz, self.noise_dim).normal_(0, 1)
            fake_data = task_G(noise, extract_feature)
            errD_fake = task_D(fake_data)
            errD = errD_real - errD_fake
            task_D.adapt(-1*errD)

            noise = torch.FloatTensor(supportsz, self.noise_dim).normal_(0, 1)
            fake_data = task_G(noise, extract_feature)
            errG1 = task_D(fake_data)
            pred1 = victim_model(fake_data)
            pred2 = task_C(fake_data)
            errG2 = self.criterion(pred1, pred2)
            errG = errG1 + errG2
            errC = -errG2
            task_G.adapt(errG)
            task_E.adapt(errG)
            task_C.adapt(errC)

            # outer loop
            real_data = query_x.to(torch.float32)
            errD_real = task_D(real_data)
            extract_feature = task_E(real_data)
            noise = torch.FloatTensor(querysz, self.noise_dim).normal_(0, 1)
            fake_data = task_G(noise, extract_feature)
            errD_fake = task_D(fake_data)
            errD = errD_real - errD_fake

            noise = torch.FloatTensor(querysz, self.noise_dim).normal_(0, 1)
            fake_data = task_G(noise, extract_feature)
            errG1 = task_D(fake_data)
            pred1 = victim_model(fake_data)
            pred2 = task_C(fake_data)
            errG2 = self.criterion(pred1, pred2)
            errG = errG1 + errG2
            errC = -errG2
            
            if meta_traing:
                self.meta_optimD.zero_grad()
                self.meta_optimEG.zero_grad()
                
                errD.backward()
                errG.backward()
                errC.backward()
                self.meta_optimD.step()
                self.meta_optimEG.step()
                self.meta_optimC.step()
