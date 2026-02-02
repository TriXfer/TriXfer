
import numpy as np
import torch
import torchvision 
import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas
from datetime import datetime
from torchvision.models import wide_resnet50_2
import torch.optim as optim
import pickle
import time
import os
from art.attacks.evasion import ProjectedGradientDescent
import sys

sys.path.append('/shared_ssd_storage/kittechen/Maggie/NEWCARD')

from algorithm.args import *
from algorithm.tools import AttackPGD,evaluate,generate_atk,accuracy,fpandfn1,precision_score1, recall_score1, f1_score1
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 
from cardv2.helper.loopsv2 import validate,train_distill,DistillKL
from cardv2.crd.criterionv2 import CRDLoss
import types



def lr_decay(epoch, total_epoch,args):
        if args.lr_schedule == 'piecewise':
            if total_epoch == 200:
                epoch_point = [100, 150]
            elif total_epoch == 110: 
                epoch_point = [100, 105] # Early stop for Madry adversarial training
            elif total_epoch ==50:
                epoch_point = [25, 40]
            else:
                epoch_point = [int(total_epoch/2),int(total_epoch-5)]
            if epoch < epoch_point[0]:
                if args.warmup_lr and epoch < args.warmup_lr_epoch:
                    return 0.001 + epoch / args.warmup_lr_epoch * (args.lr-0.001)
                return args.lr
            if epoch < epoch_point[1]:
                return args.lr / 10
            else:
                return args.lr / 100
        elif args.lr_schedule == 'cosine':
            if args.warmup_lr:
                if epoch < args.warmup_lr_epoch:
                    return 0.001 + epoch / args.warmup_lr_epoch * (args.lr-0.001)
                else:
                    return np.max([args.lr * 0.5 * (1 + np.cos((epoch-args.warmup_lr_epoch) / (total_epoch-args.warmup_lr_epoch) * np.pi)), 1e-4])
            return np.max([args.lr * 0.5 * (1 + np.cos(epoch / total_epoch * np.pi)), 1e-4])
        elif args.lr_schedule == 'constant':
            return args.lr
        else:
            raise NotImplementedError  
def forward(self, x: torch.Tensor):
    # Reshape and permute the input tensor
    x = self._process_input(x)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    x = self.encoder(x)

    # Classifier "token" as used by standard language architectures
    x = x[:, 0]
    feature = x
    x = self.heads(x)

    return x , feature
def card(model_s,model_t,train_loader,test_loader,criterion,numofclass,device,args):
    #---------maggie add-----------
    print("device:",device)
    model_s = model_s.to(device)  
    model_t = model_t.to(device) 
    #---------maggie add-----------

    best_acc = 0
    opt = torch.optim.SGD(model_s.parameters(), args.lr, 
                            weight_decay=args.weight_decay,
                            momentum=args.momentum)
    minv = 0
    maxv = 0
    # Get the shape of a single sample
    n_data = len(train_loader.dataset)
    # Combine to infer overall shape
    overall_shape = 0
    if not args.is_sample: 
        for data, label in train_loader:
            x = torch.max(data).item()
            y = torch.min(data).item()
            overall_shape = data.shape[1:]
            maxv = x if max == 0 else max(maxv, x)
            minv = y if min == 0 else min(minv, y)
    else:
        for data, label,index,contraindex in train_loader:
            x = torch.max(data).item()
            y = torch.min(data).item()
            overall_shape = data.shape[1:]
            maxv = x if max == 0 else max(maxv, x)
            minv = y if min == 0 else min(minv, y)
    print(minv,maxv)
    model_t.forward = types.MethodType(forward, model_t)
    classifier = generate_atk(model_s,criterion=criterion,optimizer=opt,min=minv,max=maxv,shape= overall_shape,number=numofclass)
    attack = ProjectedGradientDescent(estimator=classifier,norm=args.norm, eps=args.ae_eps/255,eps_step=args.ae_step/255,targeted = args.target,max_iter= args.ae_ite)
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = nn.KLDivLoss()
    batch = next(iter(train_loader))
    s = batch[0].shape[1:]
    print(s)
    data = torch.rand(1, *s)
    pholder = model_s(data.to(device))
    args.s_dim = model_s.feature.shape[1]
    args.n_data = n_data
    
    # _,feature = model_t(data)
    #---------maggie add-----------
    _,feature = model_t(data.to(device)) 
    #---------maggie add-----------
    
    args.t_dim = feature.shape[1]
        
    criterion_kd = CRDLoss(args,s)
    module_list.append(criterion_kd.embed_s)
    module_list.append(criterion_kd.embed_t)
    trainable_list.append(criterion_kd.embed_s)
    trainable_list.append(criterion_kd.embed_t)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss
    print(len(trainable_list))
    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # routine
    std = []
    adv = []
    dataframetime = pandas.DataFrame({'seed':[],'epochs':[],'time':[]})
    time0 = time.time()
    for epoch in range(1, args.n_epochs + 1):
        lr_current = lr_decay(epoch, args.n_epochs,args)
        optimizer.param_groups[0].update(lr=lr_current)
        print(f"==> {epoch}/{args.n_epochs}-epoch training start...")

        time1 = time.time()
        # easy to say CUDA out of memory
        train_acc, train_loss,df = train_distill(epoch, train_loader, module_list, criterion_list, optimizer, args,dataframetime,s,numofclass)
        time2 = time.time()
        dataframetime = df
        print('epoch {}, total time {:.2f} trainacc{}'.format(epoch, time2 - time1,train_acc))

        

        test_acc, adv_test, test_loss = validate(test_loader, model_s, criterion_cls, numofclass, args,attack,numofclass)
        print(f"Test Accuracy:{test_acc}, Test Loss:{test_loss}, Adv Accuracy: {adv_test} ")
        # if (epoch+1)% args.n_eval_step ==0:
        #     std.append(test_acc)
        #     adv.append(adv_test)
        #     with open('Historyforplotting/card'+"/standardhistory.pkl","wb") as file:
        #         pickle.dump(std,file)
        #     with open('Historyforplotting/card' +"/advhistory.pkl","wb") as file:
        #         pickle.dump(adv,file)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(args.model_folder, '{}_gamma{}_beta{}_cof{}best_on{}.pth'.format(args.target_model,args.gamma,args.beta,args.cof,args.target_dataset))
            print('saving the best model!')
            torch.save(state, save_file)
    time3 = time.time()
    print(f'total time used: {time3 - time0}')
    print('best accuracy:', best_acc)
    dataframetime.to_csv(f'result/Contrastive/recordfortime_seed{args.seed}_task_{args.task_name}',index=False)
    # save model
    state = {
        'opt': args,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(args.model_folder, '{}_gamma{}_beta{}last_on{}.pth'.format(args.target_model,args.gamma,args.beta,args.target_dataset))
    torch.save(state, save_file)
