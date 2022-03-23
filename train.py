import os,sys,time,logging,datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from utils import *
from argsParser import getArgsParser,checkArgs
import torch.utils
import torch.utils.checkpoint
from tqdm import tqdm

from data_csv_gao import DataLoader as csv_loader
from transformer_encoder import TransformerModel, loss_function_liu, MSE_loss,tcorr

torch.set_num_threads(20)
# Arg parser
parser = getArgsParser()
args = parser.parse_args()
assert args.mode == "train"
checkArgs(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark=True

# Dataset
train_dataset = csv_loader(args)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=16, drop_last=True)

#network
model = TransformerModel()      # model = TransformerModel(args)
# GPU parallel
gpu_list = [2,0,1,3]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
model = nn.DataParallel(model, device_ids=[0,1,2,3])
model.cuda()
model.train()

# model loss
# model_loss = MSE_loss()
model_loss = loss_function_liu()
model_corr = tcorr()
optimizer = optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.999), weight_decay=args.wd)


def train_sample(sample):

    optimizer.zero_grad()

    sample_cuda = tocuda(sample)

    outputs = model(sample_cuda["features"])
    loss = model_loss(outputs,sample_cuda["target"])
    corr = model_corr(outputs,sample_cuda["target"])

    loss.backward()

    optimizer.step()

    return loss.data.cpu().item(),corr


# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=args.start_epoch)      #step 
    last_loss = None
    this_loss = None
    for epoch_idx in range(args.start_epoch, args.end_epochs):
        global_step = len(train_loader) * epoch_idx
        if last_loss is None:
            last_loss = 999999
        else:
            last_loss = this_loss
        this_loss = []

        for batch_idx, sample in enumerate(train_loader):
            start_time = time.time()
            global_step = len(train_loader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            
            loss, corr = train_sample(sample)
            this_loss.append(loss)

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logckptdir+args.info.replace(" ","_"), epoch_idx  ))
        this_loss = np.mean(this_loss)

        lr_scheduler.step()