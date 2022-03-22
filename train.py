import os,sys,time,logging,datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from argsParser import getArgsParser,checkArgs
import torch.utils
import torch.utils.checkpoint
from tqdm import tqdm

from data_csv_gao import DataLoader as csv_loader
from transformer_encoder import TransformerModel, loss_function_liu

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
optimizer = optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

