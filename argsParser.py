# Args parser: define and parse all runtime settings
# by: Jiayu Yang
# date: 2019-10-02
# a sample modified by Shiyu Gao

import argparse

def getArgsParser():
    parser = argparse.ArgumentParser(description='Ubiquant')
    

    parser.add_argument('--lrepochs', type=str, default="10,12,14,20:2", help='epoch ids to downscale lr and the downscale rate')
    parser.add_argument('--end_epochs', type=int, default=40, help='number of epochs to train')
    parser.add_argument('--start_epochs', type=int, default=0, help='number of epochs to train')
    parser.add_argument('--summary_freq', type=int, default=1, help='print and summary frequency')
    parser.add_argument('--save_freq', type=int, default=2, help='save checkpoint frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='train batch size')
    parser.add_argument('--mode', default='train', help='train or test ro validation', choices=['train', 'test', 'val'])
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--logckptdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
    parser.add_argument('--info', default='None', help='Info about current run')
    
    return parser


def checkArgs(args):
    # Check if the settings is valid
    assert args.mode in ["train", "val", "test"]
    if args.resume:
        assert len(args.loadckpt) == 0
    if args.loadckpt:
        assert args.resume is 0