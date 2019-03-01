'''
Created on Feb 25, 2019

@author: airingzhang
'''
import os
import sys
import shutil
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from net import Net
from mnist_dataset import MNIST
from loss import Loss
from utils import Logger


def preprocess_data(src_dir, split_ratio=0.1):
    if os.path.exists(os.path.join(src_dir, 'training.pt')) and \
        os.path.exists(os.path.join(src_dir, 'testing.pt')) and \
        os.path.exists(os.path.join(src_dir, 'training_split.pt')) and \
        os.path.exists(os.path.join(src_dir, 'validation_split.pt')):
        return 
    MNIST.preprocessing(src_dir, split_ratio)
    
def get_lr(epoch, args):
        if epoch <= args.epochs * 0.1:
            lr = args.lr
        elif epoch <= args.epochs * 0.5:
            lr = args.lr*0.1
            #lr =  args.lr*(1 - epoch*2.0/args.epochs)
        else:
            lr = 0.01 * args.lr
        return lr
    

def train(args, model, train_loader, loss_form, use_cuda, optimizer, epoch):
    model.train()
    lr = get_lr(epoch, args)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data = Variable(data.cuda(async = True))
            target = Variable(target.cuda(async = True))
        else:
            data = Variable(data.cuda(async = True))
            target = Variable(target.cuda(async = True))
        optimizer.zero_grad()
        output = model(data)
        loss = loss_form(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}, lr: {},  [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, lr, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    print('Train Epoch: {}, lr: {},  [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, lr, len(train_loader.dataset), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))

def test(args, model, test_loader, loss_form, use_cuda):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if use_cuda:
                data = Variable(data.cuda(async = True))
                target = Variable(target.cuda(async = True))
            else:
                data = Variable(data.cuda(async = True))
                target = Variable(target.cuda(async = True))
            output = model(data)
            test_loss += torch.sum(loss_form(output, target)).data[0] # sum up batch loss 
            _, pred = torch.max(output, 1)
            correct += pred.eq(target.view_as(pred)).float().sum().data[0]

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def prepare_dataloader(src_dir, args, train_full=False):
    # return the data loader depending on whether to use all training set
    if train_full:
        training_set_path = os.path.join(src_dir, 'training.pt')
        testing_set_path = os.path.join(src_dir, 'testing.pt')
    else:
        training_set_path = os.path.join(src_dir, 'training_split.pt')
        testing_set_path = os.path.join(src_dir, 'validation_split.pt')
        
    stats_file = os.path.join(src_dir, 'stats.npy')
    train_loader = torch.utils.data.DataLoader(
        MNIST(training_set_path, train=True, stats_file=stats_file),
        batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(
        MNIST(testing_set_path, train=False, stats_file=stats_file),
        batch_size=args.test_batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    
    return train_loader, test_loader

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--start-epoch', type=int, default=1, metavar='N',
                        help='starting epoch (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--num-workers', type=int, default=64, metavar='N',
                        help='thread num for dataloader')
     
    parser.add_argument('--save-freq', default=5, type=int, metavar='S',
                        help='save frequency')
    
    parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
    
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')    
    
    parser.add_argument('--data-dir', type=str, default='', metavar='N',
                        help='data dir where raw files located')
    
    parser.add_argument('--split-ratio', type=float, default=0.1, metavar='N',
                        help='split ratio of validation set')
    
    parser.add_argument('--train-full', type=int, default=0, metavar='N',
                        help='indicate if all training set is used for training')
    
    
    
    args = parser.parse_args()
    # pre-process data 
    data_dir = args.data_dir
    split_ratio = args.split_ratio
    preprocess_data(data_dir, split_ratio)
    save_dir = args.save_dir if args.save_dir else data_dir
    
    #set up gpu, use CUDA_VISIBLE_DEVICES to control gpu is often preferable 
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)
    
    train_loader, test_loader = prepare_dataloader(data_dir, args, train_full=args.train_full)
    model = Net()
    loss_form = Loss()
    if use_cuda:
        model.cuda()
        loss_form.cuda()
        
    start_epoch = 1    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']+1
    if args.start_epoch>1:
        start_epoch = args.start_epoch
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    # save logs to file for retrospective checking 
    logfile = os.path.join(save_dir,'log')
    sys.stdout = Logger(logfile)
    
    # save the .py files to retain the scene for possible future comparison      
    pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
    if not os.path.exists(os.path.join(save_dir, 'code')):
        os.mkdir(os.path.join(save_dir, 'code'))
    for f in pyfiles:
        shutil.copy(f, os.path.join(save_dir, 'code', f))

    # training and testing (validation)
    for epoch in range(start_epoch, args.epochs + 1):
        train(args, model, train_loader, loss_form, use_cuda, optimizer, epoch)
        test(args, model, test_loader, loss_form, use_cuda)
        if epoch % args.save_freq == 0:
            state_dict = model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
    
            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'args': args},
                os.path.join(save_dir, '%03d.ckpt' % epoch))

        
if __name__ == '__main__':
    main()
