import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
import scipy.io as sio
from torchattacks import PGD, FGSM, BPDA
import numpy as np
# from models.binarized_modules import BinarizeLinear,BinarizeConv2d

# torch.backends.cudnn.benchmark = False
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
                    
                    
parser.add_argument('-pf', '--prob_file', type=str, default='../Prob_data/prob_0p6_0p3_linear_chip7.mat',
                    metavar='FILE', help='probability table mat file')
parser.add_argument('--ternary', type=bool, default=False,
                    help='if True, step size is 1')
                    
parser.add_argument('--sd', default=0.0, type=float, help='momentum')                    
                    
#RRAM:
# '/home/syin11/shared/pythonsim/XNORNET_SRAM/prob_64_11_0111.mat'                    
#noise related arguments  

#prob_0p8_0p4_linear.mat

#prob_1p0_0p5_linear.mat         
parser.add_argument('--ideal_ADC', type=int, default=0,
                    help='use single math model for quantization of partial sums')
                    
parser.add_argument('--common_prob', type=int, default=0,
                    help='use single math model for quantization of partial sums')
parser.add_argument('--bitwise_prob', type=int, default=0, help='use bit-wise probabilities (including noise) for quantization')
parser.add_argument('--noise_inject', type=bool, default=False, help='if True, noise injection is included in common_prob')
parser.add_argument('--trpgd', type=int, default=0, help='if True, pgd attack generated inputs are used for evaluation')
parser.add_argument('--tspgd', type=int, default=0, help='if True, pgd attack generated inputs are used for evaluation')
parser.add_argument('--chunkwise', type=int, default=0, help='if True, chunkwise compute is performed')
parser.add_argument('--savepgd', type=int, default=0, help='if True, pgd attack generated inputs are saved')
parser.add_argument('--pgditers', type=int, default=10, help='if True, pgd attack generated inputs are saved')
parser.add_argument('--anb', type=int, default=1, help='if True, pgd attack generated inputs are saved')
parser.add_argument('--wnb', type=int, default=1, help='if True, pgd attack generated inputs are saved')
parser.add_argument('--pni', type=int, default=0, help='if True, pgd attack generated inputs are saved')

def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()
    
    
    trpgd = args.trpgd 
    tspgd = args.tspgd     
    savepgd = args.savepgd
    pgditers = args.pgditers   
    
    #####################end noise specific code###########################################
    if args.evaluate:
        args.results_dir = '../results_eval/'
    if args.save == '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset, 'act_precision': args.anb}

    if args.model_config != '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)
    # logging.info(model)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()

#        1. filter out unnecessary keys
        # 
        # for k, v in pretrained_dict.items():
            # # print(k,v)
            # if k not in model_dict:
                # print('herpies')
                # print(f'key: {k} | val: {v}')
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        
        w = list(model.parameters())
        
        with open('weights_vgg_pni.txt','w') as f:
            for k, v in pretrained_dict.items():
                if 'weight' in k:
                    # f.write("Layer {}".format(k))
                    f.write(str(v.std().item()))
                    f.write('\n')
        
        
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(pretrained_dict)
        
        
                
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
                     

    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})
    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    # criterion = nn.NLLLoss()
    criterion.type(args.type)
    model.type(args.type)
    
    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    
    # if savepgd and pgd:
            # savepgd_attack = PGD(model, eps=0.031, alpha=2/255, iters=7) 
            # # inputs=savepgd_attack(val_data, target)
        
            # savepgd_attack.save(data_loader=val_loader, file_name="cifar10_pgd.pt", accuracy=True)
    
    if args.evaluate:
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0, pgd=tspgd, savepgd=savepgd, pgditers=pgditers)
        logging.info('\n'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(val_loss=val_loss, val_prec1=val_prec1, 
                             val_prec5=val_prec5))
        return

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    logging.info('training regime: %s', regime)


    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)
        
        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer, pgd=trpgd, savepgd=savepgd, pgditers=pgditers, sd=args.sd)

        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, epoch, pgd=tspgd, savepgd=savepgd, pgditers=pgditers, sd = args.sd)

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'regime': regime
        }, is_best, path=save_path)
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                    train_error5=100 - train_prec5, val_error5=100 - val_prec5)
        #results.plot(x='epoch', y=['train_loss', 'val_loss'],
        #             title='Loss', ylabel='loss')
        #results.plot(x='epoch', y=['train_error1', 'val_error1'],
        #             title='Error@1', ylabel='error %')
        #results.plot(x='epoch', y=['train_error5', 'val_error5'],
        #             title='Error@5', ylabel='error %')
        results.save()


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None, pgd=False, savepgd=False, pgditers=10, sd = 0):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus[0])
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    
    if pgd:
        # print("hippyty hoppity1")
        pgd_attack = PGD(model, eps=0.031, alpha=0.008, iters=pgditers)
        # pgd_attack = BPDA(model, step_size = 1., iters = pgditers, linf=False)
        
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        
        data_time.update(time.time() - end)

        inputs = inputs + torch.randn_like(inputs) * sd
        # print(inputs.max(), inputs.min())
        
        if pgd: 
            # print("hippyty hoppity2")
            inputs_adversarial=pgd_attack(inputs, target)
        
        
        if args.gpus is not None:
            target = target.cuda()
        
        with torch.no_grad():
            input_var = Variable(inputs.type(args.type))
            
            if pgd:
                input_adversarial_var = Variable(inputs_adversarial.type(args.type))   
        
        
        target_var = Variable(target)
        # compute output
        output = model(input_var)
                  

        loss = criterion(output, target_var)
        
        if pgd:
            # print("hippyty hoppity3")
            output_adversarial = model(input_adversarial_var)
        
            loss_adversarial = criterion(output_adversarial, target_var)
        
            loss = 0.5*(loss + loss_adversarial)
        
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        if pgd:
            # print("hippyty hoppity4")
            prec1, prec5 = accuracy(output_adversarial.data, target, topk=(1, 5))
        else:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer, pgd, savepgd,pgditers=10, sd = 0):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer, pgd=pgd, savepgd=savepgd, pgditers=pgditers, sd = sd)


def validate(data_loader, model, criterion, epoch, pgd, savepgd, pgditers=10, sd = 0):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None, pgd=pgd, savepgd=savepgd, pgditers=pgditers, sd = sd)


if __name__ == '__main__':
    main()
