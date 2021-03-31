import sys
import os
import numpy as np
from optparse import OptionParser
import torch.backends.cudnn as cudnn
import torch
from torch import optim
from model import CLAS
from utils.loss import clas_loss
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def adjust_learning_rate(optimizer, lr):
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest = 'epochs', default = 30, type='int', help = 'Epochs num')
    parser.add_option('-b', '--batch-size', dest = 'batchsize', default = 4, type = 'int', help ='Batch size')
    parser.add_option('-l', '--learning-rate', dest = 'lr', default = 1e-4, type='float', help='Learning rate')
    (options, args) = parser.parse_args()
    return options

def draw_loss_curve(loss_dict, epochs):
    plt.figure()
    x = np.arange(epochs) + 1
    plt.plot(x, np.stack(loss_dict['SGA_loss'], axis=0)-1, 'b-', label='SGA_loss-1')
    plt.plot(x, np.stack(loss_dict['OTA_loss'], axis=0)+1, 'g--', label='OTA_loss+1')
    plt.plot(x, np.stack(loss_dict['SGS_loss'], axis=0), 'r-.', label='SGS_loss')
    plt.plot(x, np.stack(loss_dict['OTS_loss'], axis=0), 'c:', label='OTS_loss')
    plt.axvline(10, color='m', linestyle='--', label='Stage2')
    plt.legend()
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss_curve.png')
    plt.close()

def train_net(net,
              epochs,
              batch_size,
              lr):
    print('''
        Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        '''.format(epochs, batch_size, lr))
    loss_fun = clas_loss()
    Epoch_loss_dict = {'SGA_loss':[], 'OTA_loss':[], 'SGS_loss':[], 'OTS_loss':[]}
    reg_params = list()
    reg_params += net.reg.parameters()
    other_params = filter(lambda p: id(p) not in list(map(id, reg_params)), net.parameters())
    optimizer = optim.Adam([{'params': reg_params, 'lr': 0.5 * lr}, {'params': other_params, 'lr': lr}], weight_decay=0.0005)
    for epoch in range(epochs):
        state_train = np.random.get_state()
        np.random.shuffle(train)
        np.random.set_state(state_train)
        np.random.shuffle(train_gt)
        if epoch == 25:  # 25
            adjust_learning_rate(optimizer, 1e-5)
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        Epoch_i_loss_dict = {'SGA_loss': 0, 'OTA_loss': 0, 'SGS_loss': 0, 'OTS_loss': 0}
        for batch_i in range(train.shape[0] // batch_size):
            imgs = train[batch_i * batch_size: (batch_i + 1) * batch_size, ]
            gt = train_gt[batch_i * batch_size: (batch_i + 1) * batch_size, ]
            imgs = torch.from_numpy(imgs).unsqueeze(dim=1).cuda()
            gt = torch.from_numpy(gt).cuda()
            seg_prob, forward_deformation_flow, backward_deformation_flow = net(imgs)
            loss_dict = loss_fun(imgs, [seg_prob, forward_deformation_flow, backward_deformation_flow], gt)
            if epoch < 10: loss = loss_dict['SGA_loss'] + loss_dict['OTA_loss']
            else: loss = loss_dict['SGA_loss'] + loss_dict['OTA_loss'] + 0.2 * loss_dict['SGS_loss'] + 0.4 * loss_dict['OTS_loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key, k_data in loss_dict.items():
                Epoch_i_loss_dict[key] += k_data.item()
                print('Epoch : {}/{} --- batch : {}/{} --- {} : {}'.format(epoch + 1, epochs, batch_i + 1, train.shape[0] // batch_size, key, k_data.item()))
        for key, k_data in Epoch_i_loss_dict.items():
            Epoch_loss_dict[key].append(k_data / (train.shape[0] // batch_size))
            print('Epoch finished ! {} : {}'.format(key, k_data / (train.shape[0] // batch_size)))
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(net.state_dict(), './checkpoints/' + 'CP{}.pth'.format(epoch + 1))
        print('Checkpoint saved !')
    draw_loss_curve(Epoch_loss_dict, epochs)

if __name__ == '__main__':
    # load data
    A2C = np.load('./data/A2C.npy')
    A4C = np.load('./data/A4C.npy')
    A2C_gt = np.load('./data/A2C_gt.npy')
    A4C_gt = np.load('./data/A4C_gt.npy')
    train, train_gt = np.concatenate([A2C, A4C], axis=0), np.concatenate([A2C_gt, A4C_gt], axis=0)
    args = get_args()
    net = CLAS()
    net.cuda()
    cudnn.benchmark = True
    try:
        train_net(net = net,
                  epochs = args.epochs,
                  batch_size = args.batchsize,
                  lr = args.lr)
    except KeyboardInterrupt:
        print('Interrupt!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
