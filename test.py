import os
from model import CLAS
import torch
from utils.read_data import *
from utils.post_process import imfill_select_whole_seq
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
net = CLAS()
net.cuda()
net.load_state_dict(torch.load('./checkpoints/CP30.pth'))
net.eval()
ch2 = np.load('./data/A2C_test.npy')
ch4 = np.load('./data/A4C_test.npy')
for pat in range(50):
    print('Patient : {}'.format(pat + 1))
    imgs_ch2 = ch2[pat:pat+1,]
    imgs_ch4 = ch4[pat:pat+1,]
    imgs_ch2 = torch.from_numpy(imgs_ch2).unsqueeze(1).cuda()
    imgs_ch4 = torch.from_numpy(imgs_ch4).unsqueeze(1).cuda()
    seg_ch2, _, _, = net(imgs_ch2)
    seg_ch4, _, _, = net(imgs_ch4)
    ch2_mask = np.array((torch.argmax(seg_ch2, dim=1)).cpu())
    ch4_mask = np.array((torch.argmax(seg_ch4, dim=1)).cpu())
    ch2_mask_post = imfill_select_whole_seq(ch2_mask) # hole filling & Non-maximum connected domain suppression
    ch4_mask_post = imfill_select_whole_seq(ch4_mask)





