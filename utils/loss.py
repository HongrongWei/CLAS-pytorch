import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).cuda()
    result = result.scatter_(1, input, 1)
    return result

class oneclass_DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(oneclass_DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        assert predict.shape == target.shape, "predict & target shape don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(torch.pow(predict, 2) + torch.pow(target, 2), dim=1) + self.smooth
        loss = 1 - num / den
        return torch.mean(loss)

class hard_DiceLoss(nn.Module): # For images that have ground truth
    '''
    input:
        predict shape: batch_size * class_num * H * W
        target shape: batch_size * H * W
    '''
    def __init__(self, n_class = 4):
        super(hard_DiceLoss, self).__init__()
        self.loss = oneclass_DiceLoss()
        self.n_class = n_class

    def forward(self, predict, target):
        target = make_one_hot(target.unsqueeze(dim=1), self.n_class)
        assert predict.shape == target.shape, "predict & target shape don't match"
        total_loss = 0
        for i in range(target.shape[1]):
            total_loss += self.loss(predict[:, i], target[:, i])
        return total_loss / target.shape[1]

class soft_DiceLoss(nn.Module): # To keep consistency between cardiac segmentation and tracking
    '''
    input:
        predict1 & predict2 shape: batch_size * class_num * t * H * W
    '''
    def __init__(self):
        super(soft_DiceLoss, self).__init__()
        self.loss = oneclass_DiceLoss()

    def forward(self, predict1, predict2):
        assert predict1.shape == predict2.shape, "predict1 & predict2 shape don't match"
        total_loss = 0
        for i in range(predict1.shape[1]):
            total_loss += self.loss(predict1[:, i], predict2[:, i])
        return total_loss / predict1.shape[1]

class cross_correlation_loss(nn.Module):
    '''
    Input:
        I & J : true X_i & pseudo X_i (forward/backward tracking)
    '''
    def __init__(self):
        super(cross_correlation_loss, self).__init__()

    def forward(self, I, J, n=9):
        sum_filter = torch.ones((1, 1, n, n)).cuda()
        I2 = torch.mul(I, I)
        J2 = torch.mul(J, J)
        IJ = torch.mul(I, J)
        I_sum = torch.conv2d(I, sum_filter, padding = n // 2, stride = (1, 1))
        J_sum = torch.conv2d(J, sum_filter, padding = n // 2, stride = (1, 1))
        I2_sum = torch.conv2d(I2, sum_filter, padding = n // 2, stride = (1, 1))
        J2_sum = torch.conv2d(J2, sum_filter, padding = n // 2, stride = (1, 1))
        IJ_sum = torch.conv2d(IJ, sum_filter, padding = n // 2, stride = (1, 1))
        win_size = n ** 2
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        cc = cross * cross / (I_var * J_var + 1e-5)
        return -torch.mean(cc)

class smooothing_loss(nn.Module):
    def __init__(self):
        super(smooothing_loss, self).__init__()
    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
        dx = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.mul(dx, dx)
        dy = torch.mul(dy, dy)
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0

class clas_loss(nn.Module):
    def __init__(self, n_class = 4):
        super(clas_loss, self).__init__()
        self.CEloss = nn.CrossEntropyLoss()
        self.hard_Diceloss = hard_DiceLoss()
        self.soft_Diceloss = soft_DiceLoss()
        self.ccloss = cross_correlation_loss()
        self.smloss = smooothing_loss()
        self.n_class = n_class

    def warp_label(self, label, flow):
        warp_label_class = list()
        for i in range(self.n_class):
            warp_label_class.append(F.grid_sample(label[:, i:i+1], flow))
        return torch.cat(warp_label_class, dim=1)

    def forward(self, x, pred, gt):
        '''
        Args:
            x: imgs shape: [batch_size, 1, T, H, W]
            pred:
                seg_prob shape: [batch_size, 4, T, H, W] # class: background, endo, myo, la
                deformation field (forward & backward) shape: [batch_size, T-1, H, W, 2(x&y)]
            gt: gt shape: [batch_size, 2(ED&ES), H, W]
        '''
        T = x.size(2)
        prob, flow_forward, flow_backward = pred[0], pred[1], pred[2]
        ## SGA - appearance level segmentation
        SGA_celoss = (self.CEloss(prob[:,:,0], gt[:, 0]) + self.CEloss(prob[:,:,-1], gt[:, -1])) / 2
        SGA_diceloss = (self.hard_Diceloss(prob[:,:,0], gt[:, 0]) + self.hard_Diceloss(prob[:,:,-1], gt[:, -1])) / 2
        SGA_loss = SGA_celoss + SGA_diceloss
        ## OTA - appearance level tracking
        # warp original image
        warp_image_forward, warp_image_backward = list(), list()
        for i in range(T-1):
            warp_image_forward.append(F.grid_sample(x[:, :, i], flow_forward[:, i]))
        warp_image_forward = torch.stack(warp_image_forward, dim=2)
        for i in range(T-1):
            warp_image_backward.append(F.grid_sample(x[:, :, i+1], flow_backward[:, i]))
        warp_image_backward = torch.stack(warp_image_backward, dim=2)
        cc_loss_forward, cc_loss_backward, sm_loss_forward, sm_loss_backward = 0, 0, 0, 0
        for i in range(T-1):
            cc_loss_forward += self.ccloss(x[:, :, i+1], warp_image_forward[:, :, i])
            cc_loss_backward += self.ccloss(x[:, :, i], warp_image_backward[:, :, i])
            sm_loss_forward += self.smloss(flow_forward[:, i])
            sm_loss_backward += self.smloss(flow_backward[:, i])
        cc_loss = (cc_loss_forward + cc_loss_backward) / 2 / (T-1)
        sm_loss = (sm_loss_forward + sm_loss_backward) / 2 / (T-1)
        OTA_loss = cc_loss + 10 * sm_loss
        ## SGS & OTS - shape level segmentation & tracking
        # warp label
        gt_ED = make_one_hot(gt[:, 0:1], 4)
        gt_ES = make_one_hot(gt[:, 1: ], 4)
        warp_label_forward, warp_label_backward = list(), list()
        warp_label_forward.append(gt_ED.float())
        warp_label_backward.append(gt_ES.float())
        for i in range(T-1):
            warp_label_forward.append(self.warp_label(warp_label_forward[i], flow_forward[:, i]))
        warp_label_forward = torch.stack(warp_label_forward, dim=2)
        for i in range(T-1):
            warp_label_backward.append(self.warp_label(warp_label_backward[i], flow_backward[:, T-2-i]))
        warp_label_backward = torch.cat(warp_label_backward, dim=2)
        warp_label_backward_rev = warp_label_backward[:,:,::-1]
        SGS_loss = (self.soft_Diceloss(prob[:,:,1:-1], warp_label_forward[:,:,1:-1]) + self.soft_Diceloss(prob[:,:,1:-1], warp_label_backward_rev[:,:,1:-1]))/2
        OTS_loss = (self.hard_Diceloss(warp_label_forward[:,:,-1], gt[:, 1]) + self.hard_Diceloss(warp_label_backward_rev[:,:,0], gt[:, 0]))/2
        return {
            'SGA_loss': SGA_loss,
            'OTA_loss': OTA_loss,
            'SGS_loss': SGS_loss,
            'OTS_loss': OTS_loss
        }
