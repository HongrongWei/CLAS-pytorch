import numpy as np
import SimpleITK as sitk
from skimage.transform import resize

def sample_frames(image_sequence, info_dir, T=10):
    # Note that in the camus dataset, each A2C/A4C sequences have been normalized (begin from ED phase and end with ES phase, or reverse)
    # temporally down sampling (10 frames)
    ori_frame_num = image_sequence.shape[0]
    gap = (ori_frame_num - 2) / (T - 2)
    sample_id = np.zeros(ori_frame_num)
    # get ED and ES frames
    sample_id[0], sample_id[-1] = 1, 1
    if ori_frame_num > T:
        for i in range(T - 2):
            j = np.int(np.round(gap * (i + 1)))
            sample_id[j] = 1
    else: sample_id[:] = 1 # in camus dataset, all sequences have no less than 10 frames
    sample_sequence = image_sequence[sample_id > 0]
    if int((open(info_dir, 'r')).read().split('\n')[0].split(': ')[-1]) != 1:  # Once the sequences were in the manner of ES-ED, reverse it
        sample_sequence = np.flipud(sample_sequence)
    return sample_sequence

def resize_images(imgs, target_shape):
    '''
    :param image sequence with shape: T * H1 * W1
    :param target_shape: (H, W)
    :return: image sequence with shape: T * H * W
    '''
    C = imgs.shape[0]
    H, W = target_shape[0], target_shape[1]
    re_imgs = np.zeros((C, H, W),dtype=np.float32)
    for i in range(C):
        re_imgs[i, :, :] = resize(imgs[i, :, :], output_shape = (H, W), order=1, mode='constant', preserve_range=True, anti_aliasing=True)
    return re_imgs

def resize_gt(imgs, target_shape):
    C = imgs.shape[0]
    W,H = target_shape[0],target_shape[1]
    re_imgs = np.zeros((C, W, H),dtype=np.long)
    for i in range(C):
        re_imgs[i, :, :] = resize(imgs[i, :, :], output_shape = (W, H), order=0, mode='constant', preserve_range=True, anti_aliasing=False)
    return re_imgs

def normalize_images(imgs):
    for i in range(imgs.shape[0]):
        min = np.min(imgs[i])
        max = np.max(imgs[i])
        imgs[i] = ((imgs[i] - min) / (max - min) - 0.5) / 0.5
    return imgs

def read_preprocess_sequences(dir, pat_num=450, T=10):
    ch2 = np.zeros((pat_num, T, 256, 256), dtype=np.float32)
    ch4 = np.zeros((pat_num, T, 256, 256), dtype=np.float32)
    for i in range(pat_num):
        pat = str(i + 1).zfill(4)
        # Chamber 2
        ch2_s = sitk.GetArrayFromImage(sitk.ReadImage(dir + pat + '/patient' + pat + '_2CH_sequence.mhd'))
        ch2_s = sample_frames(ch2_s, info_dir = dir + pat + '/Info_2CH.cfg', T=T)
        ch2[i] = resize_images(ch2_s, [256, 256])
        ch2[i] = normalize_images(ch2[i])
        # Chamber 4
        ch4_s = sitk.GetArrayFromImage(sitk.ReadImage(dir + pat + '/patient' + pat + '_4CH_sequence.mhd'))
        ch4_s = sample_frames(ch4_s, info_dir = dir + pat + '/Info_4CH.cfg', T=T)
        ch4[i] = resize_images(ch4_s, [256, 256])
        ch4[i] = normalize_images(ch4[i])
    return ch2, ch4

def read_process_EDES_gt(dir, pat_num=450):
    ch2_gt = np.zeros((pat_num, 2, 256, 256), dtype=np.long)
    ch4_gt = np.zeros((pat_num, 2, 256, 256), dtype=np.long)
    for i in range(pat_num):
        pat = str(i + 1).zfill(4)
        # Chamber 2
        CH2_ED_gt_dir = dir + pat + '/patient' + pat + '_2CH_ED_gt.mhd'
        CH2_ED_gt = sitk.GetArrayFromImage(sitk.ReadImage(CH2_ED_gt_dir))
        CH2_ES_gt_dir = dir + pat + '/patient' + pat + '_2CH_ES_gt.mhd'
        CH2_ES_gt = sitk.GetArrayFromImage(sitk.ReadImage(CH2_ES_gt_dir))
        ch2_gt[i] = resize_gt(np.concatenate([CH2_ED_gt, CH2_ES_gt], axis=0), [256, 256])
        # Chamber 4
        CH4_ED_gt_dir = dir + pat + '/patient' + pat + '_4CH_ED_gt.mhd'
        CH4_ED_gt = sitk.GetArrayFromImage(sitk.ReadImage(CH4_ED_gt_dir))
        CH4_ES_gt_dir = dir + pat + '/patient' + pat + '_4CH_ES_gt.mhd'
        CH4_ES_gt = sitk.GetArrayFromImage(sitk.ReadImage(CH4_ES_gt_dir))
        ch4_gt[i] = resize_gt(np.concatenate([CH4_ED_gt, CH4_ES_gt], axis=0), [256, 256])
    return ch2_gt, ch4_gt

if __name__ == '__main__':
    A2C_train, A4C_train = read_preprocess_sequences('../data/training/patient', pat_num=450)
    A2C_train_gt, A4C_train_gt = read_process_EDES_gt('../data/training/patient', pat_num=450)
    A2C_test, A4C_test = read_preprocess_sequences('../data/testing/patient', pat_num=50)
    np.save('../data/A2C_train.npy', A2C_train)
    np.save('../data/A4C_train.npy', A4C_train)
    np.save('../data/A2C_train_gt.npy', A2C_train_gt)
    np.save('../data/A4C_train_gt.npy', A4C_train_gt)
    np.save('../data/A2C_test.npy', A2C_test)
    np.save('../data/A4C_test.npy', A4C_test)