from myutils import *
import pdb


def eval_two_volumes_maxpool(root, target, pool_kernel, device):
    pre = read_nifti(root)
    label = read_nifti(target)
    k = pool_kernel
    s = max(1, k - 1)
    kernel = (k, k, k)
    stride = (s, s, s)
    # pre[pre < threshold] = 0
    # pre[pre >= threshold] = 1
    pre[pre > 0] = 1
    label[label > 0] = 1
    # 转Tensor时出现 TypeError: can't convert np.ndarray of type numpy.uint16.
    pre = pre.astype(np.uint8)
    label = label.astype(np.uint8)

    pre = torch.Tensor(pre).view((1, 1, *pre.shape)).to(device)
    # pre = torch_dilation(pre, 5)
    label = torch.Tensor(label).view((1, 1, *label.shape)).to(device)
    # label = torch_dilation(label, 5)

    pre = torch.nn.functional.max_pool3d(pre, kernel, 1, 0)
    label = torch.nn.functional.max_pool3d(label, kernel, 1, 0)

    dice_score = dice_error(pre, label)

    total_loss_iou = iou(pre, label).cpu()
    total_loss_tiou = t_iou(pre, label).cpu()
    recall, acc = soft_cldice_f1(pre, label)
    cldice = (2. * recall * acc) / (recall + acc)

    # print('\n Validation IOU: {:.3f}\n T-IOU: {:.3f}'
    #       '\n ClDice: {:.3f} \n ClAcc: {:.3f} \n ClRecall: {:.3f} \n Dice-score: {:.3f}'
    #       .format(total_loss_iou, total_loss_tiou, cldice, acc, recall, dice_score))
    print('\n Validation IOU: {}\n T-IOU: {}'
          '\n ClDice: {} \n ClAcc: {} \n ClRecall: {} \n Dice-score: {}'
          .format(total_loss_iou, total_loss_tiou, cldice, acc, recall, dice_score, ':.8f'))
    return {'iou': total_loss_iou,
            'tiou': total_loss_tiou,
            'cldice': cldice,
            'acc': acc,
            'recall': recall,
            'score': dice_score}


def eval_two_volume_dirs_maxpool(target, root, data, pool_kernel, threshold, device):
    # pre = read_tiff_stack(join(root, data))
    # label = read_tiff_stack(join(target, data))
    pre = read_nifti(join(root, data))
    label= read_nifti(join(target, data))
    k = pool_kernel
    s = max(1, k - 1)
    kernel = (k, k, k)
    stride = (s, s, s)
    # pre[pre < threshold] = 0
    # pre[pre >= threshold] = 1
    pre[pre > 0] = 1
    label[label > 0] = 1
    # 转Tensor时出现 TypeError: can't convert np.ndarray of type numpy.uint16.
    pre = pre.astype(np.uint8)
    label = label.astype(np.uint8)

    pre = torch.Tensor(pre).view((1, 1, *pre.shape)).to(device)
    # pre = torch_dilation(pre, 5)
    label = torch.Tensor(label).view((1, 1, *label.shape)).to(device)
    # label = torch_dilation(label, 5)

    pre = torch.nn.functional.max_pool3d(pre, kernel, 1, 0)
    label = torch.nn.functional.max_pool3d(label, kernel, 1, 0)

    dice_score = dice_error(pre, label)

    total_loss_iou = iou(pre, label).cpu()
    total_loss_tiou = t_iou(pre, label).cpu()
    recall, acc = soft_cldice_f1(pre, label)
    cldice = (2. * recall * acc) / (recall + acc)

    # print('\n Validation IOU: {}\n T-IOU: {}'
    #       '\n ClDice: {} \n ClAcc: {} \n ClRecall: {} \n Dice-score: {}'
    #       .format(total_loss_iou, total_loss_tiou, cldice, acc, recall, dice_score, '.8f'))
    return {'iou':total_loss_iou,
            'tiou':total_loss_tiou,
            'cldice':cldice,
            'acc':acc,
            'recall':recall,
            'score':dice_score}


def avg(num, total):
    return total / num


if __name__ == "__main__":
    val_root = "/media/root/data4/zyt/validate/155829_nii/155829_label.nii.gz"
    val_target = "/media/root/data4/zyt/validate/155829_nii/test/whole/nnunet705_16G_en/155829_whole.nii.gz"
    # val_root = "/media/root/data4/zyt/validate/162245_nii/162245_label.nii.gz"
    # val_target = "/media/root/data4/zyt/validate/162245_nii/test/whole/nnunet705/162245_whole.nii.gz"
    # val_root = "/media/root/data4/zyt/validate/141410_nii/141410_label.nii.gz"
    # val_root = "/media/root/data4/zyt/validate/141410_nii/141410_label_rect.nii.gz"
    # val_target = "/media/root/data4/zyt/validate/141410_nii/test/whole/nnunet705_s1/141410_whole.nii.gz"
    kernel_size = 10  # maxpooling kernel size
    threshold = 125  # probability threshold of the positive class, no need for nnunet prediction
    device = torch.device('cuda:3')
    loss = eval_two_volumes_maxpool(val_target, val_root, kernel_size, device)
'''
    # data_root = "/media/root/data4/zyt/validate/155829_nii/labels/"
    # data_target = "/media/root/data4/zyt/validate/155829_nii/test/nnunet703/"
    # for p in os.listdir(dataroot):
    #     if ".nii.gz" in p:
    #         num = p.split(".")[0].split("_")[1]
    #         tpath = join(val_target, num+".tiff")
    #         load_nifti_to_tiff(join(data_target, p), tpath)
    iou_1, tiou_1, cldice_1, acc_1, recall_1, dice_score_1 = 0, 0, 0, 0, 0, 0
    count = 0
    for data in sorted(os.listdir(data_target)):
        if ".nii.gz" in data or ".tiff" in data:
            count += 1
            loss = eval_two_volume_dirs_maxpool(data_target, data_root, data, kernel_size, threshold, device)
            iou_1 += loss['iou']
            tiou_1 += loss['tiou']
            cldice_1 += loss['cldice']
            acc_1 += loss['acc']
            recall_1 += loss['recall']
            dice_score_1 += loss['score']
    iou_t = avg(count, iou_1)
    tiou_t = avg(count, tiou_1)
    cldice_t = avg(count, cldice_1)
    acc_t = avg(count, acc_1)
    recall_t = avg(count, recall_1)
    score_t = avg(count, dice_score_1)
    print('\n Validation IOU: {}\n T-IOU: {}'
          '\n ClDice: {} \n ClAcc: {} \n ClRecall: {} \n Dice-score: {}'
          .format(iou_t, tiou_t, cldice_t, acc_t, recall_t, score_t, '.8f'))
'''
