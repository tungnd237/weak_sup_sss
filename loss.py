import torch
import torch.nn as nn

def cal_mix_loss(gt, pred, label):
    recon_loss = gt - torch.sum(pred)
    mute_loss = pred*label
    mix_loss = recon_loss + mute_loss
    return mix_loss


def cal_frame_loss(gt_label, pred_label):
    criterion = nn.CrossEntropyLoss()
    frame_loss = criterion(gt_label, pred_label)
    return frame_loss