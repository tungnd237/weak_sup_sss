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


def cal_total_loss(audio_sources, time_labels, src_pred, score_pred):
    beta1, beta2 = 0.9, 0.999
    mix_loss = cal_mix_loss(audio_sources, src_pred, time_labels)
    frame_loss = cal_frame_loss(time_labels, score_pred)
    total_loss = beta1 * mix_loss + beta2 * frame_loss
    return total_loss