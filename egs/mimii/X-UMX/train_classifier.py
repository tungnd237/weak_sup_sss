import torch
import torch.nn as nn
import torchaudio

import numpy as np
import yaml
import random
import glob
import os
import tqdm
from asteroid.models import WeakSupModel, Classifier
from asteroid.models.x_umx import _STFT, _Spectrogram
import wandb
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

wandb.init(project="weak_sup")

class UrbanSoundDataset(torch.utils.data.Dataset):

    def __init__(self, file_path, class_id_lst, audio_dir, target_sample_rate = 16000, target_time = 4, source_random = True):
        self.file_path_lst = file_path
        self.class_id_lst = class_id_lst
        self.audio_dir = audio_dir 
        self.target_sample_rate = target_sample_rate
        self.target_time = target_time
        self.source_random = source_random
        self.data = {index: {} for index in range(len(self.file_path_lst))}

    def __len__(self):
        return len(self.file_path_lst)

    def __getitem__(self, index):
        
        if len(self.data[index]) == 0:
            # TODO: adpative pooling dimension 정리
            # set adaptivepooling 
            m = nn.AdaptiveAvgPool1d(126)
            # assemble the mixture of target
            audio_sources = {src: torch.zeros(1, self.target_sample_rate* self.target_time).cuda() for src in self.class_id_lst}
            time_labels = {src: torch.zeros(1, 126).cuda() for src in self.class_id_lst}

            # load sources
            audio_target_path, class_id_target = self._get_audio_sample_path(index, self.source_random)
        
            for idx, source in enumerate(class_id_target):

                # load, downsample, convert to mono
                audio, sr = torchaudio.load(audio_target_path[idx])
                audio = self._resample_if_necessary(audio, sr, self.target_sample_rate)
                audio = torch.mean(audio, dim = 0).reshape(1, -1).cuda()
                
                # randomly select the starting point and truncate if it's longer than target length
                # generate time label
                start = random.randint(0, self.target_sample_rate)
                end = start + audio.shape[1]

                if end > self.target_sample_rate * self.target_time:
                    end_truncated = end - self.target_sample_rate * self.target_time
                    audio_sources[source][:, start:] = audio[:, :-end_truncated]
                    time_labels[source][:, start:] = 1
                else:
                    audio_sources[source][:, start:end] = audio
                    time_labels[source][:, start:end] = 1
                time_labels[source] = m(time_labels[source])
            
                assert audio_sources[source].shape[1] == self.target_sample_rate * self.target_time

                # generate time label
                # TODO: define the function which generate time labels

            # generate mixture
            audio_mix = torch.stack(list(audio_sources.values())).sum(0)
            audio_sources = torch.stack([audio_sources[src] for src in self.class_id_lst], dim=0)
            time_labels = torch.stack([label for label in time_labels.values()], dim=0)
                
        
            # convert class_id_target to binary label
            binary_class_label = torch.zeros(len(self.class_id_lst))

            for idx, id in enumerate(self.class_id_lst):
                if id in class_id_target:
                    binary_class_label[idx] = 1


            self.data[index] = {"audio_mix": audio_mix,
                            "audio_sources": audio_sources,
                            "time_labels": time_labels,
                            "binary_class_label": binary_class_label,}     
            
        else:
            audio_mix = self.data[index]['audio_mix']
            audio_sources = self.data[index]['audio_sources']
            time_labels = self.data[index]['time_labels']
            binary_class_label = self.data[index]['binary_class_label']

        return audio_mix, audio_sources, time_labels, binary_class_label


    def _resample_if_necessary(self, audio, sr, target_sample_rate):        
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
        return audio


    def _get_audio_sample_path(self, index, source_random = True):
        target_path_lst = []
        class_id_lst_ = self.class_id_lst.copy()
        anchor_path = self.file_path_lst[index]
        fold = os.path.split(os.path.split(anchor_path)[0])[1]
        anchor_id = int((os.path.split(anchor_path)[1]).split('-')[1])

        if source_random:
            num_src = random.randint(0, 4)
            class_id_lst_.remove(anchor_id)
            class_id_target = random.sample(class_id_lst_, num_src)
            
            for id in class_id_target:
               path = random.choice(glob.glob(os.path.join(self.audio_dir, fold) + '/*-{id}-*'.format(id = id)))
               target_path_lst.append(path)

            target_path_lst.append(anchor_path)
            class_id_target.append(anchor_id)
 
        else:
            class_id_target = class_id_lst_
            for id in class_id_target:
                path = random.choice(glob.glob(os.path.join(self.audio_dir, fold) + '/*-{id}-*'.format(id = id)))
                target_path_lst.append(path)

        class_id_target.sort() 
        return target_path_lst, class_id_target

    def _get_time_label(self, audio):
        time_label = None
        return time_label


def file_path_generator(audio_dir, class_id_lst):

    # 1: car_horn, 3: dog_bark, 6: gun_shot, 7: jackhammer, 8: siren
    # train_fold: 1-6, eval_fold: 7,8
    train_files = []
    eval_files = []

    train_fold = [os.path.join(audio_dir, 'fold{num}'.format(num = num)) for num in range(1,7)]
    eval_fold = [os.path.join(audio_dir, 'fold{num}'.format(num = num)) for num in [7,8]]

    # set train and eval files
    train_files = load_file_path(train_fold, class_id_lst)
    eval_files = load_file_path(eval_fold, class_id_lst)
    return train_files, eval_files


def load_file_path(fold, class_id_lst):
    files = []
    for fold_path in fold:
        for class_id in class_id_lst:
            for track_path in tqdm.tqdm(glob.glob(f'{fold_path}/*-{class_id}-*-*.wav')):
                files.append(track_path)

    return files


##########################################################################
# calculate loss

def cal_mix_loss(audio_mix_gt, pred, binary_class_label):
    # audio_mix_gt(wav): (B, 1, sr*time)
    # mix_spec: (time, B, 1, freq)
    # pred:(n_src, B, 1, freq, time) --> (B, n_src, freq, time)
    # binary_class_label: (B, n_src)
   
    # change dimension
    pred = pred.squeeze(2).permute(1, 0, 2, 3)

    # transform ground truth waveform to spectrogram
    stft = _STFT(window_length=1024, n_fft=1024, n_hop=512, center=True)
    spec = _Spectrogram(spec_power=True, mono = True)
    get_spec = nn.Sequential(stft, spec).cuda() # Return: Spec, Angle

    mix_spec, _ = get_spec(audio_mix_gt)  # (time, B, 1, freq)
    mix_spec = mix_spec.squeeze(2).permute(1, 2, 0)  # (B, freq, time)
    freq, time = mix_spec.shape[1], mix_spec.shape[2]

    mix_target = torch.zeros(1, freq, time).cuda()
    mute_target = torch.zeros(1, freq, time).cuda()

    num_batch = binary_class_label.shape[0]
    for batch_idx in range(num_batch):
        class_label = binary_class_label[batch_idx, :].reshape(-1, 1, 1)
        mix_target_single = torch.sum(class_label*pred[batch_idx, :, :, :], dim = 0).unsqueeze(0)
        
        mute_label = torch.logical_not(class_label)
        mute_target_single = torch.sum(torch.abs(mute_label*pred[batch_idx, :, :, :]), dim = 0).unsqueeze(0)     
        
        mix_target = torch.cat([mix_target, mix_target_single], dim = 0)
        mute_target = torch.cat([mute_target, mute_target_single], dim = 0)
    mix_target = mix_target[1:, :, :]
    mute_target = mute_target[1:, :, :]

    recon_loss = torch.abs(mix_spec - mix_target)
            
    mix_loss = torch.mean(recon_loss + mute_target)
   
    return mix_loss


def cal_frame_loss(gt_label, pred_label):
    # gt_label: [B, n_src, 1, time] --> [B, n_src, time]
    gt_label = gt_label.squeeze(2)
    num_src = gt_label.shape[1]
    criterion = nn.CrossEntropyLoss()
    frame_loss = 0

    for src in range(num_src):
        frame_loss += criterion(gt_label[:, src, :], pred_label[:, src, :])
    frame_loss = torch.mean(frame_loss)
    return frame_loss


def cal_total_loss(audio_mix, time_labels, src_pred, score_pred, binary_class_label):
    beta1, beta2 = 0.9, 0.999
    mix_loss = cal_mix_loss(audio_mix, src_pred, binary_class_label)
    frame_loss = cal_frame_loss(time_labels, score_pred)
    total_loss = beta1 * mix_loss + beta2 * frame_loss

    return total_loss

def cal_wav_sup_loss(audio_sources, src_pred):
    # audio_sources: (B, n_src, 1, sr*time) --> (B, n_src, sr*time)
    # src_pred: (n_src, B, 1, sr*time) 

    # set dimension as the same
    audio_sources = audio_sources.squeeze(2) 
    src_pred = src_pred.permute(1, 0 ,2, 3).squeeze(2)

    sup_loss = torch.mean(torch.abs(audio_sources - src_pred))

    return sup_loss


def cal_spec_sup_loss(audio_sources, src_pred, param):
    # audio_sources: (B, n_src, 1, sr*time) --> (B, n_src, sr*time)
    # src_pred: (n_src, B, 1, freq, time) --> (n_src, B, freq, time)

    get_spec = nn.Sequential(
            _STFT(window_length=param['window_length'], n_fft=param['window_length'], n_hop=param['n_hop']),
            _Spectrogram(spec_power=True, mono=1)).cuda() 

    total_loss = 0
    criterion = nn.L1Loss()
    # convert audio source wav to spectrogram
    n_src = audio_sources.shape[1]
    audio_sources = audio_sources.squeeze(2)
    src_pred = src_pred.squeeze(2)
    
    for src in range(n_src):
        gt_spec_src, _ = get_spec(audio_sources[:, src, :].unsqueeze(1))
        gt_spec_src = gt_spec_src.squeeze(2).permute(1, 2, 0) # (batch, freq, time)
        total_loss += criterion(src_pred[src, :, :,], gt_spec_src)

    return total_loss

##########################################################################

def sisnr(x, s, eps=1e-8):
    
    """
    calculate training loss
    input:

          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))

    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)

    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)

    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))




if __name__ == "__main__":
    with open("weaksup.yaml") as stream:
        param = yaml.safe_load(stream)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"]= param['gpu']

    print("========== Dataset Generator ==========")
    class_id_lst = [1, 3, 6, 7, 8] 
    audio_dir = param["audio_dir"]
    
    train_files, eval_files = file_path_generator(audio_dir, class_id_lst)
    train_dataset = UrbanSoundDataset(train_files, class_id_lst, audio_dir, source_random = param['source_random'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = param["batch_size"], shuffle = True, drop_last = True)

    eval_dataset = UrbanSoundDataset(eval_files, class_id_lst, audio_dir, source_random = param['source_random'])
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size = param["batch_size"], shuffle = False)
   
    print("========== Model Training ==========")
    input_size = param["input_size"]
    hidden_size = param["hidden_size"]
    num_layer = param["num_layer"]
    epoch_val = param["epoch_val"]
    
    model = Classifier(input_size, hidden_size, num_layer).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = param['lr'])
    loss_fn = cal_frame_loss
    
    get_spec = nn.Sequential(
            _STFT(window_length=param['window_length'], n_fft=param['window_length'], n_hop=param['n_hop']),
            _Spectrogram(spec_power=True, mono=1)).cuda() 

    for epoch in range(1, param["epochs"]):
        print("epoch:", epoch)
        i = 1
        losses = []
        for batch in train_loader:
            audio_sources = batch[1].cuda() # (B, n_src, 1, sr*time)
            time_labels = batch[2].cuda() # (B, n_src, 1, sr*time)

            n_src = audio_sources.shape[1]
            audio_sources = audio_sources.squeeze(2)
            temp, _ = get_spec(audio_sources[:, 0, :].unsqueeze(1))
            audio_sources_spec = torch.zeros_like(temp)
            print(audio_sources_spec.shape)
            for i in range(n_src):
                audio_sources_spec_temp, _ = get_spec(audio_sources[:, i, :].unsqueeze(1))
                audio_sources_spec = torch.stack([audio_sources_spec,audio_sources_spec_temp], dim = 0)
            audio_sources_spec = audio_sources_spec[1:, :, :, :]
            audio_sources_spec = audio_sources_spec.cuda()
            print("GT:", audio_sources_spec.shape)
            optimizer.zero_grad()
            

            score_pred = model(audio_sources_spec) 
            loss = loss_fn(time_labels, score_pred)
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            i+=1
        wandb.log({"Train Loss:": sum(losses) / len(losses)})
        if epoch % 10 == 0:
            print(f"epoch {epoch}: loss {sum(losses) / len(losses)}")
        
        
        if epoch % epoch_val == 0:
            print("========== Model Evaluation ==========")
            model.eval()

            total_eval_loss = 0

            si_sdr = {src: 0 for src in class_id_lst}
            
            for batch in eval_loader:
                audio_sources = batch[1].cuda() # (B, n_src, 1, sr*time)
                time_labels = batch[2].cuda() # (B, n_src, 1, sr*time)

                n_src = audio_sources.shape[1]
                audio_sources = audio_sources.squeeze(2)
                audio_sources_spec = torch.stack([get_spec(audio_sources[:, i, :].unsqueeze(1)) for i in range(n_src)]).cuda() #source_mask = (time, batch, freq)

                score_pred = model(audio_sources_spec) 
                total_eval_loss += loss_fn(time_labels, score_pred)/len(eval_loader)


            # Add audio to wandb
            wandb.log({"eval_loss:": total_eval_loss})
            time_labels = time_labels.squeeze(2)
            score_pred = score_pred.squeeze(2)
            for idx, src in enumerate(class_id_lst):
                wandb.log({f"gt_label_{src}": time_labels[0, idx, :].detach().cpu().numpy(),
                f"pred_label_{src}": score_pred[0, idx, :].detach().cpu().numpy(),})
            model.train()
