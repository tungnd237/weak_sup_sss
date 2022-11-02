import torch
import torch.nn as nn
import torch.nn.functional as F
from .x_umx import _STFT, _ISTFT, _Spectrogram


class WeakSupModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, num_sources = 5, window_length = 1024, in_chan = 1024, n_hop = 512, spec_power  = True, nb_channels = 1, sample_rate = 16000):
        super(WeakSupModel, self).__init__()
        self.sep_model = Separator(input_size, hidden_size, num_layer, num_sources = 5, window_length = 1024, in_chan = 1024, n_hop = 512, spec_power  = True, nb_channels = 1, sample_rate = 16000).cuda()
        #self.classifier = Classifier(input_size, hidden_size=100, num_layer=2).cuda() 

    def forward(self, wav):

        score = 0
        spec_out, wave_out = self.sep_model(wav)
        #score = self.classifier(spec_out)

        return spec_out, score, wave_out



# TODO: mel_spectrogram으로 고치기
class Separator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, num_sources = 5, window_length = 1024, in_chan = 1024, n_hop = 512, spec_power  = True, nb_channels = 1, sample_rate = 16000):
        super(Separator, self).__init__()
        stft = _STFT(window_length=window_length, n_fft=in_chan, n_hop=n_hop, center=True)
        spec = _Spectrogram(spec_power=spec_power, mono=(nb_channels == 1))
        self.get_spec = nn.Sequential(stft, spec) # Return: Spec, Angle
        self.decoder = _ISTFT(window = stft.window, n_fft = in_chan, hop_length = n_hop, center = True).cuda()

        self.num_sources = num_sources
        self.sample_rate = sample_rate
        self.hidden_size = hidden_size if hidden_size else 600
        self.num_layers = num_layer if num_layer else 3
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, batch_first = False, bidirectional =True).cuda()   #input = (T, dim)  output  = (T,dim,num_sources)
        self.fc = nn.Linear(self.hidden_size*2, (n_hop+1)*self.num_sources).cuda()
       
    def apply_masks(self, mixture, spec_mask):
        masked_tf_rep = torch.stack([mixture * spec_mask[:, :, :, i] for i in range(self.num_sources)]).cuda() #source_mask = (time, batch, freq)
        return masked_tf_rep

    def forward_masker(self, mixture): 
    
        time, batch, freq = mixture.shape
        
        h0 = torch.zeros(self.num_layers*2, mixture.size(1), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers*2, mixture.size(1), self.hidden_size).cuda()
        
        lstm_out, _ = self.lstm(mixture,(h0,c0)) #(time, batch, 1200) 
        out = self.fc(lstm_out)
        spec_mask = F.sigmoid(out)  
        spec_mask = spec_mask.reshape(time, batch, freq, self.num_sources) #(time, batch, sources*freq) --> (time, batch, freq, sources)
        return spec_mask

    def forward(self, wav):
        """
        spec_mask: (time, batch, freq, sources)
        masked_mixutre: (sources, freq, batch, channel, time)
        spec_out: (sources, batch, channel, freq, time)
        wave_out: (sources, batch, channel, time)
        
        """
        
        mixture, ang = self.get_spec(wav)   # (B, 1, sr*time) --> (time, B, nb_channels, freq)
        mixture = mixture.squeeze(2) # (time, B, freq)
        
        spec_mask = self.forward_masker(mixture.clone()) # (time, batch, freq, n_src)                   
        masked_mixture = self.apply_masks(mixture, spec_mask).permute(0, 3, 2, 1).unsqueeze(3) # (n_src, freq, batch, 1,  time)   
        spec_out = masked_mixture.permute(0, 2, 3, 1, 4) # (n_src, B, 1, freq, time)
    
        wave_out = self.decoder(spec_out, ang)


        return spec_out, wave_out


#
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer):
        super(Classifier, self).__init__()
        
        self.input_size = 513
        self.hidden_size = hidden_size if hidden_size else 100
        self.num_layers = num_layer if num_layer else 2
        self.fc = nn.Linear(self.hidden_size*2, 1).cuda()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, batch_first = False, bidirectional =True).cuda()

    def forward(self, masked_mixture):
        # masked_mixture: (n_src, B, 1, freq, time)
        
        h0 = torch.zeros(self.num_layers*2, masked_mixture.size(1), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers*2, masked_mixture.size(1), self.hidden_size).cuda()

        score_lst = []
        for i in range(masked_mixture.shape[0]):
            single_spec = masked_mixture[i].squeeze(1).permute(2, 0, 1) # (B, freq, time) --> (time, B, freq)
            lstm_out, _ = self.lstm(single_spec,(h0,c0)) # (time, B, 200)
            single_score = self.fc(lstm_out) # (time, B, 1)
            score_lst.append(single_score.squeeze(2).permute(1, 0)) # (B, time)
        score_lst = torch.stack([score for score in score_lst], dim=0).permute(1, 0, 2) # (n_src, B, time)
       
        return score_lst
