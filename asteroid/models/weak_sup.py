import scipy as sp
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from .x_umx import _STFT, _ISTFT, _Spectrogram

class Separator(nn.Module):
    def __init__(self,input_size, hidden_size, num_layer, num_sources, window_length, in_chan, n_hop, spec_power, nb_channels, sample_rate):
        super(Separator, self).__init__()
        stft = _STFT(window_length=window_length, n_fft=in_chan, n_hop=n_hop, center=True)
        spec = _Spectrogram(spec_power=spec_power, mono=(nb_channels == 1))
        self.get_spec = nn.Sequential(stft, spec) # Return: Spec, Angle
        self.decoder = _ISTFT(window = stft.window, n_fft = in_chan, hop_length = n_hop, center = True)

        self.num_sources = num_sources
        self.sample_rate = sample_rate
        self.hidden_size = hidden_size if hidden_size else 600
        self.num_layers = num_layer if num_layer else 3
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, batch_first = False, bidirectional =True)   #input = (T, dim)  output  = (T,dim,num_sources)
        self.classifier = Classifier(input_size, hidden_size=100, num_layer=2)  #TODO: input_size
       
    def apply_masks(self, mixture, spec_mask):
        masked_tf_rep = torch.stack([mixture * spec_mask[:, :, :, i] for i in range(self.num_sources)]) #source_mask = (time, batch, freq)
        return masked_tf_rep

    def forward_masker(self, mixture):

        time, batch, freq = mixture.shape
        fc = nn.Linear(self.hidden_size*2, mixture.shape[-1]*self.num_sources)

        h0 = torch.zeros(self.num_layers*2, mixture.size(1), self.hidden_size)
        c0 = torch.zeros(self.num_layers*2, mixture.size(1), self.hidden_size)

        lstm_out, _ = self.lstm(mixture,(h0,c0)) #(time, batch, 1200) 
        out = fc(lstm_out)
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
        mixture, ang = self.get_spec(wav)   # (time, batch, nb_channels, freq)
        mixture = mixture.squeeze(2) # (time, batch, freq)

        spec_mask = self.forward_masker(mixture)  
                               
        masked_mixture = self.apply_masks(mixture, spec_mask).permute(0, 3, 2, 1).unsqueeze(3) 
        spec_out = masked_mixture.permute(0, 2, 3, 1, 4) 
        wave_out = self.decoder(spec_out, ang)  

        #score = self.classifier(masked_mixture)

        return masked_mixture, wave_out


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer):
        super(Classifier, self).__init__()
        
        self.hidden_size = hidden_size if hidden_size else 100
        self.num_layers = num_layer if num_layer else 2

        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, batch_first = False, bidirectional =True)

    def forward(self, masked_mixture):
        fc = nn.Linear(self.hidden_size*2, masked_mixture.shape[-1])

        h0 = torch.zeros(self.num_layers*2, masked_mixture.size(2), self.hidden_size)
        c0 = torch.zeros(self.num_layers*2, masked_mixture.size(2), self.hidden_size)

        score_lst = []
        for i in range(masked_mixture.shape[0]):
            single_spec = masked_mixture[i]
            lstm_out, _ = self.lstm(single_spec,(h0,c0))
            single_score = fc(lstm_out)
            score_lst.append(single_score)
        score_lst = torch.Tensor(score_lst)
        return score_lst
