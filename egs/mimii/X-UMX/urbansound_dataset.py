import torch
import jams
import os
import tqdm
import glob
import torchaudio

class UrbanSoundDenoisedDataset(torch.utils.data.Dataset):

    def __init__(self, train, audio_dir, target_sample_rate = 16000, sources = ['car_horn', 'dog_bark' ,'gun_shot', 'jackhammer', 'siren'], target_time =4):
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.target_time = target_time
        self.sources =  sources
        self.train = train
        self.mix_file_path = self.load_mix_path(self.audio_dir, self.train)
        self.data = {index: {} for index in range(len(self.mix_file_path))}
        

    def __len__(self):
        return len(self.mix_file_path)

    def __getitem__(self, index):
     
        if len(self.data[index]) == 0:
            src_lst = []
            
            audio_sources = {src: torch.zeros(1, self.target_sample_rate* self.target_time).cuda() for src in self.sources}
            time_labels = {src: torch.zeros(1, 126).cuda() for src in self.sources}
            
            # load mixture
            mixture_path = self.mix_file_path[index]
            audio_mix, sr = torchaudio.load(mixture_path)
            audio_mix = self._resample_if_necessary(audio_mix, sr, self.target_sample_rate)
            
            # load audio_sources
            event_name = os.path.split(self.mix_file_path[index])[-1].split('.')[0] 
            sources_dir = os.path.join(os.path.split(self.mix_file_path[index])[0], event_name + "_events")
            sources_file_list = [os.path.join(sources_dir, f) for f in os.listdir(sources_dir)]
            
            for src_path in sources_file_list:
                src = os.path.split(src_path)[-1].split('.')[0][12:]
                src_lst.append(src)
                audio, sr = torchaudio.load(src_path)
                audio = self._resample_if_necessary(audio, sr, self.target_sample_rate)
                audio = torch.mean(audio, dim = 0).reshape(1, -1).cuda()
                audio_sources[src] = audio
                
            # Add annotation
            annotation_path = os.path.join(os.path.split(self.mix_file_path[index])[0], event_name + ".jams")
            jam = jams.load(annotation_path)
            annotation_values = jam["annotations"][0]['data']
            source_num = len(annotation_values)
        
            for i in range(source_num):
                time_label_temp = torch.zeros(1, self.target_sample_rate* self.target_time).cuda()
                source_label = annotation_values[i].value['label']
                if source_label in src_lst:
                    start_time = int(annotation_values[i].value['event_time']*self.target_sample_rate)
                    duration = int(annotation_values[i].value['event_duration']*self.target_sample_rate)
                    time_label_temp[:, start_time:start_time+duration] = 1.0
                    time_labels[source_label]= nn.functional.adaptive_avg_pool1d(time_label_temp, 126)
                    
            # generate mixture
            audio_sources = torch.stack([audio_sources[src] for src in self.sources], dim=0)
            time_labels = torch.stack([label for label in time_labels.values()], dim=0)       
                    
            # # convert class_id_target to binary label
            # binary_class_label = torch.zeros(len(self.sources))

            # for idx, source in enumerate(src_lst):
            #     if id in src_lst:
            #         binary_class_label[idx] = 1
                    
            
            self.data[index] = {"audio_mix": audio_mix,
                            "audio_sources": audio_sources,
                            "time_labels": time_labels,}
                            # "binary_class_label": binary_class_label,} 
                
                    
        else:
            audio_mix = self.data[index]['audio_mix']
            audio_sources = self.data[index]['audio_sources']
            time_labels = self.data[index]['time_labels']
            # binary_class_label = self.data[index]['binary_class_label']
                
            
        return audio_mix, audio_sources, time_labels
    
    
    def load_mix_path(self, audio_dir, train = True):
        file_path = []
        set = 'train' if train else 'valid'
        for track_path in tqdm.tqdm(glob.glob(f'{audio_dir}/{set}/*.wav')):
            file_path.append(track_path)
        file_path = file_path[:1]
        return file_path
    
    def _resample_if_necessary(self, audio, sr, target_sample_rate):        
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
        return audio
