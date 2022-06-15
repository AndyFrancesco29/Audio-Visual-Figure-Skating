import torch
import torch.utils.data as data
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class FeatureDataset(data.Dataset):
    def __init__(self,
                 root_path,
                 is_train=True,
                 ):

        self.root_path = root_path
        if is_train:
            file_path = root_path + '/train_dataset.txt'
            f = open(file_path, 'r')
            data_info = f.readlines()
        else:
            file_path = root_path + '/test_dataset.txt'
            f = open(file_path, 'r')
            data_info = f.readlines()

        self.total_data = []
        for data in data_info:
            data = data.strip('\n')
            data = data.split()
            self.total_data.append(data)
            
        
    
    def __getitem__(self, index):
        data_info = self.total_data[index]
        data_index = data_info[0]
        
        audio_path = "/path/to/audio/feature/" + data_index + '.npy'
        video_path = "/path/to/video/feature/" + data_index + '.npy'

        audio_feature = torch.from_numpy(np.load(audio_path))
        video_feature = torch.from_numpy(np.load(video_path))
        
        tes = float(data_info[1])
        pcs = float(data_info[2])

        return audio_feature, video_feature, tes, pcs, data_index
        # return tes, pcs

    def __len__(self):
        return len(self.total_data)

    
def av_collate_fn(batch):
    audios = [item[0] for item in batch]
    videos = [item[1] for item in batch]
    inv_audios = [torch.flip(item[0], [0]) for item in batch]
    inv_videos = [torch.flip(item[1], [0]) for item in batch]
    tes = [item[2] for item in batch]
    pcs = [item[3] for item in batch]
    data_index = [item[4] for item in batch]
    
    audio_len = [item[0].shape[0] for item in batch]
    video_len = [item[1].shape[0] for item in batch]

    audios = pad_sequence(audios, batch_first=True)
    audios = torch.unsqueeze(audios, dim=2)
    videos = pad_sequence(videos, batch_first=True)
    # audios = torch.nn.functional.pad(audios, (0, 0, 0, 163-audios.shape[1]), 'constant', 0)

    inv_audios = pad_sequence(inv_audios, batch_first=True)
    inv_audios = torch.unsqueeze(inv_audios, dim=2)
    inv_videos = pad_sequence(inv_videos, batch_first=True)
    # videos = torch.nn.functional.pad(videos, (0, 0, 0, 1190-videos.shape[1]), 'constant', 0)

    tes = torch.FloatTensor(tes)
    pcs = torch.FloatTensor(pcs)

    scores = [tes, pcs]

    return audios, videos, inv_audios, inv_videos, audio_len, video_len, scores, data_index