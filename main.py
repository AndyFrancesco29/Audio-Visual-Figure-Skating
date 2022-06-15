from typing_extensions import final
import torch
import torch.utils.data as data
import os
import numpy as np
from model import scoring_head, head
from dataset.dataset_fs800 import FeatureDataset, av_collate_fn
from scipy.stats import spearmanr 
import math
# from torch.optim import lr_sheduler
# import time
# import warnings

dev = 0

def validation(dataloader, model, criterion, score_index):
    model.eval()
    val_loss = 0
    val_truth = []
    val_pred = []


    for audio_feature, video_feature, inv_audio_feature, inv_video_feature, audio_len, video_len, score, data_index in dataloader:
        batch_size, _, _, _ = audio_feature.shape
        audio_feature = audio_feature.cuda(device=dev)
        video_feature = video_feature.cuda(device=dev)
        inv_audio_feature = inv_audio_feature.cuda(device=dev)
        inv_video_feature = inv_video_feature.cuda(device=dev)
        target = score[score_index].cuda(device=dev)

        with torch.no_grad():
            output = model(audio_feature, video_feature, inv_audio_feature, inv_video_feature, audio_len, video_len)
        val_pred.append(output.detach().data.cpu().numpy())
        val_truth.append(target.cpu().numpy())

        loss = criterion(output, target)

        val_loss += loss.item() * batch_size

    val_truth = np.concatenate(val_truth)
    val_pred = np.concatenate(val_pred)
    spear = spearmanr(val_truth, val_pred)
    print(len(dataloader.dataset))
    val_loss = val_loss / len(dataloader.dataset)
    return val_loss, spear.correlation


# build dataset
train_dataset = FeatureDataset(root_path = '/path/to/txt/', is_train = True)
val_dataset = FeatureDataset(root_path = '/path/to/txt/', is_train = False)

train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=16, num_workers=8, shuffle=True, collate_fn=av_collate_fn)
val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=16, num_workers=8, collate_fn=av_collate_fn)

# model
model = scoring_head(depth=2, input_dim=768, dim=512, input_len=16, num_scores=1, bidirection=True).cuda(device=dev)

epochs = 500
warm_up_epochs = 10
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-6)

# criterion
criterion = torch.nn.MSELoss()

# other parameter
score_index = 4

min_val_loss = 10000
max_spear_cor = 0

model.train()

for epoch_idx in range(epochs):
    print("="*25)
    print("epoch ", epoch_idx)
    
    for audio_feature, video_feature, inv_audio_feature, inv_video_feature, audio_len, video_len, score, data_index in train_dataloader:
        audio_feature = audio_feature.cuda(device=dev)
        video_feature = video_feature.cuda(device=dev)
        inv_audio_feature = inv_audio_feature.cuda(device=dev)
        inv_video_feature = inv_video_feature.cuda(device=dev)
        
        target = score[score_index].cuda(device=dev)

        train_loss = 0
        optimizer.zero_grad()

        output = model(audio_feature, video_feature, inv_audio_feature, inv_video_feature, audio_len, video_len)

        loss = criterion(output, target)
        train_loss = loss.item()

        loss.backward()
        optimizer.step()
    
    # validation
    val_loss, spear = validation(val_dataloader, model, criterion, score_index)
    print("val_loss: ", val_loss, " | spear corr: ", spear)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        
        torch.save(model.state_dict(), "./fs800_result/checkpoint_pe.pth")
    if spear > max_spear_cor:
        max_spear_cor = spear
    print("min validation loss: ", min_val_loss, " | max spear corr: ", max_spear_cor)
    print("checkpoint_pe")
    
    
    print(optimizer.param_groups[0]['lr'])
    # scheduler.step()
    
    
