import torch
import numpy as np


def create_2d_batch_from_3d_input(input_list):
    org_batch_size = len(input_list)
    batched_frames = np.zeros((0, input_list[0][0].shape[1], input_list[0][0].shape[2]))
    labels = np.zeros(0)
    for video, label in input_list:
        labels = np.concatenate((labels, label.repeat(video.shape[0])))
        batched_frames = np.concatenate((batched_frames, video))
    batched_frames = np.expand_dims(batched_frames, axis=1)
    return torch.FloatTensor(batched_frames), torch.LongTensor(labels)