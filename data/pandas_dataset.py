import torch
import numpy as np
from data import transforms


class PandasDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, target_length, target_height, target_width):
        self.dataset = dataframe.reset_index()
        self.transforms = transforms.CustomTransforms(target_length, target_height, target_width)
        self.target_length = target_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset.iloc[index].file
        label = self.dataset.iloc[index].label
        img = np.load(file, allow_pickle=True)
        img = self.select_x_random_frames(img)
        img = self.transforms.apply_transforms(img)
        return img, label

    def select_x_random_frames(self, img):
        if len(img) >= self.target_length:
            indexes = np.random.choice(range(len(img)), size=self.target_length, replace=False)
        else:
            indexes = np.random.choice(range(len(img)), size=self.target_length)
        frames = np.zeros((0, img.shape[1], img.shape[2]))
        for i in indexes:
            frames = np.concatenate((frames, np.expand_dims(img[i], axis=0)))
        return frames


