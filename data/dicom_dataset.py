import torch
import numpy as np
from data import transforms
import pydicom


class DicomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, target_height, target_width):
        super(DicomDataset, self).__init__()
        self.dataset = dataframe.reset_index()
        self.transforms = transforms.CustomTransforms(None, target_height, target_width)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset.iloc[index].file
        img, iid, uid = self.load_dicom(file)
        if img is not None:
            img = self.transforms.apply_transforms(img)
        return img, (iid, uid, file)


    def load_dicom(self, file):
        try:
            data = pydicom.read_file(file)
            if hasattr(data, 'pixel_array'):
                img = data.pixel_array
            else:
                raise AttributeError('No pixel data for file: {}'.format(file))
            if hasattr(data, 'InstanceNumber'):
                iid = data.InstanceNumber
            else:
                raise AttributeError('No instance number for file: {}'.format(file))
            if hasattr(data, 'PatientID'):
                uid = data.PatientID
            else:
                raise AttributeError('No PatientID for file: {}'.format(file))
        except AttributeError as ae:
            img, iid, uid = None, None, None
            print("Failed to read dicom attribute with error: {}".format(ae))
        except:
            print("Unknown error")
        return img, iid, uid


