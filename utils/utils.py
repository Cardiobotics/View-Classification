import os
import pandas as pd


def load_filenames(path):
    files = []
    for dirName, _, fileList in os.walk(path, followlinks=True):
        for filename in fileList:
            files.append(os.path.join(dirName, filename))
    return files


def load_sub_dir_names(path):
    return next(os.walk(path, followlinks=True))[1]


def get_filenames_and_label(path):
    files = []
    labels = []
    for dirName, _, fileList in os.walk(path, followlinks=True):
        _, cls = os.path.split(dirName)
        for filename in fileList:
            files.append(os.path.join(dirName, filename))
            labels.append(int(cls))
    return files, labels

def get_filenames(path):
    files = []
    for dirName, _, fileList in os.walk(path, followlinks=True):
        for filename in fileList:
            files.append(os.path.join(dirName, filename))
    return files


def dataframe_from_class_folder(path):
    files, labels = get_filenames_and_label(path)
    dataset = pd.DataFrame({'file': files, 'label': labels})
    return dataset


def dataframe_from_folder(path):
    files = get_filenames(path)
    dataset = pd.DataFrame({'file': files})
    return dataset


def finetuning_class_conversion(dataframe, finetuning_classes, merged_classes):
    original_classes = dataframe['label'].unique()
    original_classes.sort()
    ft_class = 0
    class_conv_dict = {}
    misc_class = get_misc_class(finetuning_classes, merged_classes)
    for oc in original_classes:
        if oc in finetuning_classes:
            merged, merge_class = check_merge(oc, merged_classes)
            if merged:
                nc = class_conv_dict[merge_class]
                class_conv_dict[oc] = nc
            else:
                class_conv_dict[oc] = ft_class
                ft_class += 1
        else:
            class_conv_dict[oc] = misc_class
    dataframe_ft = dataframe.rename(columns={'label': 'label_old'})
    dataframe_ft['label'] = dataframe_ft['label_old'].apply(lambda x: class_conv_dict[x])
    return dataframe_ft, class_conv_dict


def check_merge(cls, merged_classes):
    for mc in merged_classes:
        mc.sort()
        if cls in mc and cls != mc[0]:
            return True, mc[0]
    return False, cls


def get_misc_class(ft_classes, merged_classes):
    m_reduction = 0
    for mc in merged_classes:
        m_reduction += (len(mc) - 1)
    return len(ft_classes) - m_reduction

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
