import torch
import time
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
import torchvision.models as models
from torchvision.models.inception import BasicConv2d, InceptionAux
from arguments import get_predict_args
from data import pandas_dataset, collate
import numpy as np
from utils.utils import AverageMeter, dataframe_from_class_folder, finetuning_class_conversion
from sklearn.model_selection import train_test_split
import os
import deepspeed


def main():

    args = get_predict_args()
    dataframe = dataframe_from_class_folder(args.source_dataset_folder)
    if args.finetune:
        dataframe, class_conv_dict = finetuning_class_conversion(dataframe, args.allowed_classes, [[0, 1], [2, 3], [4, 6]])
        print(class_conv_dict)
    df_val = dataframe
    val_dataset = pandas_dataset.PandasDataset(df_val, target_length=1, target_height=484, target_width=636)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=args.n_workers,
                                                   pin_memory=args.no_pin_memory, drop_last=args.keep_last,
                                                   collate_fn=collate.create_2d_batch_from_3d_input)
    model = get_model(args)
    device = torch.cuda.current_device()
    model.to(device)
    model.eval()


    
    val_acc = AverageMeter()
    class_acc = {}
    for i in range(0, args.n_outputs):
        class_acc[i] = AverageMeter()
    
    for k, (v_inputs, v_labels) in enumerate(val_dataloader):
        v_inputs = v_inputs.to(device, non_blocking=True)
        v_labels = v_labels.to(device, non_blocking=True)
        with torch.no_grad():
            v_outputs = model(v_inputs)

        v_metric_labels = v_labels.cpu().detach().numpy()
        _, v_preds = torch.max(v_outputs.data.cpu(), 1)

        v_acc = accuracy_score(v_metric_labels, v_preds)
        val_acc.update(v_acc)
        class_acc[int(v_metric_labels)].update(v_acc)
        
        # End of validation epoch prints and updates
        #print('Finished Test: {}\t '
        #    'Pred: {} \t'
        #    'Label: {} \t'
        #    'Accuracy score: {metric:.3f} \t'
        #    'F1 Score: {f1:.3f} \t'
        #    .format(k, v_preds, v_metric_labels, metric=v_acc, f1=v_f1))



    # End of validation epoch prints and updates
    print('Finished Test: {}\t '
          'Accuracy score: {metric.avg:.3f} \t'
          .format(k, metric=val_acc))
    for i in range(0, args.n_outputs):
        print('{}: {metric.avg:.3f} \t'
            .format(i, metric=class_acc[i]))


def get_model(args):
    if args.model_name == 'inception':
        if args.pre_trained_checkpoint is not None:
            model = models.inception_v3(pretrained=False, transform_input=False)
            model.fc = nn.Linear(2048, args.n_outputs)
            model.AuxLogits = InceptionAux(768, args.n_outputs)
            model.aux_logits = False
            new_conv = BasicConv2d(1, 32, kernel_size=3, stride=2)
            model.Conv2d_1a_3x3 = new_conv
            sd = torch.load(args.pre_trained_checkpoint)['model']
            model.load_state_dict(sd)
        else:
            print('Missing model.')
    return model

if __name__ == '__main__':
    main()
