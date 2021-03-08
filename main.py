import torch
import time
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
import torchvision.models as models
from torchvision.models.inception import BasicConv2d, InceptionAux
from arguments import get_args
from data import pandas_dataset, collate
import numpy as np
from utils.utils import AverageMeter, dataframe_from_folder
from sklearn.model_selection import train_test_split
import os
import deepspeed


def main():

    args = get_args()
    dataframe = dataframe_from_folder(args.source_dataset_folder)
    df_train, df_val = train_test_split(dataframe, test_size=0.15, stratify=dataframe['label'])
    train_dataset = pandas_dataset.PandasDataset(df_train, target_length=1, target_height=484, target_width=636)
    val_dataset = pandas_dataset.PandasDataset(df_val, target_length=1, target_height=484, target_width=636)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=args.n_workers, shuffle=True,
                                             pin_memory=args.pin_memory, drop_last=args.drop_last,
                                             collate_fn=collate.create_2d_batch_from_3d_input)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, num_workers=args.n_workers,
                                                   shuffle=True,
                                                   pin_memory=args.pin_memory, drop_last=args.drop_last,
                                                   collate_fn=collate.create_2d_batch_from_3d_input)
    if args.model_name == 'inception':
        model = models.inception_v3(pretrained=True, transform_input=False)
        model.fc = nn.Linear(2048, args.n_outputs)
        model.AuxLogits = InceptionAux(768, args.n_outputs)
        new_conv = BasicConv2d(1, 32, kernel_size=3, stride=2)
        first_layer_sd = model.Conv2d_1a_3x3.state_dict()
        first_layer_sd['conv.weight'] = first_layer_sd['conv.weight'].mean(dim=1, keepdim=True)
        new_conv.load_state_dict(first_layer_sd)
        model.Conv2d_1a_3x3 = new_conv
    elif args.model_name == 'resnet':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(in_features=2048, out_features=args.n_outputs, bias=True)
        new_conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        first_layer_sd = model.conv1.state_dict()
        first_layer_sd['weight'] = first_layer_sd['weight'].mean(dim=1, keepdim=True)
        new_conv.load_state_dict(first_layer_sd)
        model.conv1 = new_conv
    elif args.model_name == 'resnext':
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(in_features=2048, out_features=args.n_outputs, bias=True)
        new_conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        first_layer_sd = model.conv1.state_dict()
        first_layer_sd['weight'] = first_layer_sd['weight'].mean(dim=1, keepdim=True)
        new_conv.load_state_dict(first_layer_sd)
        model.conv1 = new_conv
    device = 'cuda'
    model.to(device)

    #model = nn.DataParallel(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, train_dataloader, __ = deepspeed.initialize(args=args, model=model, model_parameters=parameters, training_data=train_dataset, collate_fn=collate.create_2d_batch_from_3d_input)

    torch.backends.cudnn.benchmark = args.cuddn_auto_tuner

    class_counts = np.array(args.n_outputs * [1])
    for i in dataframe['label'].value_counts().index:
        class_counts[i] = dataframe['label'].value_counts().loc[i]
    # Calculate the inverse normalized ratio for each class
    weights = class_counts / class_counts.sum()
    weights = 1 / weights
    weights = weights / weights.sum()
    weights = torch.FloatTensor(weights).cuda()

    criterion = nn.CrossEntropyLoss(weight=weights)

    #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=2e-5)

    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, steps_per_epoch=len(dataloader), epochs=args.epochs)

    for i in range(args.epochs):
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        train_f1 = AverageMeter()
        for j, data in enumerate(train_dataloader):
            batch_start = time.time()
            #inputs = inputs.to(device, non_blocking=True)
            #labels = labels.to(device, non_blocking=True)
            inputs, labels = data[0].to(model_engine.local_rank), data[1].to(model_engine.local_rank)
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            model_engine.backward(loss)
            model_engine.step()

            metric_labels = labels.cpu().detach().numpy()
            _, preds = torch.max(outputs.data.cpu(), 1)
            acc = accuracy_score(metric_labels, preds)
            t_f1 = f1_score(metric_labels, preds, average='macro')

            train_loss.update(loss)
            train_acc.update(acc)
            train_f1.update(t_f1)
            batch_time = time.time() - batch_start
            if j % 100 == 0:
                print('Training Batch: [{}/{}] in epoch: {} \t '
                      'Training Loss: {loss.val:.4f} ({loss.avg:.4f}) \t '
                      'Training Accuracy Score: {metric.val:.3f} ({metric.avg:.3f}) \t'
                      'Training F1 Score: {f1.val:.3f} ({f1.avg:.3f}) \t'
                      'Batch Time: {bt:.5f} seconds \t'
                      .format(j+1, len(train_dataloader), i + 1, loss=train_loss, metric=train_acc, f1=train_f1, bt=batch_time))
            #scheduler.step()

        # End of training epoch prints and updates
        print('Finished Training Epoch: {} \t '
              'Training Loss: {loss.avg:.4f} \t '
              'Training Accuracy score: {metric.avg:.3f} \t'
              .format(i + 1, loss=train_loss, metric=train_acc))

        if i % 10 == 0:
            model.eval()
            val_loss = AverageMeter()
            val_acc = AverageMeter()
            val_f1 = AverageMeter()
            for k, (v_inputs, v_labels) in enumerate(val_dataloader):
                v_inputs = v_inputs.to(device, non_blocking=True)
                v_labels = v_labels.to(device, non_blocking=True)
                with torch.no_grad():
                    v_outputs = model(v_inputs)
                    v_loss = criterion(v_outputs, v_labels)

                v_metric_labels = v_labels.cpu().detach().numpy()
                _, v_preds = torch.max(v_outputs.data.cpu(), 1)
                v_acc = accuracy_score(v_metric_labels, v_preds)
                v_f1 = f1_score(v_metric_labels, v_preds, average='macro')

                val_loss.update(v_loss)
                val_acc.update(v_acc)
                val_f1.update(v_f1)

            # End of validation epoch prints and updates
            print('Finished Validation Epoch: {} \t '
                  'Validation Loss: {loss.avg:.4f} \t '
                  'Validation Accuracy score: {metric.avg:.3f} \t'
                  'Validation F1 Score: {f1.val:.3f} ({f1.avg:.3f}) \t'
                  .format(i + 1, loss=val_loss, metric=val_acc, f1=val_f1))

    checkpoint_name = os.path.join(args.checkpoint_save_path, 'all_classes_100_epochs.pth')
    save_checkpoint(checkpoint_name, model, optimizer)


def save_checkpoint(save_file_path, model, optimizer):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(save_states, save_file_path)


if __name__ == '__main__':
    main()
