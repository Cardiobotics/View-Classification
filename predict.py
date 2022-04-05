import torch
from arguments import get_predict_args
from utils.utils import dataframe_from_folder
from data import dicom_dataset, collate
import torchvision.models as models
import torch.nn as nn
from torchvision.models.inception import BasicConv2d, InceptionAux
import pandas as pd
import time


def main():
    args = get_predict_args()

    dataframe = dataframe_from_folder(args.source_dataset_folder)
    dataset = dicom_dataset.DicomDataset(dataframe, target_height=484, target_width=636)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.n_workers, shuffle=False,
                                             pin_memory=args.no_pin_memory, drop_last=args.keep_last,
                                             collate_fn=collate.create_2d_batch_from_3d_input_predict)
    model = get_model(args)
    device = torch.cuda.current_device()
    model.to(device)
    #model = nn.DataParallel(model)
    model.eval()

    data_dict = {'us_id': [], 'instance_id': [], 'prediction': [], 'file': []}

    time_start = time.time()

    for i, (inputs, (iid, uid, file)) in enumerate(dataloader):
        if inputs is None:
            continue
        with torch.no_grad():
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
        pred = majority_vote(outputs).cpu().numpy()
        data_dict['us_id'].append(uid)
        data_dict['instance_id'].append(iid)
        data_dict['prediction'].append(pred)
        data_dict['file'].append(file)
        if i % 1000 == 0:
            curr_time = time.time() - time_start
            datapoints_per_second = (i+1) / curr_time
            time_left = ((len(dataloader) - (i+1))/datapoints_per_second)/60
            print('Processed files: {} \t Time left (minutes): {}'.format((i+1), time_left))

    output_df = pd.DataFrame(data_dict)
    output_df.to_csv(args.result_file, sep=';', index=False)
    print(output_df)

def majority_vote(outputs):
    # Get indexes for max output for each frame
    _, i = torch.max(outputs, dim=1)
    # Get index for the max of binned indexes
    _, i2 = torch.max(torch.bincount(i), dim=0)
    return i2

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
            model = models.inception_v3(pretrained=True, transform_input=False)
            model.fc = nn.Linear(2048, args.n_outputs)
            model.AuxLogits = InceptionAux(768, args.n_outputs)
            model.aux_logits = False
            new_conv = BasicConv2d(1, 32, kernel_size=3, stride=2)
            first_layer_sd = model.Conv2d_1a_3x3.state_dict()
            first_layer_sd['conv.weight'] = first_layer_sd['conv.weight'].mean(dim=1, keepdim=True)
            new_conv.load_state_dict(first_layer_sd)
            model.Conv2d_1a_3x3 = new_conv
    elif args.model_name == 'resnet':
        if args.pre_trained_checkpoint is not None:
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(in_features=2048, out_features=args.n_outputs, bias=True)
            new_conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.conv1 = new_conv
            sd = torch.load(args.pre_trained_checkpoint)['model']
            model.load_state_dict(sd)
        else:
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(in_features=2048, out_features=args.n_outputs, bias=True)
            new_conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            first_layer_sd = model.conv1.state_dict()
            first_layer_sd['weight'] = first_layer_sd['weight'].mean(dim=1, keepdim=True)
            new_conv.load_state_dict(first_layer_sd)
            model.conv1 = new_conv
    elif args.model_name == 'resnext':
        if args.pre_trained_checkpoint is not None:
            model = models.resnext50_32x4d(pretrained=False)
            model.fc = nn.Linear(in_features=2048, out_features=args.n_outputs, bias=True)
            new_conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.conv1 = new_conv
            sd = torch.load(args.pre_trained_checkpoint)['model']
            model.load_state_dict(sd)
        else:
            model = models.resnext50_32x4d(pretrained=True)
            model.fc = nn.Linear(in_features=2048, out_features=args.n_outputs, bias=True)
            new_conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            first_layer_sd = model.conv1.state_dict()
            first_layer_sd['weight'] = first_layer_sd['weight'].mean(dim=1, keepdim=True)
            new_conv.load_state_dict(first_layer_sd)
            model.conv1 = new_conv
    return model

if __name__ == '__main__':
    main()
