#!/bin/bash

cd ..

python main.py \
--finetune \
--source_dataset_folder /media/ola/4fe0f1b8-1f60-4a27-a7df-72f752f56fa5/View_Train_Dataset_Grayscale \
--allowed_classes 0 1 2 3 4 6 12 \
--n_outputs 5 \
--epochs 200