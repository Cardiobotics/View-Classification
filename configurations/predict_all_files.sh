#!/bin/bash
cd ..
python predict.py \
--source_dataset_folder /media/ola/7540de01-b8d5-4df4-883c-1a8429f18b56/uttag2_grp1/GE/rorligsv \
--pre_trained_checkpoint /home/ola/Projects/View-Classification/saved_models/inception_2c_3c_4c_lax_merged_100_epochs.pth \
--n_outputs 5 \
--result_file /media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/targets/view_class_merged_grp1.csv
python predict.py \
--source_dataset_folder /media/ola/7540de01-b8d5-4df4-883c-1a8429f18b56/uttag2_grp2/GE/rorligsv \
--pre_trained_checkpoint /home/ola/Projects/View-Classification/saved_models/inception_2c_3c_4c_lax_merged_100_epochs.pth \
--n_outputs 5 \
--result_file /media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/targets/view_class_merged_grp2.csv
python predict.py \
--source_dataset_folder /media/ola/7540de01-b8d5-4df4-883c-1a8429f18b56/uttag2_grp5/GE/rorligsv \
--pre_trained_checkpoint /home/ola/Projects/View-Classification/saved_models/inception_2c_3c_4c_lax_merged_100_epochs.pth \
--n_outputs 5 \
--result_file /media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/targets/view_class_merged_grp5.csv
python predict.py \
--source_dataset_folder /media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/uttag2_grp3/GE/rorligsv \
--pre_trained_checkpoint /home/ola/Projects/View-Classification/saved_models/inception_2c_3c_4c_lax_merged_100_epochs.pth \
--n_outputs 5 \
--result_file /media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/targets/view_class_merged_grp3.csv
python predict.py \
--source_dataset_folder /media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/uttag2_grp4/GE/rorligsv \
--pre_trained_checkpoint /home/ola/Projects/View-Classification/saved_models/inception_2c_3c_4c_lax_merged_100_epochs.pth \
--n_outputs 5 \
--result_file /media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/targets/view_class_merged_grp4.csv