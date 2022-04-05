#!/bin/bash
cd ..
python predict.py \
--source_dataset_folder /home/ola/Projects/Custom_Scripts/output/hundred_top_loss/UCGAI00369.01 \
--pre_trained_checkpoint /home/ola/Projects/View-Classification/saved_models/inception_2c_3c_4c_lax_100_epochs.pth \
--n_outputs 5 \
--result_file /media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/targets/test_view_class1.csv
python predict.py \
--source_dataset_folder /home/ola/Projects/Custom_Scripts/output/hundred_top_loss/UCGAI01803.01 \
--pre_trained_checkpoint /home/ola/Projects/View-Classification/saved_models/inception_2c_3c_4c_lax_100_epochs.pth \
--n_outputs 5 \
--result_file /media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/targets/test_view_class2.csv