#!/usr/bin/env bash
d=$(date +%Y%m%d%H%M)

###KITTI Monocular Experiments 
python3 run_mono_training.py --camera_height 1.70 --stereo_baseline 0.52 --pretrained_plane_dir 'results/plane-model-med-res-kitti' --data_dir '/media/m2-drive/datasets/KITTI-odometry-downsized' --estimator 'orbslam' --estimator_type 'mono' --train_seq '00' '02' '06' '07' '08' '11' '13' '14' '15' '16' '19' --val_seq '05' --test_seq '09' --date $d --lr 1e-4 --wd 0 --num_epochs 20 --lr_decay_epoch 12 --save_results


### Plane Segmentation Network Training
#python3 run_plane_training.py  --num_scales 3 --pretrained_dir 'results/202007100900-kitti-unscaled' --data_dir '/media/m2-drive/datasets/KITTI-odometry-downsized' --train_seq '00' '02' '06' '07' '08' '11' '13' '14' '15' '16' '19' --val_seq '05' --test_seq '09' --date $d --lr 1e-4 --num_epochs 10 --lr_decay_epoch 6 --save_results


