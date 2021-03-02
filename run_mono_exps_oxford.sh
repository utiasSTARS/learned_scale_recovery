#!/usr/bin/env bash
d=$(date +%Y%m%d%H%M)

### Oxford RobotCar Experiments
#d=$(date +%Y%m%d%H%M)
python3 run_mono_training.py --flow_type 'classical' --max_depth 5 --stereo_baseline 0.24 --camera_height 1.52 --pretrained_plane_dir 'results/plane-model-oxford-202102092331' --data_dir '/media/datasets/oxford-robotcar-downsized' --estimator 'stereo' --estimator_type 'mono' --train_seq 'all'  --val_seq '2014-11-18-13-20-12_0' --test_seq '2014-11-18-13-20-12_1' --date $d --lr 1e-4 --wd 0 --num_epochs 20 --lr_decay_epoch 12 --save_results


### Plane Estimator Training
#python3 run_plane_training.py --pretrained_dir 'results/final_models/vo-oxford-unscaled-202102092331' --data_dir '/media/datasets/oxford-robotcar-downsized' --estimator 'stereo' --train_seq 'all'  --val_seq '2014-11-18-13-20-12_0' --test_seq '2014-11-18-13-20-12_1' --date $d --lr 1e-4 --num_epochs 10 --lr_decay_epoch 4 --save_results


