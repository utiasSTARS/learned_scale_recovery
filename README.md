# Learned Scale Recovery
Using a priori knowledge of a camera over a ground plane, we incorporate a scale recovery loss into the popular self-supervised depth and egomotion training procedure. In doing so, we enforce the observed scale factor to converge to unity during training, such that depth and egomotion predictions are accurate up to metric scale.

Accompanying code for 'Self-Supervised Scale Recovery for Monocular Depth and Egomotion Estimation'

<img src="https://github.com/utiasSTARS/learned_scale_recovery/blob/master/data/loss-diagram.png" width="600px"/>


## Dependencies:
* numpy
* scipy
* [pytorch](https://pytorch.org/) 
* [liegroups](https://github.com/utiasSTARS/liegroups)
* [pyslam](https://github.com/utiasSTARS/pyslam)
* [tensorboardX](https://github.com/lanpa/tensorboardX)

# Datasets

We trained and tested on the KITTI dataset. Download the raw dataset [here](http://www.cvlibs.net/datasets/kitti/raw_data.php). We provide a dataloader, but we first require that the data be preprocessed. To do so, run `create_kitti_odometry_data.py` within the `data` directory (be sure to specify the source and target directory). 

For Oxford Robotcar training, we downloaded sequences using the [dataset scraper](https://github.com/mttgdd/RobotCarDataset-Scraper). Once downloaded, the data can be preprocessed by running `synchronize_gt_with_imgs.py` in `data/oxford`, followed by `create_oxford_data.py` within the `data` directory (be sure to specify the source and target directory).

# Paper Reproduction

Our pretrained models are available online. To download them, run the following bash script from the source directory:

```
bash download_data.sh
```

This will populate the results directory with the appropriate models we trained for our experiments. Run `evaluate_model.py` within the `paper_plots_and_data` directory to evaluate a specific model on any desired sequence. This will save a `.pkl` file with the results. To compare multiple models, see `compare_methods.py` (for table II), `compare_and_plot_scale.py` (for table III), and `compare_retraining_methods.py` (for table IV). For plotting, run any of the `plot*.py` scripts. `generate_video_imgs.py` will store a sequence of depth and plane segmentation images within the `video` directory.


# Training

Two bash scripts are provided that will run the training experiments (for Depth + Egomotion training, and for Plane Segmentation training for KITTI and Robotcar: 

`run_mono_exps_kitti.sh`
`run_mono_exps_oxford.sh`

Prior to training, the data directory should be modified accordingly to point to the processed KITTI data. During training, to visualize the training procedure, open a tensorboard from the main directory:

`tensorboard --logdir runs` 
