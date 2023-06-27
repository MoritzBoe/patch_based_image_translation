# Improving generative adversarial networks for patch-based unpaired image-to-image translation
Implementation for *Stitching Aware Training and Inference (SATI)* and the benchmark methods used in our publication.

## Prerequisites
* [Anaconda Distribution](https://www.anaconda.com/distribution/#download-section).
* CUDA capable GPU.

## Installation
Clone the repository
```
git clone https://git.scc.kit.edu/sl6831/ganpatchinference.git
cd ganpatchinference
```
Create a new virtual environment:
```
conda env create -f requirements.yml
```
Activate the virtual environment:
```
conda activate StitchingAwareTrainingAndInference
```

## Data
For new real-world data, create a LightningDataModule in data_modules.py and add it to the get_data_module function. In this README, we describe usage on the synthetic *tiling strategy benchmark dataset*.

Create two separate benchmark datasets to obtain unpaired training data by running benchmark_dataset/create_benchmark_dataset.py with:
```
python benchmark_dataset/create_benchmark_dataset.py
```
Two new folders (tiling_strategy_benchmark_1 and tiling strategy_benchmark_2) appear in the datasets folder.

## Training
The scrips are created to work with Weigths & Biases. You need to create an account at https://wandb.ai/. For personal usage, this is free. To setup W&B initialize it with:
```
wandb login YOURAPIKEY
```
You can find your API-Key under https://wandb.ai/settings.

The final setting used for our experiments is trained with:
```
python lit_cyclegan.py --entity YOURUSERNAME --project SATI_benchmark --num_channels_A 3 --num_channels_B 3 --batch_size 5 --crop_size 256 --criterion_disc MSE --criterion_idt_A MSE --criterion_idt_B MSE --criterion_cycle_ABA MSE --criterion_cycle_BAB MSE --criterion_inpainting DweightedMSE -lir --inpainting_start_epoch 0 --minmax_scale_A -1 0 --minmax_scale_B 0 1 --n_epochs_lr_fix 300 --n_epochs_lr_decay 300 --ngf 96
```

Further hyperparameters can be displayed by running:
```
python lit_cyclegan.py -h
```
The optional hyperparameters are the hyperparameters used to configure the CycleGAN. The pl.Trainer arguments are the standard pytorch lightning hyperparameters. The number of pixels used to create the overlapping images during training can e.g. be changed with `--inpainting_pixel 32`.

At the moment, only the final model after training is saved to W&B. This behavior can be changed in the `ModelCheckpoint` callback in `lit_cyclegan.py`. Please denote, that a single checkpoint is ~680MB. Together with the images created for logging (3.2MB per logged epoch), logging data grows fast. To reduce logged image data, `--num_val_samples` can be set to one and `--val_frequency` can be increased from one (every epoch) e.g. to 5 (every 5 epochs).

## Continue training
A checkpoint can be used to continue training. Please denote, that the initial checkpoint hyperparameters are overwritten by the argparse arguments. Therefore, when the initial number of generator feature maps was set to 96, this has to be parsed again. The checkpoint path in W&B can be accessed by clicking on the Artifacts symbol of the corresponding run. The full name of the artifact looks like `[username]/[project]/[model_artifact]`. Using the above trained model to continue training with different loss functions can look like this:
```
python lit_cyclegan.py --entity YOURUSERNAME --project SATI_benchmark --mode start_from_local_ckpt --ckpt_artifact [username]/[project]/[model_artifact] --num_channels_A 3 --num_channels_B 3 --batch_size 5 --crop_size 256 --criterion_disc MSE --criterion_idt_A MSE --criterion_idt_B MSE --criterion_cycle_ABA L1 --criterion_cycle_BAB L1 --criterion_inpainting InpaintingL1 -lir --inpainting_start_epoch 0 --minmax_scale_A -1 0 --minmax_scale_B 0 1 --n_epochs_lr_fix 800 --n_epochs_lr_decay 200 --ngf 96
```
Because the epoch count is not reset, `--n_epochs_lr_fix` and `--n_epochs_lr_decay` have to be adapted. In our case the first training took 600 epochs. Therefore, with `--n_epochs_lr_fix 800` and  `--n_epochs_lr_decay 200`, the training is continued for 200 epochs with the initial starting learning rate. Afterwards it decays to zero throughout 200 epochs.


## Inference
Inference.py is used to create new synthetic images from the label images. A checkpoint artifact has to be provided (see continue training). Results for SATI can be created with:
```
python inference.py --ca [username]/[project]/[model_artifact] -cs 512 -o 128 -kp 50 -ni 50
```
The results will be written to a new path `results/[project]/[model_artifact]`.
Again, the hyperparameters can be displayed with:
```
python inference.py -h
```
To create the results without tiling (GPU needs enough VRAM) and with simple tiling, run:
```
python inference.py --ca [username]/[project]/[model_artifact] -cs 2048 -o 0 -kp 0 -ni 50
python inference.py --ca [username]/[project]/[model_artifact] -cs 512 -o 0 -kp 0 -ni 50
```

For the weighted tiling results used in Stain-Transforming Cycle-Consistent Generative Adversarial Networks for Improved Segmentation of Renal Histopathology (Bel et. al. 2019) run:
```
python inference_weighted.py --ca [username]/[project]/[model_artifact] -ps 512 -cs 256 -sh 128 -ni 50
```
Please denote, that for our work we trained models without SATI to create the simple tiling and weighted tiling results. Nonetheless, the SATI models can be used for inference with simple tiling and weighted tiling.

## Evaluation
Results can be evaluated with eval/eval_color.py. For the results with SATI, this can e.g. be done with:
```
python eval/eval_color.py --img_real datasets/tiling_strategy_benchmark_1/labels --img_fake results/[project]/[model_artifact]/crop_512/overlap_128/keep_predicted_50
```
The `--img_real` argument has to be set to the folder of the images used to create the synthetic images. The `--img_fake` argument has to be set to the folder where the synthetic images are saved after inference. Therefore, for the images using simple tiling and weighted tiling, the results can be calculated with:
```
python eval/eval_color.py --img_real datasets/tiling_strategy_benchmark_1/labels --img_fake results/[project]/[model_artifact]/crop_2048/overlap_0/keep_predicted_0

python eval/eval_color.py --img_real datasets/tiling_strategy_benchmark_1/labels --img_fake results/[project]/[model_artifact]/weighted/patch_size_512/crop_256/shift_128/
```
Again, to reproduce our results, models without SATI need to be used to create the results for weighted tiling and simple tiling.

## Docker
Build the docker container with:
```
sudo sh docker_build.sh
```
Before you run the container, the W&B API-Key and your entity needs to be written to the `run.sh` file. Inside the `run.sh` file, you can run all scripts used for our experiments (see training section). Run the container by running `docker_run.sh` with:
```
sudo sh docker_run.sh
```

