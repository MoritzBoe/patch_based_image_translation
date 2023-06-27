# log into W&B and run python script

# W&B API-Key. Get the key from wandb.ai/settings
API_KEY="YOURAPIKEY"

# Your W&B Entity. Usually your username.
ENTITY="YOURENTITY"

# create benchmark dataset
python -u benchmark_dataset/create_benchmark_dataset.py

# log into W&B and train CycleGAN model
wandb login $API_KEY
python -u lit_cyclegan.py --entity $ENTITY --project SATI_benchmark --num_channels_A 3 --num_channels_B 3 \
    --batch_size 1 --crop_size 256 --criterion_disc MSE --criterion_idt_A MSE --criterion_idt_B MSE --criterion_cycle_ABA MSE \
    --criterion_cycle_BAB MSE --criterion_inpainting DweightedMSE -lir --inpainting_start_epoch 0 --minmax_scale_A -1 0 --minmax_scale_B 0 1 \
    --n_epochs_lr_fix 300 --n_epochs_lr_decay 300 --ngf 96

