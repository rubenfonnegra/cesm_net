source /home/mirplab/Documents/kevin/venvs/cesm_venv/bin/activate
cd /home/mirplab/Documents/kevin/01-cesm_net/


# Experiment Unet CC Patches Breast
python3 run_models.py   --gpus 0 \
                        --dataset_name cesm \
                        --result_dir Results/02-patches/02-Unet/ \
                        --projection CC \
                        --exp_name Unet_Patches_CC\
                        --data_dir Data/data_without_background/ \
                        --image_size 256 \
                        --channels 1 \
                        --batch_size 20 \
                        --n_epochs 201 \
                        --workers 12 \
                        --sample_interval 50 \
                        --checkpoint_interval 50 \
                        --model Unet \
                        --sample_size 50 \
                        --use_wandb True \

python3 run_models.py   --gpus 0 \
                        --generate \
                        --epoch 200 \
                        --model Unet \
                        --exp_name Unet_Patches_CC \
                        --result_dir Results/02-patches/02-Unet/ \
                        --sample_size 50 \
                        --dataset_name cesm \
                        --projection CC \
                        --data_dir Data/data_without_background/  \
                        --image_size 256 \
                        --channels 1 \

python3 val_metrics.py  --name_exp Unet_Patches_CC \
                        --name_exp_fig "Unet, Patches, CC" \
                        --path_results Results/02-patches/02-Unet/ \
                        --path_data Data/data_without_background/  \
                        --model Unet \
                        --epoch 200

# Experiment Unet CC Patches 20%
python3 run_models.py   --gpus 0 \
                        --dataset_name cesm \
                        --result_dir Results/02-patches/02-Unet/ \
                        --projection CC \
                        --exp_name Unet_Patches_20_CC\
                        --data_dir Data/borde_bg_20/ \
                        --image_size 256 \
                        --channels 1 \
                        --batch_size 20 \
                        --n_epochs 201 \
                        --workers 12 \
                        --sample_interval 50 \
                        --checkpoint_interval 50 \
                        --model Unet \
                        --sample_size 50 \
                        --use_wandb True \

python3 run_models.py   --gpus 0 \
                        --generate \
                        --epoch 200 \
                        --model Unet \
                        --exp_name Unet_Patches_20_CC \
                        --result_dir Results/02-patches/02-Unet/ \
                        --sample_size 50 \
                        --dataset_name cesm \
                        --projection CC \
                        --data_dir Data/borde_bg_20/  \
                        --image_size 256 \
                        --channels 1 \

python3 val_metrics.py  --name_exp Unet_Patches_CC \
                        --name_exp_fig "Unet, Patches 20%, CC" \
                        --path_results Results/02-patches/02-Unet/ \
                        --path_data Data/borde_bg_20/  \
                        --model Unet \
                        --epoch 200

# Experiment Unet MLO Patches 20%
python3 run_models.py   --gpus 0 \
                        --dataset_name cesm \
                        --result_dir Results/02-patches/02-Unet/ \
                        --projection MLO \
                        --exp_name Unet_Patches_20_MLO\
                        --data_dir Data/borde_bg_20/ \
                        --image_size 256 \
                        --channels 1 \
                        --batch_size 20 \
                        --n_epochs 201 \
                        --workers 12 \
                        --sample_interval 50 \
                        --checkpoint_interval 50 \
                        --model Unet \
                        --sample_size 50 \
                        --use_wandb True \

python3 run_models.py   --gpus 0 \
                        --generate \
                        --epoch 200 \
                        --model Unet \
                        --exp_name Unet_Patches_20_MLO \
                        --result_dir Results/02-patches/02-Unet/ \
                        --sample_size 50 \
                        --dataset_name cesm \
                        --projection MLO \
                        --data_dir Data/borde_bg_20/  \
                        --image_size 256 \
                        --channels 1 \

python3 val_metrics.py  --name_exp Unet_Patches_MLO \
                        --name_exp_fig "Unet, Patches 20%, MLO" \
                        --path_results Results/02-patches/02-Unet/ \
                        --path_data Data/borde_bg_20/  \
                        --model Unet \
                        --epoch 200 \
                        --projection MLO \
