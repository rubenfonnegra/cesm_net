source /media/labmirp/Datos/venvs/cesm_env/bin/activate
cd /media/labmirp/Datos/workspaces/cesm_net/


# Experiment Unet CC
python3 run_models.py   --gpus 0 \
                        --dataset_name cesm \
                        --result_dir Results/07-Unet-Based/ \
                        --projection CC \
                        --exp_name Unet-Deep-Patches-CC\
                        --data_dir Data/cesm_patches/data_without_background/ \
                        --image_size 256 \
                        --channels 1 \
                        --batch_size 30 \
                        --n_epochs 401 \
                        --workers 12 \
                        --sample_interval 50 \
                        --checkpoint_interval 50 \
                        --model UNet_Deep \
                        --sample_size 20 \
                        --use_wandb True \

python3 run_models.py   --gpus 0 \
                        --generate \
                        --epoch 400 \
                        --model UNet_Deep \
                        --exp_name Unet-Deep-Patches-CC \
                        --result_dir Results/07-Unet-Based/ \
                        --sample_size 20 \
                        --dataset_name cesm \
                        --projection CC \
                        --data_dir Data/cesm_patches/data_without_background/  \
                        --image_size 256 \
                        --channels 1 \

python3 val_metrics.py  --name_exp Unet-Deep-Patches-CC \
                        --name_exp_fig "Deep Unet, Patches, CC" \
                        --path_results Results/07-Unet-Based/ \
                        --path_data Data/cesm_patches/data_without_background/  \
                        --model UNet_Deep \
                        --epoch 400 \


# Experiment Unet CC Patches 20%
python3 run_models.py   --gpus 0 \
                        --dataset_name cesm \
                        --result_dir Results/07-Unet-Based/ \
                        --projection CC \
                        --exp_name Unet-Deep-Patches-20-CC\
                        --data_dir Data/cesm_patches/borde_bg_20/ \
                        --image_size 256 \
                        --channels 1 \
                        --batch_size 30 \
                        --n_epochs 401 \
                        --workers 12 \
                        --sample_interval 50 \
                        --checkpoint_interval 50 \
                        --model UNet_Deep \
                        --sample_size 20 \
                        --use_wandb True \

python3 run_models.py   --gpus 0 \
                        --generate \
                        --epoch 400 \
                        --model UNet_Deep \
                        --exp_name Unet-Deep-Patches-20-CC \
                        --result_dir Results/07-Unet-Based/ \
                        --sample_size 20 \
                        --dataset_name cesm \
                        --projection CC \
                        --data_dir Data/cesm_patches/borde_bg_20/  \
                        --image_size 256 \
                        --channels 1 \

python3 val_metrics.py  --name_exp Unet-Deep-Patches-20-CC \
                            --name_exp_fig "Deep Unet, Patches 20%, CC" \
                            --path_results Results/07-Unet-Based/ \
                            --path_data Data/cesm_patches/borde_bg_20/  \
                            --model UNet_Deep \
                            --epoch 400 \
