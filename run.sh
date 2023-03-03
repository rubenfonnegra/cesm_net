source /media/labmirp/Datos/venvs/cesm_env/bin/activate
cd /media/labmirp/Datos/workspaces/cesm_net/


# Experiment Unet CC
python3 run_models.py   --gpus 0 \
                        --dataset_name cesm \
                        --result_dir Results/ \
                        --projection CC \
                        --exp_name Residual_PA_New_Patches_CC\
                        --data_dir Data/cesm_patches/cdd_cesm_patches/ \
                        --image_size 256 \
                        --channels 1 \
                        --batch_size 20 \
                        --n_epochs 101 \
                        --workers 12 \
                        --sample_interval 10 \
                        --checkpoint_interval 10 \
                        --model Residual-PA-Unet \
                        --type_model Attention \
                        --sample_size 50 \
                        --use_wandb True \

# python3 run_models.py   --gpus 0 \
#                         --generate \
#                         --epoch 100 \
#                         --model Residual-PA-Unet \
#                         --exp_name Residual_PA_New_Patches_CC \
#                         --result_dir Results/ \
#                         --sample_size 50 \
#                         --dataset_name cesm \
#                         --projection CC \
#                         --data_dir Data/cdd_cesm_patches/  \
#                         --image_size 256 \
#                         --channels 1 \

# python3 val_metrics.py  --name_exp Unet-Deep-Patches-CC \
#                             --name_exp_fig "Unet deep, Patches, CC" \
#                             --path_results Results/07-Unet-Based/ \
#                             --path_data Data/cdd_cesm_patches/  \
#                             --model UNet_Deep \
#                             --epoch 200
