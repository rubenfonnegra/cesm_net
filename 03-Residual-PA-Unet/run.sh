source /media/labmirp/Datos/venvs/cesm_env/bin/activate
cd /media/labmirp/Datos/workspaces/cesm_net/03-Residual-PA-Unet/


#### Experiments with data 20 porcent ####
# Experiment Unet CC
python3 run_models.py   --gpus 0 \
                        --dataset_name cesm \
                        --result_dir /media/labmirp/Datos/workspaces/cesm_net/Results/03-Residual-Pixel-Attention-Unet/ \
                        --projection CC \
                        --exp_name residual-PA-unet-original \
                        --data_dir /media/labmirp/Datos/workspaces/cesm_net/Data/cesm_patches/original/ \
                        --image_size 256 \
                        --channels 1 \
                        --batch_size 5 \
                        --n_epochs 100 \
                        --workers 12 \
                        --sample_interval 10 \
                        --checkpoint_interval 50 \
                        --model UNet
#python3 run_models.py --gpus 0 --generate --epoch 200 --exp_name unet_CC_bg_30_01 --sample_size 50 --dataset_name cesm --projection CC --data_dir Data/cesm_patches/borde_bg_30/ --image_size 256 --channels 1

# Experiment Unet MLO
#python3 run_models.py --gpus 0 --dataset_name cesm --projection MLO --exp_name unet_MLO_bg --data_dir Data/cesm_patches/borde_bg_20/ --image_size 256 --channels 1 --batch_size 30 --n_epochs 201 --workers 12 --sample_interval 100 --checkpoint_interval 50 --model UNet
#python3 run_models.py --gpus 0 --generate --epoch 200 --exp_name unet_MLO_bg --sample_size 20 --dataset_name cesm --projection MLO --data_dir Data/cesm_patches/borde_bg_20/ --image_size 256 --channels 1
