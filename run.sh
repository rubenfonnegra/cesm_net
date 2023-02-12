source /home/mirplab/Documents/kevin/venvs/cesm_venv/bin/activate
cd /home/mirplab/Documents/kevin/01-cesm_net/


#### Experiments with data 20 porcent ####
# Experiment Unet CC
python3 run_models.py   --gpus 0 \
                        --dataset_name cesm \
                        --result_dir /home/mirplab/Documents/kevin/01-cesm_net/Results \
                        --projection CC \
                        --exp_name residual-PA2-unet-data-image-complete-increase-lr-pixel \
                        --data_dir /media/mirplab/TB2/Experiments-Mammography/01_Data/data_img_complete/ \
                        --image_size 256 \
                        --channels 1 \
                        --batch_size 10 \
                        --n_epochs 601 \
                        --workers 12 \
                        --sample_interval 20 \
                        --checkpoint_interval 50 \
                        --model Residual-PA-Unet \
                        --sample_size 20 \
                        --use_wandb True \
                        --img_complete 

python3 run_models.py   --gpus 0 \
                        --generate \
                        --epoch 600 \
                        --model Residual-PA-Unet \
                        --exp_name residual-PA2-unet-data-image-complete-increase-lr-pixel \
                        --result_dir /home/mirplab/Documents/kevin/01-cesm_net/Results \
                        --sample_size 20 \
                        --dataset_name cesm \
                        --projection CC \
                        --data_dir /media/mirplab/TB2/Experiments-Mammography/01_Data/data_img_complete/ \
                        --image_size 256 \
                        --channels 1 \
                        --img_complete

# python3 run_models.py   --gpus 0 \
#                         --dataset_name cesm \
#                         --result_dir /home/mirplab/Documents/kevin/01-cesm_net/Results \
#                         --projection CC \
#                         --exp_name SA-Unet-Generator-data-image-complete \
#                         --data_dir /media/mirplab/TB2/Experiments-Mammography/01_Data/data_img_complete/ \
#                         --image_size 256 \
#                         --channels 1 \
#                         --batch_size 10 \
#                         --n_epochs 401 \
#                         --workers 12 \
#                         --sample_interval 20 \
#                         --checkpoint_interval 50 \
#                         --model UNet \
#                         --sample_size 20 \
#                         --use_wandb True \
#                         --img_complete 

# python3 run_models.py   --gpus 0 \
#                         --generate \
#                         --epoch 400 \
#                         --model UNet \
#                         --exp_name SA-Unet-Generator-data-image-complete \
#                         --result_dir /home/mirplab/Documents/kevin/01-cesm_net/Results \
#                         --sample_size 20 \
#                         --dataset_name cesm \
#                         --projection CC \
#                         --data_dir /media/mirplab/TB2/Experiments-Mammography/01_Data/data_img_complete/ \
#                         --image_size 256 \
#                         --channels 1 \
#                         --img_complete
# Experiment Unet MLO
#python3 run_models.py --gpus 0 --dataset_name cesm --projection MLO --exp_name unet_MLO_bg --data_dir Data/cesm_patches/borde_bg_20/ --image_size 256 --channels 1 --batch_size 30 --n_epochs 201 --workers 12 --sample_interval 100 --checkpoint_interval 50 --model UNet
#python3 run_models.py --gpus 0 --generate --epoch 200 --exp_name unet_MLO_bg --sample_size 20 --dataset_name cesm --projection MLO --data_dir Data/cesm_patches/borde_bg_20/ --image_size 256 --channels 1
