source /media/labmirp/Datos/venvs/cesm_env/bin/activate
cd /media/labmirp/Datos/1_Kevin/02-CEM-Exp/03-Ruben-Implementation/


#### Experiments with data 10 porcent ####
# Experiment Unet CC bg_10
python3 run_models.py --gpus 0 --dataset_name cesm --projection CC --exp_name unet_CC_bg_10 --data_dir Data/cesm_patches/bg_porcent_10/ --image_size 256 --channels 1 --batch_size 25 --n_epochs 201 --workers 12 --sample_interval 100 --checkpoint_interval 50 --model UNet
python3 run_models.py --gpus 0 --generate --epoch 200 --exp_name unet_CC_bg_10 --sample_size 20 --dataset_name cesm --projection CC --data_dir Data/cesm_patches/bg_porcent_10/ --image_size 256 --channels 1

# Experiment Unet MLO bg_10
python3 run_models.py --gpus 0 --dataset_name cesm --projection MLO --exp_name unet_MLO_bg_10 --data_dir Data/cesm_patches/bg_porcent_10/ --image_size 256 --channels 1 --batch_size 25 --n_epochs 201 --workers 12 --sample_interval 100 --checkpoint_interval 50 --model UNet
python3 run_models.py --gpus 0 --generate --epoch 200 --exp_name unet_MLO_bg_10 --sample_size 20 --dataset_name cesm --projection MLO --data_dir Data/cesm_patches/bg_porcent_10/ --image_size 256 --channels 1

#### Experiments with data 20 porcent ####
# Experiment Unet CC bg_20
python3 run_models.py --gpus 0 --dataset_name cesm --projection CC --exp_name unet_CC_bg_20 --data_dir Data/cesm_patches/bg_porcent_20/ --image_size 256 --channels 1 --batch_size 25 --n_epochs 201 --workers 12 --sample_interval 100 --checkpoint_interval 50 --model UNet
python3 run_models.py --gpus 0 --generate --epoch 200 --exp_name unet_CC_bg_20 --sample_size 20 --dataset_name cesm --projection CC --data_dir Data/cesm_patches/bg_porcent_20/ --image_size 256 --channels 1

# Experiment Unet MLO bg_20
python3 run_models.py --gpus 0 --dataset_name cesm --projection MLO --exp_name unet_MLO_bg_20 --data_dir Data/cesm_patches/bg_porcent_20/ --image_size 256 --channels 1 --batch_size 25 --n_epochs 201 --workers 12 --sample_interval 100 --checkpoint_interval 50 --model UNet
python3 run_models.py --gpus 0 --generate --epoch 200 --exp_name unet_MLO_bg_20 --sample_size 20 --dataset_name cesm --projection MLO --data_dir Data/cesm_patches/bg_porcent_20/ --image_size 256 --channels 1

# Experiment GAN CC bg_20
python3 run_models.py --gpus 0 --dataset_name cesm --projection CC --exp_name gan_CC_bg_20 --data_dir Data/cesm_patches/bg_porcent_20/ --image_size 256 --channels 1 --batch_size 25 --n_epochs 201 --workers 12 --sample_interval 100 --checkpoint_interval 50 --model GAN
python3 run_models.py --gpus 0 --generate --epoch 200 --exp_name gan_CC_bg_20 --sample_size 20 --dataset_name cesm --projection CC --data_dir Data/cesm_patches/bg_porcent_20/ --image_size 256 --channels 1

# Experiment GAN MLO bg_20
python3 run_models.py --gpus 0 --dataset_name cesm --projection MLO --exp_name gan_MLO_bg_20 --data_dir Data/cesm_patches/bg_porcent_20/ --image_size 256 --channels 1 --batch_size 25 --n_epochs 201 --workers 12 --sample_interval 100 --checkpoint_interval 50 --model GAN
python3 run_models.py --gpus 0 --generate --epoch 200 --exp_name gan_MLO_bg_20 --sample_size 20 --dataset_name cesm --projection MLO --data_dir Data/cesm_patches/bg_porcent_20/ --image_size 256 --channels 1
