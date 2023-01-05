
# On CESM dataset

# Training

### UNet 

First model. UNet backbone. No adv loss. 

```
source /media/labmirp/Datos/venvs/cesm_env/bin/activate 

python run_models.py --gpus 0 \
            --dataset_name cesm \
            --projection MLO \
            --exp_name unet_cs \
            --data_dir Data/cesm_patches/ \
            --image_size 256 --channels 1 \
            --batch_size 25 \
            --n_epochs 201 \
            --sample_interval 100 --checkpoint_interval 50 \
            --model UNet
```



## Validation

