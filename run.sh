source /home/mirplab/Documents/kevin/venvs/cesm_venv/bin/activate
cd /home/mirplab/Documents/kevin/01-cesm_net/


# Experiment Unet CC
python3 run_models.py   --gpus 0 \
                        --dataset_name cesm \
                        --result_dir Results/07-Unet-Based/ \
                        --projection CC \
                        --exp_name Unet-Deep-Patches-CC\
                        --data_dir Data/cdd_cesm_patches/ \
                        --image_size 256 \
                        --channels 1 \
                        --batch_size 20 \
                        --n_epochs 201 \
                        --workers 12 \
                        --sample_interval 50 \
                        --checkpoint_interval 50 \
                        --model UNet_Deep \
                        --sample_size 20 \
                        --use_wandb True \

python3 run_models.py   --gpus 0 \
                        --generate \
                        --epoch 200 \
                        --model UNet_Deep \
                        --exp_name Unet-Deep-Patches-CC \
                        --result_dir Results/07-Unet-Based/ \
                        --sample_size 20 \
                        --dataset_name cesm \
                        --projection CC \
                        --data_dir Data/cdd_cesm_patches/  \
                        --image_size 256 \
                        --channels 1 \

python3 val_metrics.py  --name_exp Unet-Deep-Patches-CC \
                            --name_exp_fig "Unet deep, Patches, CC" \
                            --path_results Results/07-Unet-Based/ \
                            --path_data Data/cdd_cesm_patches/  \
                            --model UNet_Deep \
                            --epoch 200
