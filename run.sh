source /media/labmirp/Datos/venvs/cesm_env/bin/activate
cd /media/labmirp/Datos/workspaces/cesm_net/


# Experiment Unet CC
python3 run_models.py   --gpus 0 \
                        --dataset_name cesm \
                        --result_dir Results/07-Unet-Based/ \
                        --projection CC \
                        --exp_name Unet-Not-Deep-Image-Complete-CC-2\
                        --data_dir Data/cesm_patches/data_img_complete/ \
                        --image_size 256 \
                        --channels 1 \
                        --batch_size 20 \
                        --n_epochs 601 \
                        --workers 12 \
                        --sample_interval 50 \
                        --checkpoint_interval 50 \
                        --model UNet \
                        --sample_size 20 \
                        --use_wandb True \

python3 run_models.py   --gpus 0 \
                        --generate \
                        --epoch 600 \
                        --model UNet \
                        --exp_name Unet-Not-Deep-Image-Complete-CC-2 \
                        --result_dir Results/07-Unet-Based/ \
                        --sample_size 20 \
                        --dataset_name cesm \
                        --projection CC \
                        --data_dir Data/cesm_patches/data_img_complete/  \
                        --image_size 256 \
                        --channels 1 \

python3 comparation_exp.py  --name_exp Unet-Not-Deep-Image-Complete-CC-2 \
                            --name_exp_fig "Unet not deep, Image Complete, CC" \
                            --path_results Results/07-Unet-Based/ \
                            --path_data Data/cesm_patches/data_img_complete/  \
                            --model UNet \
                            --epoch 600
