source /media/labmirp/Datos/venvs/cesm_env/bin/activate
cd /media/labmirp/Datos/workspaces/cesm_net/01-Data-Extraction/


python3 data_tool.py --pr_id CC \
                    --b_id L \
                    --name sura_crop_image \
                    --name_data cesm \
                    --plot_resize \
                    --crop

python3 data_tool.py --pr_id CC \
                    --b_id R \
                    --name sura_crop_image \
                    --name_data cesm \
                    --plot_resize \
                    --crop \

python3 data_tool.py --pr_id MLO\
                    --b_id L \
                    --name sura_crop_image \
                    --name_data cesm \
                    --plot_resize \
                    --crop \

python3 data_tool.py --pr_id MLO \
                    --b_id R\
                    --name sura_crop_image \
                    --name_data cesm\
                    --plot_resize \
                    --crop \
