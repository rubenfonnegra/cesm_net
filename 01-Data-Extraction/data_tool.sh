source /home/mirplab/Documents/kevin/venvs/cesm_venv/bin/activate
cd /home/mirplab/Documents/kevin/01-cesm_net/01-Data-Extraction/


python3 data_tool.py    --pr_id "CC" \
                        --b_id "L" \
                        --name "sura_crop_image" \
                        --name_data "cesm" \
                        --path_data 
                        --plot_resize
python3 data_tool.py --pr_id "MLO" --name "cdd_cesm_full_image" --name_data "cdd-cesm" --plot_resize
