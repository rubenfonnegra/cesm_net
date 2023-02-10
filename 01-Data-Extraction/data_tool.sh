source /media/labmirp/Datos/venvs/cesm_env/bin/activate
cd /media/labmirp/Datos/workspaces/cesm_net/01-Data-Extraction/


python3 data_tool.py --b_id "L" --pr_id "CC" --name "data_img_complete"
python3 data_tool.py --b_id "R" --pr_id "CC" --name "data_img_complete"
# python3 data_tool.py --b_id "L" --pr_id "MLO" --name "data_without_background"
# python3 data_tool.py --b_id "R" --pr_id "MLO" --name "data_without_background"
