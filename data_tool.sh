source /media/labmirp/Datos/venvs/cesm_env/bin/activate
cd /media/labmirp/Datos/workspaces/cesm_net/


python3 data_tool.py --b_id "L" --pr_id "CC" --porcent_bg 15 --porcent_borde 15 --name "borde_bg_30"
python3 data_tool.py --b_id "R" --pr_id "CC" --porcent_bg 10 --porcent_borde 15  --name "borde_bg_30"
python3 data_tool.py --b_id "L" --pr_id "MLO" --porcent_bg 15 --porcent_borde 15  --name "borde_bg_30"
python3 data_tool.py --b_id "R" --pr_id "MLO" --porcent_bg 15 --porcent_borde 15  --name "borde_bg_30"
