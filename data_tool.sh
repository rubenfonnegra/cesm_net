source /media/labmirp/Datos/venvs/cesm_env/bin/activate
cd /media/labmirp/Datos/1_Kevin/02-CEM-Exp/03-Ruben-Implementation/


python3 data_tool.py --b_id "L" --pr_id "CC" --porcent 10 --name "bg_porcent_10"
python3 data_tool.py --b_id "R" --pr_id "CC" --porcent 10 --name "bg_porcent_10"
python3 data_tool.py --b_id "L" --pr_id "MLO" --porcent 10 --name "bg_porcent_10"
python3 data_tool.py --b_id "R" --pr_id "MLO" --porcent 10 --name "bg_porcent_10"

python3 data_tool.py --b_id "L" --pr_id "CC" --porcent 20 --name "bg_porcent_20"
python3 data_tool.py --b_id "R" --pr_id "CC" --porcent 20 --name "bg_porcent_20"
python3 data_tool.py --b_id "L" --pr_id "MLO" --porcent 20 --name "bg_porcent_20"
python3 data_tool.py --b_id "R" --pr_id "MLO" --porcent 20 --name "bg_porcent_20"