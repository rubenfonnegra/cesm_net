source /home/mirplab/Documents/kevin/venvs/cesm_venv/bin/activate
cd /home/mirplab/Documents/kevin/01-cesm_net/01-Data-Extraction/


python3 data_tool.py --pr_id "CC" --name "cdd_cesm_patches" --name_data "cdd-cesm"
python3 data_tool.py --pr_id "MLO" --name "cdd_cesm_patches" --name_data "cdd-cesm"
