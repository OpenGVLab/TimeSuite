import json
import csv

file_dir="/path_to_the_timesuite_root_folder/scripts/Ablation/Token_Shufflue/F128_CF8_PoolMax_Ablation_PoolMax/Egoschema_test_ckpt_01"

# 加载你的json数据
with open(f'{file_dir}/result.json', 'r') as f:
    data = json.load(f)

# 将数据写入CSV文件
with open(f'{file_dir}/result_submit_kaggle.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['q_uid', 'answer'])  
    for key, value in data.items():
        writer.writerow([key, value]) # 写入数据
