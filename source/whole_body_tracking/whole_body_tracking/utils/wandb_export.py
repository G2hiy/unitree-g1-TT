import wandb
import pandas as pd  # 补充必要依赖（之前的核心问题）
import os  # 新增：用于获取文件路径

api = wandb.Api()
run = api.run("874374490-xiamen-university/TT/b7df9i64")

# 修复：列表转DataFrame
metrics_list = run.history()
metrics_dataframe = pd.DataFrame(metrics_list)

# 保存并打印完整路径
file_name = "metrics.csv"
metrics_dataframe.to_csv(file_name)

# 打印文件的绝对路径
abs_path = os.path.abspath(file_name)
print(f"✅ CSV文件已保存到：{abs_path}")