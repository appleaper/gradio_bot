

import pandas as pd

# 定义要读取的 Excel 文件路径
file_path = '/home/pandas/snap/code/RapidOcr/learning/useful/temp.xlsx'

# 读取 Excel 文件，默认读取第一个工作表
df = pd.read_excel(file_path)

# 查看数据基本信息和前几行
print('数据基本信息：')
df.info()
print('数据前几行信息：')
print(df.head().to_csv(sep='\t', na_rep='nan'))