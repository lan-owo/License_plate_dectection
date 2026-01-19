import csv
import pymysql
import pandas as pd

df = pd.read_csv('result/output.csv', encoding='utf8')
df = df[df['car_id'].apply(lambda x: len(str(x)) in [7, 8])]
counts = df['car_id'].value_counts()
# 保留出现次数大于等于2次的唯一值
df = df[df['car_id'].isin(counts[counts >= 3].index)]
df = df.drop_duplicates(subset=['car_id'])
df = df[(df['car_id'].str[1].str.isalpha())]
df.to_csv('result/result.csv', index=False)