import csv
import pymysql
import pandas as pd

df = pd.read_csv('result/output.csv', encoding='utf8')
df = df[df['car_id'].apply(lambda x: len(str(x)) in [7, 8])]
counts = df['car_id'].value_counts()
# 保留出现次数大于等于2次的唯一值
df = df[df['car_id'].isin(counts[counts >= 2].index)]
df = df.drop_duplicates(subset=['car_id'])
df.to_csv('output.csv', index=False)


# 连接到 MySQL 数据库
cnx = pymysql.connect(
    host='localhost',
    user='root',
    password='Ol020124',
    database='traffic',
    port=3306,
    charset='utf8'
)
cursor = cnx.cursor()

# 读取 CSV 文件并将数据插入到数据库表中
filename = 'result.csv'  # 替换为你的 CSV 文件路径
table_name = 'car_information'  # 替换为你的目标数据库表名

with open(filename, 'r', encoding='utf8') as file:
    csv_data = csv.reader(file)
    headers = next(csv_data)  # 获取 CSV 文件的列头
    # 构建 SQL 插入语句
    insert_query = f"INSERT INTO {table_name} ({', '.join(headers)}) VALUES ({', '.join(['%s']*len(headers))})"
    # 逐行插入数据
    for row in csv_data:
        cursor.execute(insert_query, row)

# 提交更改并关闭数据库连接
cnx.commit()
cursor.close()
cnx.close()
