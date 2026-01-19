import csv
import pymysql
import pandas as pd

# 连接到 MySQL 数据库
cnx = pymysql.connect(
    host='localhost',
    user='root',
    password='Ol020124',
    database='traffic_management',
    port=3306,
    charset='utf8'
)
cursor = cnx.cursor()

# 读取 CSV 文件并将数据插入到数据库表中
filename = 'C:/Users/OuLan/Desktop/zzz.csv'  # 替换为你的 CSV 文件路径
table_name = 'car_information'  # 替换为你的目标数据库表名

with open(filename, 'r', encoding='gbk') as file:
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
