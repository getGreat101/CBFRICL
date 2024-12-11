
# 初始化一个空字典来存储数据
data_dict = {}

# 读取数据，这里假设您的数据存储在一个名为"data.txt"的文件中
with open('validation.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split()  # 去除空格并分割每一行
        if key in data_dict:
            data_dict[key].append(int(value))  # 如果键已存在，添加值到列表
        else:
            data_dict[key] = [int(value)]  # 如果键不存在，创建新的键和值

# 将字典转换为字符串格式，并保存到文本文件
# 使用json.dumps来确保格式正确，包括双引号
import json
output_str = json.dumps(data_dict, indent=4)

with open('validation_dict.txt', 'w') as output_file:
    output_file.write(output_str)


