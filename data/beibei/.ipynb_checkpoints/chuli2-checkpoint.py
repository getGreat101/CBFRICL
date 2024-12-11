
import json

# 初始化一个空字典来存储汇总后的数据
combined_data = {}

# 定义文件列表
file_names = ['buy_dict.txt', 'cart_dict.txt', 'view_dict.txt']

# 读取每个文件并汇总数据
for file_name in file_names:
    with open(file_name, 'r') as file:
        # 假设文件中的数据是以JSON格式存储的
        data = json.load(file)
        for key, values in data.items():
            if key in combined_data:
                # 合并列表，并使用集合去除重复的值
                combined_data[key] = list(set(combined_data[key] + values))
            else:
                combined_data[key] = values

# 对每个键的值进行排序
for key in combined_data:
    combined_data[key].sort()

# 将汇总后的数据转换为字符串格式，并保存到文本文件
output_str = json.dumps(combined_data, indent=4)

with open('all_dict.txt', 'w') as output_file:
    output_file.write(output_str)