import pandas as pd

# 初始化一个空DataFrame
data = pd.DataFrame({
    "Epoch": [],
    "metric1": [],
    "metric2": [],
    "metric3": [],
    "metric4": [],
    "metric5": [],
    "metric6": [],
    "metric7": [],
    "metric8": []
})

# 示例数据
epochs = [1, 2, 3, 4, 5]
for epoch in epochs:
    epoch_time = 1.23 * epoch  # 假设每个epoch的时间是1.23秒乘以epoch
    test_metric_dict = {"metric1": 0.87 * epoch, "metric2": 0.92 * epoch, "metric3": 0.78 * epoch, "metric4": 0.65 * epoch, 
                        "metric5": 0.89 * epoch, "metric6": 0.95 * epoch, "metric7": 0.67 * epoch, "metric8": 0.76 * epoch}

    # 创建一个包含当前epoch数据的DataFrame
    current_epoch_data = pd.DataFrame({
        "Epoch": [epoch + 1],
        "metric1": [test_metric_dict["metric1"]],
        "metric2": [test_metric_dict["metric2"]],
        "metric3": [test_metric_dict["metric3"]],
        "metric4": [test_metric_dict["metric4"]],
        "metric5": [test_metric_dict["metric5"]],
        "metric6": [test_metric_dict["metric6"]],
        "metric7": [test_metric_dict["metric7"]],
        "metric8": [test_metric_dict["metric8"]]
    })

    # 使用concat方法将当前epoch的数据添加到data中
    data = pd.concat([data, current_epoch_data], ignore_index=True)

# 保存为Excel文件
excel_path = './excel/beibei/c11g111l0.01r0.001.xlsx'
data.to_excel(excel_path, index=False)