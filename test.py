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
# 保存为Excel文件
excel_path = './excel/beibei/c11g111l0.01r0.001.xlsx'
data.to_excel(excel_path, index=False)
