import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

# 加载CSV文件
file_path = 'NSI方法-绘制tanθ时序图像-示例.csv'  # 如果需要，更新此路径
data = pd.read_csv(file_path, header=None, names=['Date', 'tanθ'])

# 从数据中提取日期和tanθ值
data[['Date', 'tanθ']] = data['Date'].str.extract(r'(D\d{8}):\s?([\d.]+)')

# 从日期列中去掉“D”，并转换为日期时间格式
data['Date'] = data['Date'].str.replace('D', '').astype(int)
data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')

# 将tanθ列转换为数字类型
data['tanθ'] = pd.to_numeric(data['tanθ'], errors='coerce')

# 为每年生成关键日期，只有1月1日、4月1日、7月1日和10月1日
key_dates = pd.to_datetime(['2019-01-01', '2019-04-01', '2019-07-01', '2019-10-01',
                            '2020-01-01', '2020-04-01', '2020-07-01', '2020-10-01',
                            '2021-01-01'])

# 为这些日期创建相应的标签
key_labels = ['2019.1', '2019.4', '2019.7', '2019.10', '2020.1', '2020.4', '2020.7', '2020.10', '2021.1']

# 绘制数据
plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['tanθ'], marker='o', linestyle='-', color='b')

# 设置标签和标题，字体为'Times New Roman'
plt.xlabel('Date', fontsize=18, fontname='Times New Roman')
plt.ylabel('tanθ', fontsize=18, fontname='Times New Roman')
plt.title('The Result of NSI Method', fontsize=18, fontname='Times New Roman')

# 设置y轴的限制
plt.ylim(0, 0.003)

# 根据实际数据范围设置x轴的刻度和标签
plt.xticks(key_dates, key_labels, rotation=45, fontsize=16, fontname='Times New Roman')

# 调整y轴刻度标签为Times New Roman字体，并设置字体大小
plt.yticks(fontname='Times New Roman', fontsize=18)

# 添加网格线以提高可读性，并确保它们均匀间隔
plt.grid(True, linestyle='--', color='gray', alpha=0.7)

# 显示图表
plt.tight_layout()
plt.show()
