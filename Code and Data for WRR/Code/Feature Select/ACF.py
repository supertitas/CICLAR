import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 读取存储径流序列的 CSV 文件
csv_file = "BB.xlsx"
data = pd.read_excel(csv_file)

# 从 CSV 文件中提取径流序列数据（假设存储在第一行）
flow_series = data.iloc[:, 1]

# 绘制 ACF 图
plt.figure(figsize=(10, 5))
plot_acf(flow_series, lags=24, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.grid(True)
plt.show()

