import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

# 读取CSV文件
csv_file = "RFE/Lueyang.csv"
data = pd.read_csv(csv_file)

# 分割数据，第一列是目标变量（径流量），其余列是特征（协变量）
X = data.iloc[:, 1:]  # 协变量
y = data.iloc[:, 0]   # 目标变量

# 创建随机森林回归模型
rf = RandomForestRegressor()

# 创建RFE对象，指定模型和要选择的特征数量
rfe = RFE(estimator=rf, n_features_to_select=15, step=2)

# 执行RFE特征选择
rfe.fit(X, y)

# 获取选择的特征列索引
selected_features_indexes = rfe.support_

# 根据索引获取选择的特征列名
selected_features = X.columns[selected_features_indexes]

print("最终选择的五个最有影响的特征为：", selected_features.tolist())