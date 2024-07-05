import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import pandas as pd

def load_data(excel_file, csv_file):
    # 加载未分解或者模态分解后的数据集
    dataset = pd.read_excel(excel_file)
    # 删除含有缺失值的行
    dataset = dataset.dropna()
    # 转置
    dataset = dataset.T
    u = dataset.values  # Pandas DataFrame->NumPy数组，赋值给u
    # print(dataset)

    # 协变量文件读取
    # 参数'encoding'设置为'gbk'，这通常用于读取中文字符，确保文件中的中文字符能够正确读取。
    dataset1 = pd.read_csv(csv_file, encoding='gbk')
    # 加载特征数据，: 表示选择所有行，而 2: 表示选择从第三列（索引为2）开始到最后一列的所有列
    data = dataset1.values[:, 0:]
    # print(dataset1)

    return u, data


def mape(y_true, y_pred):
    # 定义一个计算平均绝对百分比误差（MAPE）的函数。
    record = []
    for index in range(len(y_true)):
        # 遍历实际值和预测值。
        if abs(y_true[index]) > 10:
            # 如果实际值的绝对值大于10，则计算MAPE。
            temp_mape = np.abs((y_pred[index] - y_true[index]) / y_true[index])
            # 计算单个预测的MAPE。
            record.append(temp_mape)
            # 将MAPE添加到记录列表中。
    return np.mean(record) * 100
    # 返回所有记录的平均值，乘以100得到百分比。

def nrmse(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    obs_range = np.max(actual) - np.min(actual)
    nrmse_value = rmse / obs_range
    return nrmse_value

#评价指标计算
def evaluate_forecasts(test, forecasts, n_out):
    # 定义一个函数来评估预测的性能。
    rmse_dic = {}
    mae_dic = {}
    mape_dic = {}
    r2_dic = {}
    r_dic = {}
    nsec_dic = {}
    nrmse_dic = {}
    # 初始化存储各个评估指标的字典。

    for i in range(n_out):
        # 遍历每一个预测步长。
        actual = [float(row[i]) for row in test]
        # 从测试集中提取实际值。
        predicted = [float(forecast[i]) for forecast in forecasts]
        # 从预测结果中提取预测值。

        rmse = sqrt(mean_squared_error(actual, predicted))
        # 计算均方根误差（RMSE）。
        rmse_dic['t+' + str(i + 1) + ' RMSE'] = rmse
        # 将RMSE结果添加到字典中。

        nrmse_value = nrmse(actual, predicted)
        # 计算 NRMSE
        nrmse_dic['t+' + str(i + 1) + ' NRMSE'] = nrmse_value
        # 将 NRMSE 结果添加到字典中。

        mae = mean_absolute_error(actual, predicted)
        # 计算平均绝对误差（MAE）。
        mae_dic['t+' + str(i + 1) + ' MAE'] = mae
        # 将MAE结果添加到字典中。

        mape_ = mape(actual, predicted)
        # 计算平均绝对百分比误差（MAPE）。
        mape_dic['t+' + str(i + 1) + ' MAPE'] = mape_
        # 将MAPE结果添加到字典中。

        r2 = r2_score(actual, predicted)
        # 计算R平方值（R2）。
        r2_dic['t+' + str(i + 1) + ' R2'] = r2
        # 将R2结果添加到字典中。

        r = pearsonr(actual, predicted)[0]
        r_dic['t+' + str(i + 1) + ' R'] = r

        mean_actual = np.mean(actual)
        nsec = 1 - (np.sum(np.square(np.subtract(actual, predicted))) / np.sum(
            np.square(np.subtract(actual, mean_actual))))
        nsec_dic['t+' + str(i + 1) + ' NSEC'] = nsec

    return rmse_dic, nrmse_dic, mae_dic, mape_dic,  r_dic, nsec_dic
    #return nrmse_dic, r2_dic, nsec_dic
    # 返回包含所有评估指标的字典。

def print_nested_dictionary(dic_list):
    for dictionaries in dic_list:
        for dictionary in dictionaries:
            for key, value in dictionary.items():
                print(f"{key.ljust(10)}{value}")


def calculate_mean_total(prediction_test, actual_test):
    mean_pre_test = []
    # 初始化一个空列表 mean_pre_test，用于存储每个时间点的预测值总和。
    for i in range(0, len(prediction_test[0])):
        # 遍历预测结果的每个时间点。
        total_sum = 0
        # 初始化变量 total_sum，用于累加当前时间点的所有预测值。
        for j in range(0, len(prediction_test)):
            # 内层循环，遍历所有预测结果。
            total_sum += prediction_test[j][i]
            # 将当前时间点的预测值累加到 total_sum。
        mean_pre_test.append(total_sum)
        # 将累加后的总和添加到 mean_pre_test 列表。
    print("Mean Prediction Test:", mean_pre_test)

    mean_actual_test = []
    # 初始化一个空列表 mean_actual_test，用于存储每个时间点的真实值总和。
    for i in range(0, len(actual_test[0])):
        # 遍历真实结果的每个时间点。
        total_sum = 0
        # 初始化变量 total_sum，用于累加当前时间点的所有真实值。
        for j in range(0, len(actual_test)):
            # 内层循环，遍历所有真实结果。
            total_sum += actual_test[j][i]
            # 将当前时间点的真实值累加到 total_sum。
        mean_actual_test.append(total_sum)
        # 将累加后的总和添加到 mean_actual_test 列表。
    print("Mean Actual Test:", mean_actual_test)

    return mean_pre_test, mean_actual_test

def print_results(mean_actual_test, mean_pre_test, n_out):
    dic_list = []
    # 初始化一个空列表，用于存储评估结果。

    dic = evaluate_forecasts(mean_actual_test, mean_pre_test, n_out)
    # 调用evaluate_forecasts函数。
    # 此函数将计算每个预测步长的RMSE、MAE、MAPE和R2值。

    dic_list.append(dic)
    # 将评估结果（一个包含四种评估指标的字典）添加到dic_list列表中。

    # 显示预测指标数值
    print_nested_dictionary(dic_list)

def transform_and_append(list, y, n_in, n_vars, n_out, scaler):
    y = np.repeat(y, n_in * n_vars + n_out, axis=-1)
    inv_y = scaler.inverse_transform(np.reshape(y, (len(y), n_in * n_vars + n_out)))[:, 0]
    inv_y = inv_y.reshape(-1, n_out)
    list.append(inv_y)  # 将反向缩放后的值添加到列表