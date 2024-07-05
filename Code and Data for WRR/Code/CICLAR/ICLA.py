#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置环境变量，减少TensorFlow的日志信息
import pandas as pd  # 导入pandas库，用于数据处理和分析
import numpy as np  # 导入numpy库，用于数值计算
import tensorflow as tf  # 导入tensorflow库，用于深度学习模型构建和训练
from sklearn.preprocessing import StandardScaler  # 从sklearn.preprocessing导入StandardScaler，用于数据标准化
from tensorflow.keras.layers import *  # 从tensorflow.keras.layers导入所有层，用于模型构建
import keras.backend as K  # 从keras导入后端接口backend
from scipy.io import savemat, loadmat  # 从scipy.io导入savemat和loadmat，用于MATLAB文件的读写
from keras.callbacks import LearningRateScheduler  # 从keras.callbacks导入LearningRateScheduler，用于调整学习率
from tensorflow.keras.models import Model  # 重复导入，可以删除
from tensorflow.keras.layers import Input, LSTM, Dense  # 从tensorflow.keras.layers导入Input, LSTM, Dense，用于模型构建
from tensorflow.keras.regularizers import l2  # 从tensorflow.keras.regularizers导入l2，用于L2正则化
from keras.layers import LSTM, Bidirectional

from ISMA import SMA
from AM import attention_layer
from EVALUATE import load_data,calculate_mean_total,print_results,transform_and_append
from PLOT import plot_predictions
from reshapeData import series_to_supervised
import csv
# In[1]加载数据、协变量:

u, data = load_data("数据集/BB.xlsx", "数据集/FS_Beibei.csv")

# In[2]进行适应度计算,以验证集均方差为适应度函数，目的是找到一组超参数 使得网络的误差最小:

def fun(pop, P, T, Pt, Tt):
    tf.random.set_seed(0)  # 设置随机种子以确保结果可重复

    alpha = pop[0]  # 从种群参数中获取学习率
    hidden_nodes0 = int(pop[1])  # 从种群参数中获取LSTM层的神经元数量

    # 添加卷积层和池化层
    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]))
    conv1d = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    maxpooling = MaxPooling1D(pool_size=2)(conv1d)
    reshaped = Reshape((-1, 64 * maxpooling.shape[1]))(maxpooling)

    # 添加LSTM层，设置激活函数为SELU，添加了L2正则化，以防止过拟合
    lstm = LSTM(hidden_nodes0, activation='selu', return_sequences=True, kernel_regularizer=l2(0.01))(reshaped)

    # 添加注意力机制
    attention = attention_layer(lstm, train_X.shape[1])

    attention_flatten = Flatten()(attention)

    # 添加全连接层
    outputs = Dense(train_y.shape[1])(attention_flatten)

    # 创建并编译模型
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')

    #输出模型结构
    #model.summary()

    # 训练模型，verbose=2: 在每个 epoch 结束时输出详细信息
    model.fit(train_X, train_y, epochs=10, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    # 进行预测并计算均方误差
    test_pred = model.predict(test_X)
    F2 = np.mean(np.square((test_pred - test_y)))

    return F2

# In[3]初始化两个空列表，用于存储后续的预测值和实际值:

prediction_test = []
actual_test = []
prediction_train = []
actual_train = []

# In[4]神经网络构建：

#for imf in u:
for imf_idx, imf in enumerate(u):
    # 遍历u中的每个内在模态函数（IMF）。

    imf = imf.reshape(-1, 1)
    # 将当前的IMF重塑为二维数组，以便进行后续操作。

    imf = pd.DataFrame(imf)
    # 将重塑后的IMF转换为pandas DataFrame。

    data_1 = pd.DataFrame(data)
    # 将其他特征转换为pandas DataFrame。

    data_ = pd.concat((imf, data_1), axis=1)
    # 将IMF和其他特征合并为一个DataFrame，IMF作为新的列添加到data_1的前面。

    data_ = data_.values
    # 将合并后的DataFrame转换为NumPy数组，以进行后续的数值操作。

    data_ = data_.astype('float32')
    # 将数组中的数据类型转换为float32，这是为了确保数据类型的一致性，特别是在进行深度学习或其他数值计算时。
    # 输出当前处理的是第几个IMF
    print(data_)
    print(f"\nProcessing IMF {imf_idx + 1}")

    # 构建成监督学习问题
    n_in = 12  # 输入步数
    n_out = 1  # 输出步数
    n_vars = data_.shape[1]  # 获取values数组的列数，即变量的数量。
    # 构建成监督学习问题
    reframed = series_to_supervised(data_, n_in, n_out)
    # 使用series_to_supervised函数将数据转换为监督学习格式。
    # values是要转换的数据，n_in是输入序列的长度，n_out是输出序列的长度。
    # 此函数返回一个新的DataFrame，其中包含用于训练监督学习模型所需的输入和输出数据。

    # 取出保留的变量
    contain_vars = []
    # 初始化一个空列表，用于存储需要保留的列名。

    for i in range(1, n_in + 1):
        contain_vars += [('var%d(t-%d)' % (j, i)) for j in range(1, n_vars + 1)]
    # 遍历从1到n_in的数字，每次迭代都会生成一组列名。
    # 这些列名表示过去n_in个时间步的每个变量（比如var1(t-1), var2(t-1), ..., var1(t-n_in), var2(t-n_in)等）。

    data3 = reframed[contain_vars + ['var1(t)'] + [('var1(t+%d)' % (j)) for j in range(1, n_out)]]
    # 从reframed数据框中提取特定的列。
    # 'contain_vars'包含了输入序列需要的所有列。
    # ['var1(t)'] 是当前时间步的第一个变量。
    # [('var1(t+%d)' % (j)) for j in range(1, n_out)] 生成了输出序列的列名，即预测未来n_out-1个时间步的第一个变量。
    values = data3.values  # 将前面处理好的数据转换成numpy数组，方便后续的数据操作
    n_train_hours = int(values.shape[0] * 0.8)  # 80%训练集
    train = values[:n_train_hours, :]  # 从values数组中取出前n_train_hours行作为训练集。
    test = values[n_train_hours:, :]  # 从values数组中取出剩下的行作为测试集。
    # 归一化
    scaler = StandardScaler()
    train = scaler.fit_transform(train)  # 对训练集数据进行标准化
    test = scaler.fit_transform(test)  # 对测试集数据进行标准化。
    # 把数据分为输入和输出
    train_X, train_y = train[:, :n_in * n_vars], train[:, n_in * n_vars:]
    test_X, test_y = test[:, :n_in * n_vars], test[:, n_in * n_vars:]
    # 把输入重塑成3D格式 [样例，时间步， 特征]
    train_X = train_X.reshape((train_X.shape[0], n_in, n_vars))
    test_X = test_X.reshape((test_X.shape[0], n_in, n_vars))

    # 开始优化参数
    best, trace, result = SMA(train_X, train_y, test_X, test_y, fun)
    # 保存优化结果
    savemat('result/woa_para.mat', {'trace': trace, 'best': best, 'result': result})
    print("最优学习率、LSTM层神经元的参数分别为：", [int(best[i]) if i > 0 else best[i] for i in range(len(best))])

    #  加载优化的参数  这里加载结果只是起到一个临时中间媒介的作用，下一次优化的结果会覆盖之前的结果  所以每次调用的都是各自imf分量优化的参数
    pop = loadmat('result/woa_para.mat')['best'].reshape(-1, )

    alpha = pop[0]  # 学习率
    hidden_nodes0 = int(pop[1])  # 第一隐含层神经元


    def scheduler(epoch):
        # 定义一个学习率调度器函数，用于每隔10个epoch将学习率减小为原来的1/2
        if epoch % 10 == 0 and epoch != 0:
            lr = K.get_value(tmp.optimizer.lr)  # 获取当前学习率
            K.set_value(tmp.optimizer.lr, lr * 0.5)  # 将学习率设置为原来的一半
            print("lr changed to {}".format(lr * 0.5))  # 打印新的学习率值
        return K.get_value(tmp.optimizer.lr)  # 返回新的学习率值


    reduce_lr = LearningRateScheduler(scheduler)  # 使用Keras的LearningRateScheduler调用学习率调度器

    # 添加卷积层和池化层
    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]))
    conv1d = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    maxpooling = MaxPooling1D(pool_size=2)(conv1d)
    reshaped = Reshape((-1, 64 * maxpooling.shape[1]))(maxpooling)


    lstm = LSTM(hidden_nodes0, activation='selu', return_sequences=True, kernel_regularizer=l2(0.01))(reshaped)
    attention = attention_layer(lstm, train_X.shape[1])
    attention_flatten = Flatten()(attention)

    # 添加全连接层
    outputs = Dense(train_y.shape[1])(attention_flatten)

    # 创建并编译模型
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')
    # 输出最优模型的摘要
    model.summary()

    # 训练模型
    model.fit(train_X, train_y, epochs=50, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    yhat = model.predict(test_X)  # 使用模型对测试数据进行预测
    yhat = yhat.reshape(-1, 1)  # 重塑预测结果的形状



    transform_and_append(prediction_test, yhat, n_in, n_vars, n_out, scaler)
    print(prediction_test)



    # 定义文件名
    csv_file = "predictions.csv"

    # 获取最大IMF的长度
    max_imf_length = max(len(predictions) for predictions in prediction_test)

    # 将预测结果保存到CSV文件中
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入标题行
        writer.writerow(["Index"] + [f"IMF {imf_idx + 1}" for imf_idx in range(len(prediction_test))])
        # 写入每个时间步的预测结果
        for time_step in range(max_imf_length):
            row = [time_step + 1]  # 第一列是时间步
            for predictions in prediction_test:
                # 如果当前时间步小于该模态的长度，则写入预测值，否则写入空值
                if time_step < len(predictions):
                    row.append(predictions[time_step][0])
                else:
                    row.append(None)
            writer.writerow(row)

    print("预测结果已保存到 predictions.csv 文件中。")


# In[5]对IMF求和：
mean_pre_test, mean_actual_test = calculate_mean_total(prediction_test, actual_test)

# In[6]输出评估指标:
print_results(mean_actual_test, mean_pre_test, n_out)

# In[7]:

actual = [float(row[0]) for row in mean_actual_test]
# 从反向缩放后的实际值集合(mean_actual_test)中提取每个样本的第一个值。
# 这个列表包含了对应于时间步长t的所有实际值。

predicted_1st = [float(forecast[0]) for forecast in mean_pre_test]
# 从反向缩放后的预测值集合(mean_pre_test)中提取每个样本的第一个预测值。
# 这个列表包含了模型对于时间步长t+1的所有预测值。


plot_predictions(actual, predicted_1st)

