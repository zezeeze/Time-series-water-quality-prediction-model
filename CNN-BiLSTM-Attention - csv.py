import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from hydroeval import nse
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, Bidirectional, LSTM, Dense,
                                     Dropout, Multiply, Activation, Flatten,
                                     Reshape, Add, GlobalAveragePooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import os


# ------------------- 数据预处理部分---------------------
# 读取数据
data = pd.read_excel(r'Your file path.xlsx')
time = data['Time']
features = data.drop('Time', axis=1)

# 异常值处理
features_clean = features.copy()
for col in features.columns:
    Q1 = features[col].quantile(0.25)
    Q3 = features[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    features_clean[col] = features[col].mask((features[col] < lower_bound) | (features[col] > upper_bound))
features_clean = features_clean.dropna()
time = time[features_clean.index]

# 归一化处理
scalers = {}
features_scaled = pd.DataFrame()
for col in features_clean.columns:
    scaler = MinMaxScaler()
    features_scaled[col] = scaler.fit_transform(features_clean[[col]]).flatten()
    scalers[col] = scaler


# 数据增强
def augment_data(data, noise_level=0.01, scale_range=(0.9, 1.1)):
    augmented_data = data.copy()
    noise = np.random.normal(0, noise_level, augmented_data.shape)
    augmented_data += noise
    scale = np.random.uniform(scale_range[0], scale_range[1], augmented_data.shape[1])
    augmented_data *= scale
    return augmented_data


# 序列生成
def create_sequences(data, time_data, input_length, output_length, augment=False):
    X, y, time_sequences = [], [], []
    for i in range(len(data) - input_length - output_length + 1):
        seq = data[i:i + input_length].values
        if augment:
            seq = augment_data(seq)
        X.append(seq)
        y.append(data[i + input_length:i + input_length + output_length].values)
        time_sequences.append(time_data.iloc[i + input_length:i + input_length + output_length])
    return np.array(X), np.array(y), np.array(time_sequences)


# ------------------- 模型参数设置 ---------------------
input_length = 48  # 可自由修改
output_length = 168
embed_dim = features_scaled.shape[1]
rate = min(0.5, 0.2 + (input_length/48)*0.1) # 根据输入长度动态调整


# ------------------- 超参数自动调整函数 ---------------------
def auto_adjust_hyperparams(input_len):
    # CNN卷积核大小动态计算（3-7之间的奇数）
    kernel_size = max(3, min(int(input_len * 0.05), 7))
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

    # BiLSTM单元数动态计算（64-256之间）
    lstm_units = max(64, min(int(input_len * 0.2), 256))

    return kernel_size, lstm_units


# ------------------- 模型构建 ---------------------
def build_model(input_length, output_length, embed_dim):
    # 自动计算超参数
    kernel_size, lstm_units = auto_adjust_hyperparams(input_length)

    # 输入层
    inputs = Input(shape=(input_length, embed_dim))

    # 通道数动态计算（输入输出长度加权）
    total_steps = input_length + output_length
    channels = max(16, min(int(total_steps * embed_dim * 0.005), 18))


    # ------------------- 第一残差块（CNN） ---------------------
    # 1x1卷积调整维度
    res_conv = Conv1D(channels, 1, padding='same')(inputs)

    # 主CNN路径
    x = Conv1D(channels, kernel_size, padding='same', activation='relu')(inputs)
    x = Dropout(rate)(x)

    # 残差连接
    x = Add()([x, res_conv])

    # ------------------- 第二残差块（BiLSTM） ---------------------
    # 双向LSTM
    bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    bilstm = Dropout(rate)(bilstm)

    # 维度匹配
    if bilstm.shape[-1] != x.shape[-1]:
        res_bilstm = Conv1D(bilstm.shape[-1], 1, padding='same')(x)
    else:
        res_bilstm = x
    bilstm = Add()([bilstm, res_bilstm])

    # ------------------- 注意力机制 ---------------------
    # 通道注意力
    avg_pool = GlobalAveragePooling1D()(bilstm)
    avg_pool = Reshape((1, avg_pool.shape[-1]))(avg_pool)
    avg_pool = Dense(lstm_units * 2, activation='sigmoid')(avg_pool)  # BiLSTM输出维度是units*2

    # 空间注意力
    max_pool = tf.reduce_max(bilstm, axis=-1, keepdims=True)
    attention = Multiply()([avg_pool, max_pool])

    # 应用注意力
    weighted = Multiply()([bilstm, attention])

    # ------------------- 输出层 ---------------------
    flat = Flatten()(weighted)
    outputs = Dense(output_length * embed_dim)(flat)
    outputs = Reshape((output_length, embed_dim))(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# ------------------- 模型初始化 ---------------------
model = build_model(input_length, output_length, embed_dim)
model.compile(optimizer='adam', loss='mse')

# ------------------- 数据准备 ---------------------
# 生成序列
X, y, time_sequences = create_sequences(features_scaled, time, input_length, output_length)

# 数据分割
total_size = len(X)
train_size = int(total_size * 0.7)
val_size = int(total_size * 0.2)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]
time_train, time_val, time_test = time_sequences[:train_size], time_sequences[
                                                               train_size:train_size + val_size], time_sequences[
                                                                                                  train_size + val_size:]

# 数据增强（仅在训练集）
X_train_aug, y_train_aug, time_train_aug = create_sequences(features_scaled[:train_size + input_length + output_length],
                                                            time[:train_size + input_length + output_length],
                                                            input_length, output_length, augment=True)
X_train = np.concatenate([X_train, X_train_aug], axis=0)
y_train = np.concatenate([y_train, y_train_aug], axis=0)
time_train = np.concatenate([time_train, time_train_aug], axis=0)

# ------------------- 模型训练 ---------------------
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop],
                    verbose=1)


# ------------------- 预测与评估 ---------------------
def inverse_scale(y_pred, scalers, features):
    rescaled = np.zeros_like(y_pred)
    for i, col in enumerate(features.columns):
        rescaled[:, :, i] = scalers[col].inverse_transform(y_pred[:, :, i])
    return rescaled


# 预测
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

# 反归一化
train_pred_rescaled = np.zeros_like(train_pred.reshape(-1, embed_dim))
val_pred_rescaled = np.zeros_like(val_pred.reshape(-1, embed_dim))
test_pred_rescaled = np.zeros_like(test_pred.reshape(-1, embed_dim))
y_train_rescaled = np.zeros_like(y_train.reshape(-1, embed_dim))
y_val_rescaled = np.zeros_like(y_val.reshape(-1, embed_dim))
y_test_rescaled = np.zeros_like(y_test.reshape(-1, embed_dim))

for i, col in enumerate(features_scaled.columns):
    train_pred_rescaled[:, i] = scalers[col].inverse_transform(train_pred.reshape(-1, embed_dim)[:, i].reshape(-1, 1)).flatten()
    val_pred_rescaled[:, i] = scalers[col].inverse_transform(val_pred.reshape(-1, embed_dim)[:, i].reshape(-1, 1)).flatten()
    test_pred_rescaled[:, i] = scalers[col].inverse_transform(test_pred.reshape(-1, embed_dim)[:, i].reshape(-1, 1)).flatten()
    y_train_rescaled[:, i] = scalers[col].inverse_transform(y_train.reshape(-1, embed_dim)[:, i].reshape(-1, 1)).flatten()
    y_val_rescaled[:, i] = scalers[col].inverse_transform(y_val.reshape(-1, embed_dim)[:, i].reshape(-1, 1)).flatten()
    y_test_rescaled[:, i] = scalers[col].inverse_transform(y_test.reshape(-1, embed_dim)[:, i].reshape(-1, 1)).flatten()


# 评价指标
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nse_score = nse(y_pred.flatten(), y_true.flatten())
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, nse_score, r2


# 构建评价指标数据框
column_names = ['WT', 'pH', 'DO', 'EC', 'NTU', 'PPI', 'NH3-N', 'TN', 'TP']  # 修改列名


def create_metrics_df(y_true, y_pred):
    metrics = {
        '指标': ['RMSE', 'MAE', 'NSE', 'R2']
    }
    for col in column_names:
        rmse_col, mae_col, nse_col, r2_col = evaluate_model(y_true[:, features.columns.get_loc(col)],
                                                          y_pred[:, features.columns.get_loc(col)])
        metrics[col] = [rmse_col, mae_col, nse_col, r2_col]
    return pd.DataFrame(metrics).round(3)


train_metrics_df = create_metrics_df(y_train_rescaled, train_pred_rescaled)
val_metrics_df = create_metrics_df(y_val_rescaled, val_pred_rescaled)
test_metrics_df = create_metrics_df(y_test_rescaled, test_pred_rescaled)


# 创建保存结果的文件夹
result_folder = f'result_{output_length}'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# 保存模型
model.save(os.path.join(result_folder, 'trained_model.h5'))


# 保存训练集，验证集和测试集的结果到csv文件
def save_results_to_csv(y_true, y_pred, time_data, metrics_df, base_filename):
    # 保存每个指标的数据
    for col in column_names:
        col_idx = features.columns.get_loc(col)
        df = pd.DataFrame({
            'Time': time_data.flatten(),
            'True': y_true[:, col_idx],
            'Pred': y_pred[:, col_idx]
        }).sort_values(by='Time', ascending=False)

        df.to_csv(f"{base_filename}_{col}.csv", index=False)

    # 保存评价指标
    #metrics_df.to_csv(f"{base_filename}_metrics.csv", index=False)


print("训练集Time范围：", time_train.min(), "到", time_train.max())
print("验证集Time范围：", time_val.min(), "到", time_val.max())
print("测试集Time范围：", time_test.min(), "到", time_test.max())

# 生成基础文件名（不带扩展名）
train_base = os.path.join(result_folder, f'train_results_{input_length}+{output_length}')
val_base = os.path.join(result_folder, f'val_results_{input_length}+{output_length}')
test_base = os.path.join(result_folder, f'test_results_{input_length}+{output_length}')

# 保存结果
save_results_to_csv(y_train_rescaled, train_pred_rescaled, time_train, train_metrics_df, train_base)
save_results_to_csv(y_val_rescaled, val_pred_rescaled, time_val, val_metrics_df, val_base)
save_results_to_csv(y_test_rescaled, test_pred_rescaled, time_test, test_metrics_df, test_base)
