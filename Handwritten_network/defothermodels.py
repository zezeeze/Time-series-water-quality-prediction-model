import tensorflow as tf
import os
import random
import numpy as np
from Handwritten_network.tfts import AutoConfig, AutoModel, KerasTrainer,Trainer
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)

def run_train(use_model,learning_rate,epochs,predict_length,train,valid):
    set_seed(315)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # for strong seasonality data like sine or air passengers, set up skip_connect_circle True
    custom_params = AutoConfig(use_model).get_config()
    custom_params.update({"skip_connect_circle": True})

    model = AutoModel(use_model, predict_length=predict_length)
    trainer = KerasTrainer(model, optimizer=optimizer, loss_fn=loss_fn)
    history = trainer.train(train, valid, n_epochs=epochs, early_stopping=EarlyStopping("val_loss", patience=5)) #5次训练误差没有降低，就停止训练，也可以改为None

    pred1 = trainer.predict(train[0])
    pred2 = trainer.predict(valid[0])


    xunlian_predicted_data = np.array(pred1)
    ceshi_predicted_data = np.array(pred2)

    # 绘制历史数据
    plt.plot(history.history['loss'], label='train')
    # 绘制训练过程中的损失曲线。
    # history.history['loss']获取训练集上每个epoch的损失值。
    # 'label='train''设置该曲线的标签为'train'。

    plt.plot(history.history['val_loss'], label='test')
    # 绘制验证过程中的损失曲线。
    # history.history['val_loss']获取验证集上每个epoch的损失值。
    # 'label='test''设置该曲线的标签为'test'。
    plt.legend()
    # 显示图例，方便识别每条曲线代表的数据集。
    plt.show()
    # 展示绘制的图像。
    return trainer,xunlian_predicted_data, ceshi_predicted_data


