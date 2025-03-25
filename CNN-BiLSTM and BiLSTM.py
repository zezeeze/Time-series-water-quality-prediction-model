
# In[1]:
# 调用相关库
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pandas as pd
from model import modelss
from Shared_def import plot_result,data_collation


def many_model(flag,fit_or_not,epochs,batch,learning_rate,vp_train, vp_test, vt_train, vt_test, m_out,n_out,n_in,or_dim):
    model = modelss(fit_or_not,True,epochs,batch,learning_rate,vp_train, vp_test, vt_train, vt_test, m_out,n_out,n_in,or_dim)

    xunlian_predicted_data = []
    ceshi_predicted_data = []

    if flag == 'CNN_BiLSTM':
        filters, kernel_size, Dropout, Hidden_size = 128,2,0.1,32
        xunlian_predicted_data, ceshi_predicted_data = model.run_CNN_BiLSTM(filters, kernel_size, Dropout, Hidden_size)
    if flag == 'BiLSTM':
        Dropout, Hidden_size = 0.1, 64
        xunlian_predicted_data, ceshi_predicted_data = model.run_BiLSTM(Dropout,Hidden_size)
    return xunlian_predicted_data, ceshi_predicted_data

if __name__ == '__main__':
    dataset = pd.read_excel("data\XY.xlsx")

    n_in = 12
    n_out = 1
    valuess = dataset.values[:, -1:]
    valuess = valuess.astype('float32')
    or_dim = valuess.shape[1]
    scroll_window = 1
    num_samples = 10000  # num_samples一般要设置少于数据的行数

    train_ratio = 0.8
    vp_train, vp_test, vt_train, vt_test, m_out,Ytest = data_collation(valuess, n_in, n_out, or_dim, scroll_window, num_samples,train_ratio)

    flag = 'BiLSTM' #Selection model
    epochs = 80
    batch_size = 8
    learning_rate = 1e-4
    fit_or_not = 1

    (xunlian_predicted_data, ceshi_predicted_data) = many_model(flag,fit_or_not,epochs,batch_size,learning_rate,
                           vp_train, vp_test, vt_train, vt_test, m_out,n_out,n_in,or_dim)

    plot_result(valuess,ceshi_predicted_data, Ytest, n_in,n_out,flag)



