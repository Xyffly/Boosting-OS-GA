# phase 3 : 合成新样本
import numpy as np
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing

random_seed = 42
train_ratio = 0.7

class MAHAKIL(object):
    def __init__(self, pfp=0.5):
        self.data_t = None  # 保存初始时的缺陷样本
        self.pfp = pfp  # 预期缺陷样本占比
        self.T = 0  # 需要生成的缺陷样本数
        self.new = []  # 存放新生成的样本

    # 核心方法
    # return : data_new, label_new
    def fit_sample(self, data, label):
        # data : 包含度量信息的样本 数组
        # label : 样本的标签 数组
        data_t, data_f, label_t, label_f = [], [], [], []
        # 按照正例和反例划分数据集
        for i in range(label.shape[0]):
            if label[i] == 1:
                data_t.append(data[i])
                label_t.append(label[i])
            if label[i] == 0:
                data_f.append(data[i])
                label_f.append(label[i])
        self.T = len(data_f) / (1 - self.pfp) - len(data_f)
        self.data_t = np.array(data_t)
        # 计算得到马氏距离
        d = self.mahalanobis_distance(self.data_t)
        # 降序排序
        d.sort(key=lambda x: x[1], reverse=True)
        # 将正例集一分为二
        k = len(d)
        d_index = [d[i][0] for i in range(k)]
        data_t_sorted = [data_t[i] for i in d_index]
        mid = int(k/2)
        bin1 = [data_t_sorted[i] for i in range(0, mid)]
        bin2 = [data_t_sorted[i] for i in range(mid, k)]
        # 循环迭代生成新样本
        l_ = len(bin1)
        mark = [1, 3, 7, 15, 31, 63,127]
        p = self.T / l_
        is_full = True
        g = mark.index([m for m in mark if m > p][0]) + 1
        cluster = 2 ** (g - 1)  # 最后一代的子代个数
        if (self.T - mark[g-2]*l_) < cluster:
            # 说明多增加一代，还不如保持少几个的状态
            is_full = False
            g -= 1
            k = 0
        else:
            k = l_ - round((self.T - mark[g-2]*l_)/cluster)
        self.generate_new_sample(bin1, bin2, g, l_, k, is_full)
        # 返回数据与标签
        label_new = np.ones(len(self.new))
        print(data.shape)
        print(self.new)
        new1 = np.array(self.new)
        print(new1.shape)
        return np.append(data, self.new, axis=0), np.append(label, label_new, axis=0)

    def mahalanobis_distance(self, x):
        # x : 数组
        mu = np.mean(x, axis=0)  # 均值
        d = []
        for i in range(x.shape[0]):
            x_mu = np.atleast_2d(x[i] - mu)
            s = self.cov(x)
            d_squre = np.dot(np.dot(x_mu, np.linalg.inv(s)), np.transpose(x_mu))[0][0]
            d_tuple = (i, d_squre)
            d.append(d_tuple)
        return d

    @staticmethod
    def cov(x):
        # x : 数组
        s = np.zeros((x.shape[1], x.shape[1]))
        mu = np.mean(x, axis=0)  # 均值
        for i in range(x.shape[0]):
            x_xbr = np.atleast_2d(x - mu)
            s_i = np.dot(np.transpose(x_xbr), x_xbr)
            s = s + s_i
        return np.divide(s, x.shape[0])

    # 生成新样本
    def generate_new_sample(self, bin1, bin2, g, l, k, is_full):
        # bin1, bin2 是数组
        # g 遗传的剩余代数
        # l bin1的item数目
        # k 最后一代每个节点需要裁剪的个数
        # is_full 是否溢出，也即最后一代算完，是超出了T，还是未满T
        assert len(bin1) <= len(bin2)
        if g >= 2 or (g == 1 and is_full is False):
            lv_0 = []  # 子代
            for i in range(l):
                # 生成子代
                lv_0.append(np.mean(np.append(np.atleast_2d(bin1[i]), np.atleast_2d(bin2[i]), axis=0), axis=0))
            self.new.extend(lv_0)
            self.generate_new_sample(lv_0, bin1, g-1, l, k, is_full)
            self.generate_new_sample(lv_0, bin2, g-1, l, k, is_full)
        if g == 1 and is_full:
            lv_0 = []  # 子代
            for i in range(l):
                # 生成子代
                lv_0.append(np.mean(np.append(np.atleast_2d(bin1[i]), np.atleast_2d(bin2[i]), axis=0), axis=0))
            del lv_0[-1: (-k-1): -1]
            self.new.extend(lv_0)

data_files = pd.read_csv("C:/Users/Administrator.DESKTOP-G2KEBOV/Desktop/岩相分类实验数据/10and4.csv")
#设定为函数
def split_data(dataframe, train_ratio, random_state):
    '''
    将原始数据中每类数据都按照train_ratio的比例划分测试集和训练集
    参数
    dataframe：所读取的原始数据
    train_ratio:训练集所占比例
    random_state：打乱数据的随机数
    '''
    # 获取所有的类别
    lst = dataframe["class"].unique()
    # 创建固定维度的空数组以便拼接
    column_num = dataframe.shape[1]
    data_train = np.empty(shape=(0, column_num))
    data_test = np.empty(shape=(0, column_num))
    #循环对每类数据进行比例划分
    for i in lst:
        data_class = dataframe[dataframe["class"] == i]  # 筛选相同类的数据
        data_class = np.array(data_class)
        df = shuffle(data_class, random_state=random_state)  # 打乱次序
        train_size = int(len(df) * train_ratio)
        train = df[0:train_size, :]
        test = df[train_size:, :]
        data_train = np.concatenate((data_train, train), axis=0)
        data_test = np.concatenate((data_test, test), axis=0)
    return data_train, data_test


# data_train, data_test = split_data(data_files,train_ratio=train_ratio,random_state=random_seed)

data_files = np.array(data_files)
#划分训练集和测试集的X，Y
X_train = data_files[:,1:]
Y_train = data_files[:,0]


X, Y= make_classification(n_classes=2, class_sep=2,
                           weights=[0.8, 0.2], n_informative=3,
                           n_redundant=1, flip_y=0,
                           n_features=5, n_clusters_per_class=1,
                           n_samples=10)
mahakil = MAHAKIL()

print(Y)
X_resampled, y_resampled = mahakil.fit_sample(X_train, Y_train)
#将预测结果输出到Excel
X_result= pd.DataFrame(X_resampled)
Y_result= pd.DataFrame(y_resampled)
writer = pd.ExcelWriter('C:/Users/Administrator.DESKTOP-G2KEBOV/Desktop/10MAHAKIL.xlsx')
X_result.to_excel(writer,'sheet_1',float_format='%.3f')
Y_result.to_excel(writer,'sheet_2',float_format='%.3f')
writer.save()
writer.close()
'''
#数据归一化
preprocessor = preprocessing.MinMaxScaler()
preprocessor.fit(np.concatenate((X_train,X_train),axis=0))
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)
print(X_train.shape)
print(Y_train.shape)
print(Y_train)


mahakil = MAHAKIL()
X_resampled, y_resampled = mahakil.fit_sample(X_train, Y_train)
#X_resampled, y_resampled = mahakil.fit_sample(X, Y)
'''