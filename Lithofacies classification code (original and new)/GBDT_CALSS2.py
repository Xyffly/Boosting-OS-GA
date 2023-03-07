from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

random_seed = 42
train_ratio = 0.7
#读取数据并将数据转换为array格式
data_files = []
data_files = pd.read_csv("C:/Users/Administrator.DESKTOP-G2KEBOV/Desktop/岩相分类实验数据/4features949original data.csv")
data_files2 = pd.read_csv("C:/Users/Administrator.DESKTOP-G2KEBOV/Desktop/岩相分类实验数据/4features733new data.csv")
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


data_train, data_test = split_data(data_files,train_ratio=train_ratio,random_state=random_seed)
data_train2, data_test2 = split_data(data_files2,train_ratio=train_ratio,random_state=random_seed)

data_train= np.concatenate((data_train,data_train2),axis=0)
data_test= np.concatenate((data_test,data_test2),axis=0)

#划分训练集和测试集的X，Y
X_train = data_train[:,1:]
X_test = data_test[:,1:]
Y_train = data_train[:,0]
Y_test = data_test[:,0]

#数据归一化
preprocessor = preprocessing.MinMaxScaler()
preprocessor.fit(np.concatenate((X_train,X_train),axis=0))
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)

estimator = GradientBoostingClassifier()
param_grid = {"n_estimators": [110],
              #"max_features": ['sqrt', 'auto'],
              "learning_rate": [0.55],
              "max_depth": [2]
              }

GBDTClassifier_model = GridSearchCV(estimator, param_grid, scoring='accuracy', cv=10 ,verbose=1 ,n_jobs=-1)

GBDTClassifier_model.fit(X_train,Y_train)
print('最优参数：',GBDTClassifier_model.best_params_)
#最优参数：
model = GBDTClassifier_model.best_estimator_

#训练集结果
Y_train_pred = model.predict(X_train)
cfm = confusion_matrix(Y_train,Y_train_pred)
print("混淆矩阵:",cfm)
print("准确率:",precision_score(Y_train,Y_train_pred,average='micro'))
print("召回率:",recall_score(Y_train,Y_train_pred,average='macro'))
print("F1分数:",f1_score(Y_train,Y_train_pred, average='macro'))

#测试集结果
Y_pred = model.predict(X_test)
cfm = confusion_matrix(Y_test,Y_pred)
print("混淆矩阵:",cfm)
print("准确率:",precision_score(Y_test,Y_pred,average='micro'))
print("召回率:",recall_score(Y_test,Y_pred,average='macro'))
print("F1分数:",f1_score(Y_test,Y_pred, average='macro'))

#绘制混淆矩阵图
plt.matshow(cfm,cmap=plt.cm.gray)  #图1表示混淆矩阵
row_sums = np.sum(cfm,axis=1)
err_matrix = cfm/row_sums
np.fill_diagonal(err_matrix,0)
plt.matshow(err_matrix,cmap=plt.cm.gray)
plt.show()   #图2表示犯错误比例 按每一行算