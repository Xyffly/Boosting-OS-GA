#分类
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import catboost as cb
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
import matplotlib.pyplot as plt

random_seed = 42
train_ratio = 0.7
#读取数据并将数据转换为array格式
data_files = []
data_files = pd.read_csv("C:/Users/Administrator.DESKTOP-G2KEBOV/Desktop/岩相分类实验数据/4features949original data.csv")
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

# 调参，用网格搜索调出最优参数
params = {'depth': [4],
          'learning_rate': [0.15],
          'l2_leaf_reg': [1],
          'iterations': [300]}
cb = cb.CatBoostClassifier()
cb_model = GridSearchCV(cb, params, scoring="accuracy", cv=3)
cb_model.fit(X_train, Y_train)
# 查看最佳分数
print(cb_model.best_score_)  # 0.7088001891107445
# 查看最佳参数
print(cb_model.best_params_) #{'depth': 4, 'iterations': 300, 'l2_leaf_reg': 1, 'learning_rate': 0.15}

model = cb_model.best_estimator_

Y_pred = model.predict(X_test)
Y_pred_score = model.predict_proba(X_test)
auc1 = roc_auc_score(Y_test,Y_pred_score,multi_class='ovr')
print("auc:",auc1)
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
