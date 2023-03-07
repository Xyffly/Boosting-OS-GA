import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE

file='C:/Users/Administrator.DESKTOP-G2KEBOV/Desktop/SMOTE数据/13and4.csv'
data = pd.read_csv(file,encoding='gb18030')
data=data.values
print(data.shape)
Xx=data[:,1:]
print(Xx.shape)
yy=data[:,0]

print('Original dataset shape %s' % Counter(yy))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(Xx,yy)
print('Original dataset shape %s' % Counter(y_res))

data_resampled2 = pd.DataFrame(X_res, y_res)
print(data_resampled2.shape)
writer = pd.ExcelWriter('C:/Users/Administrator.DESKTOP-G2KEBOV/Desktop/13class new data.xlsx')#创建数据存放路径
data_resampled2.to_excel(writer,'sheet_1',float_format='%.3f')
writer.save()
writer.close()