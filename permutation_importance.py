import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.inspection import permutation_importance
import seaborn as sns
# import eli5
# from eli5.sklearn import PermutationImportance
np.random.seed(1)



def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    print(time_stamp)
def show(y_true, y_pred,title):
    classes=[1,2,3,4,5,6]
    C = confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5,6])
    plt.matshow(C, cmap=plt.cm.Greens)
    plt.colorbar()
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45)
    plt.yticks(tick_marks, classes)
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[i, j], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.show()

def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 10
    ca_config["estimators"] = []
    ca_config["estimators"].append(
            {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10,
             "max_depth": 5,"objective": "multi:softprob", "silent":
            True, "nthread": -1, "learning_rate": 0.1} )
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier",
            "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier",
            "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


# 1.获取原数据
# columns_names = ['1','2','3','4','5','6','7']
# data = pd.read_csv("E:/LabelData.csv",names=columns_names)
# print(type(data))

# feature_names = ['x', 'y', 'z', 'Angle', 'Score1', 'Score2', 'ActivityID']
# columns_names = ['x', 'y', 'z', 'Angle', 'Score1', 'Score2', 'ActivityID']
columns_names = ['1','2','3','4','5','6','7']
data = pd.read_csv("D:/LabelData.csv", names=columns_names)
# data = pd.DataFrame(data, columns=feature_names)
print(data)


# 2. 分割数据集为训练集、测试集
X_train,X_test,y_train,y_test = train_test_split(data[columns_names[0:6]], data[columns_names[6]], test_size=0.25, random_state=33)
print(type(X_train))

conf_mat = np.zeros([4,4])
# 3.数据标准化、归一化（先进行训练集和测试集的划分，再进行数据预处理）
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 4.定义模型，进行分类
RC1 = RandomForestClassifier()
get_time_stamp()
RC1.fit(X_train,y_train)
get_time_stamp()
#
# Permutation Importance mean
result = permutation_importance(RC1, X_test, y_test, n_repeats=10,random_state=42)
print(result)
featImp = pd.DataFrame()
featImp['Feat'] = ['x', 'y', 'z', 'Angle', 'Score1', 'Score2']
print(featImp['Feat'])
featImp['Permutation Importance mean'] = result.importances_mean
featImp = featImp.sort_values('Permutation Importance mean',ascending = False)
#
plt.figure(figsize=[20,10])
# 设置order参数：按重要程度（importance）从大到小输出的结果:
sns.barplot(x = 'Permutation Importance mean', y = 'Feat',data = featImp[:20],orient='h')
plt.xlabel('Permutation Importance mean', fontsize=15)
plt.ylabel('Feat', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
#
# Permutation Importance std
result = permutation_importance(RC1, X_test, y_test, n_repeats=10,random_state=42)
print(result)
featImp = pd.DataFrame()
featImp['Feat'] = ['x', 'y', 'z', 'Angle', 'Score1', 'Score2']
print(featImp['Feat'])
featImp['Permutation Importance std'] = result.importances_std
featImp = featImp.sort_values('Permutation Importance std',ascending = False)
#
plt.figure(figsize=[20,10])
# 设置order参数：按重要程度（importance）从大到小输出的结果:
sns.barplot(x = 'Permutation Importance std', y = 'Feat',data = featImp[:20],orient='h')
plt.xlabel('Permutation Importance std', fontsize=15)
plt.ylabel('Feat', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# Permutation Importances
result = permutation_importance(RC1, X_test, y_test, n_repeats=10,random_state=42)
print(result)
featImp = pd.DataFrame()
featImp['Feat'] = ['x', 'y', 'z', 'Angle', 'Score1', 'Score2']
print(featImp['Feat'])
featImp['Permutation Importances'] = result.importances
featImp = featImp.sort_values('Permutation Importances',ascending = False)
#
plt.figure(figsize=[20,10])
# 设置order参数：按重要程度（importance）从大到小输出的结果:
sns.barplot(x = 'Permutation Importances', y = 'Feat',data = featImp[:20],orient='h')

plt.show()

# Permutation Feature Importance-eli5:
# perm = PermutationImportance(RC1, n_iter=10)
# perm.fit(X_train, y_train)
# eli5.show_weights(perm)
# 实例化
# perm = PermutationImportance(RC1, random_state=1).fit(X_train, y_train)
# eli5.show_weights(perm)

# # 5.评价模型（score方法）
# print('Accuarcy of forest Classifier:',RC1.score(X_test,y_test))
# print('RF classification_report')
# print(classification_report(y_test,RC1.predict(X_test),target_names=['1','2','3','4','5','6']))
# predict = RC1.predict(X_test)
# plt.figure(figsize=(24, 16), dpi=60)
# show(y_test,predict,'RF_matrix')
#
