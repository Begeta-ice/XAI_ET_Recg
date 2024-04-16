# 查看随机森林的特征重要性
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
np.random.seed(1)


def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    print(time_stamp)


def show(y_true, y_pred, title):
    classes = [1, 2, 3, 4, 5, 6]
    C = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5, 6])
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
         "max_depth": 5, "objective": "multi:softprob", "silent":
             True, "nthread": -1, "learning_rate": 0.1})
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier",
                                    "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier",
                                    "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


columns_names = ['1', '2', '3', '4', '5', '6', '7']
data = pd.read_csv("D:/LabelData.csv", names=columns_names)

X_train,X_test,y_train,y_test = train_test_split(data[columns_names[0:6]], data[columns_names[6]], test_size=0.25, random_state=33)
print(type(X_train))
conf_mat = np.zeros([4,4])

# features = data[columns_names[0:6]]
# target = data[columns_names[6]]
# 训练模型
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# 创建分类器对象
RC1 = RandomForestClassifier()
get_time_stamp()
RC1.fit(X_train,y_train)
get_time_stamp()

# 计算特征重要性
feature_importance = RC1.feature_importances_
print("model.feature_importances_: {}".format(feature_importance))

featImp = pd.DataFrame()
featImp['Feat'] = ['x', 'y', 'z', 'Angle', 'Score1', 'Score2']
print(featImp['Feat'])
featImp['Feature Importance'] = feature_importance
featImp = featImp.sort_values('Feature Importance',ascending = False)


# plt.title('Input', fontdict={'weight': 'normal', 'size': 20})  # 改变图标题字体
# plt.xlabel('Time', fontdict={'weight': 'Times New Roman', 'size': 13})  # 改变坐标轴标题字体
plt.figure(figsize=[20,10])
# 设置order参数：按重要程度（importance）从大到小输出的结果:
sns.barplot(x = 'Feature Importance', y = 'Feat',data = featImp[:20],orient='h')
plt.xlabel('Feature Importance', fontsize=15)
plt.ylabel('Feat', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# print(feature_importance)
# # 对特征重要性进行排序
# indices = np.argsort(feature_importance)
# print(indices)
# 获取特征名字
# names = [data[columns_names[0:6]][i] for i in indices]
# # 创建图
# plt.figure()
# plt.title("feature importance")
# # features.shape[1]  数组的长度
# plt.bar(range(X_train.shape[1]), importances[indices])
# plt.xticks(range(X_train.shape[1]), names, rotation=90)
# plt.show()

