import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import shap
import time
import fasttreeshap
np.random.seed(1)

# rather than use the whole training set to estimate expected values, we could summarize with
# a set of weighted kmeans, each weighted by the number of points they represent. But this dataset
# is so small we don't worry about it
#X_train_summary = shap.kmeans(X_train, 50)

def print_accuracy(f):
    print("Accuracy = {0}%".format(100*np.sum(f(X_test) == Y_test)/len(Y_test)))
    time.sleep(0.5) # to let the print get out before any progress bars

shap.initjs()



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler



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
#columns_names=['x轴加速度','y轴加速度','z轴加速度','pitch','roll','yaw','温度','状态']
columns_names = ['x','y','z','Angle','ActivityID','Score1','Score2']
# print(len(columns_names))
data=pd.read_csv("LabelData.csv", dtype={'x': np.int64, 'y': np.int64, 'z': np.int64, 'Angle': np.int64,
                                                              'ActivityID': np.int64, 'Score1': np.int64, 'Score2': np.int64})


# 2.分割数据集为训练集、测试集
X_train,X_test,y_train,y_test=train_test_split(data[columns_names[0:6]],data[columns_names[6]],test_size=0.25,random_state=33)
print(type(X_train))

conf_mat = np.zeros([4,4])


# 3.数据标准化、归一化（先进行训练集和测试集的划分，再进行数据预处理）
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)


# 4.定义模型，进行分类
RC1 = RandomForestClassifier(n_jobs=12)
get_time_stamp()
RC1.fit(X_train,y_train)
get_time_stamp()


# 5.评价模型（score方法）
# print('Accuarcy of forest Classifier:',RC1.score(X_test,y_test))
# print('RF classification_report')
# print(classification_report(y_test,RC1.predict(X_test),target_names=['1','2','3','4']))#,'5','6']))
# predict = RC1.predict(X_test)
# plt.figure(figsize=(24, 16), dpi=60)
# show(y_test,predict,'RF_matrix')


explainer = fasttreeshap.TreeExplainer(RC1, n_jobs=36)


shap_values = explainer.shap_values(X_test)

np.save('shap_values.npy', np.array(shap_values))

# w = shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
# shap.save_html(f"shap_pd_single%d.html" % 0, w, False)

# Load SHAP value from file
# shap_values = list(np.load('shap_values.npy'))

# Plot shap
# shap.summary_plot(shap_values, X_test, show=True)



#
# plt.show()
# plt.close()

# explain all the predictions in the test set
# explainer = shap.TreeExplainer(RC1)
# # explainer = fasttreeshap.TreeExplainer(RC1, data=X_train, n_jobs=24)
#
# # X_test = shap.sample(X_test, nsamples=10)
# shap_values = explainer.shap_values(X_test)
# w = shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, show=False)
# shap.save_html(f"shap_pd_single%d.html" % 0, w, False)
#
#
# # explain all the predictions in the test set
# # explainer = shap.KernelExplainer(RC1.predict_proba, shap.sample(X_train, nsamples=10000))
# # shap_values = explainer.shap_values(shap.sample(X_test, nsamples=100))
# shap.summary_plot(shap_values, X_test, show=True, plot_size=(24, 12))
