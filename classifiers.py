import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import sklearn.naive_bayes as nb
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.impute import SimpleImputer
#from GCForest import gcForest
#from gcforest.gcforest import GCForest
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import time


def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    print(time_stamp)
def show(y_true, y_pred,title):
    classes=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
    C = confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                                                 23,24,25,26,27,28,29,30,31,32,33,34,35,36,37])
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
columns_names = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 'label']
data=pd.read_excel("E:/total3_temp.xls",names=columns_names)


# 2.分割数据集为训练集、测试集
X_train,X_test,y_train,y_test=train_test_split(data[columns_names[0:27]],data[columns_names[27]],test_size=0.25,random_state=33)
print(type(X_train))

conf_mat = np.zeros([4,4])

# 3.数据标准化、归一化（先进行训练集和测试集的划分，再进行数据预处理）
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

# 4.定义模型，进行分类
RC1 = RandomForestClassifier()
get_time_stamp()
RC1.fit(X_train,y_train)
get_time_stamp()

# 5.评价模型（score方法）
print('Accuarcy of forest Classifier:',RC1.score(X_test,y_test))
print('RF classification_report')
print(classification_report(y_test,RC1.predict(X_test),target_names=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37']))
predict = RC1.predict(X_test)
show(y_test,predict,'RF_matrix')

# 尝试多种经典的ML模型
naive=nb.GaussianNB()
naive.fit(X_train, y_train)
predict= naive.predict(X_test)
show(y_test,predict,'naive_matrix')
print('Accuarcy of bayes Classifier:',naive.score(X_test,y_test))
print('bayes classification_report')
print(classification_report(y_test,naive.predict(X_test),target_names=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37']))
lb = LabelBinarizer()
clf = KNeighborsClassifier(n_neighbors=7)
get_time_stamp()
clf.fit(X_train, y_train)
get_time_stamp()
clfpre=clf.predict(X_test)
pkl_filename1 = "knn.pkl"
with open(pkl_filename1, 'wb') as file:
     pickle.dump(clf, file)
# print('Recall: %s' % recall_score(X_test, clfpre))
print('Accuarcy of knn Classifier:',clf.score(X_test,y_test))
print('KNN classification_report')
print(classification_report(y_test,clf.predict(X_test),target_names=['1','2','3','4','5','6','7','8','9','10','11','12',
                                                                     '13','14','15','16','17','18','19','20',
                                                                     '21','22','23','24','25','26','27','28','29',
                                                                     '30','31','32','33','34','35','36','37']))
show(y_test,clfpre,'knn_matrix')

lr=LogisticRegression()

sgdc=SGDClassifier(loss="hinge", penalty="l2")
lr.fit(X_train,y_train)

show(y_test,RC1.predict(X_test),'RF_matrix')
sgdc.fit(X_train,y_train)
predict=sgdc_y_predict=sgdc.predict(X_test)
show(y_test,predict,'SGD_matrix')
predict=lr.predict(X_test)
show(y_test,predict,'lr_matrix')
print(predict)
print('Accuarcy of SGD Classifier:',sgdc.score(X_test,y_test))
print('SGD classification_report')
print(classification_report(y_test,sgdc_y_predict,target_names=['1','2','3','4','5','6','7','8','9','10','11','12','13',
                                                                '14','15','16','17','18','19','20','21',
                                                                '22','23','24','25','26','27','28','29',
                                                                '30','31','32','33','34','35','36','37']))

print('Accuracy of LR Classifier:', lr.score(X_test,y_test))
print('LR classification_report')
print(classification_report(y_test,lr.predict(X_test),target_names=['1','2','3','4','5','6','7','8','9','10','11','12',
                                                                    '13','14','15','16','17','18','19','20','21',
                                                                    '22','23','24','25','26','27','28','29',
                                                                    '30','31','32','33','34','35','36','37']))
sv=svm.SVC(C=2,kernel='rbf',gamma=5,decision_function_shape='ovr')
sv.fit(X_train,y_train)
predict=sv_y_predict=sv.predict(X_test)
show(y_test,predict,'SVM_matrix')
print('Accuarcy of svm Classifier:',sv.score(X_test,y_test))
print('SVM classification_report')
print(classification_report(y_test,sv_y_predict,target_names=['1','2','3','4','5','6','7','8','9','10','11','12','13',
                                                              '14','15','16','17','18','19','20','21',
                                                              '22','23','24','25','26','27','28','29',
                                                              '30','31','32','33','34','35','36','37']))
# print(X_train)
# # print(y_train)
# # print(X_test)
# # print(sgdc_y_predict)
# # print(lr_y_predict)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_test=np.array(y_test)
y_train=np.array(y_train)
print(X_train.shape)
print(X_test.shape)
print(y_train)
print(y_test.shape)
model = gcForest(shape_1X=[1,7], window=[7], tolerance=0.0)
model.fit(X_train, y_train)
model_y_predict=model.predict(X_test)
accuarcy = accuracy_score(y_true=y_test, y_pred=model_y_predict)
print('gcForest accuarcy : {}'.format(accuarcy))

config=get_toy_config()
gc = gcForest(config)
#X_train_enc是每个模型最后一层输出的结果，每一个类别的可能性
X_train_enc = gc.fit_transform(X_train, y_train)
y_pred = gc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy of Other classifier using gcforest's X_encode = {:.2f} %".format(acc * 100))
clf = RandomForestClassifier(n_estimators=1000, max_depth=None, n_jobs=-1)
clf.fit(X_train_enc, y_train)
y_pred = clf.predict(X_test)
acc1 = accuracy_score(y_test, y_pred)
print("Test Accuracy of Other classifier using gcforest's X_encode = {:.2f} %".format(acc1 * 100))