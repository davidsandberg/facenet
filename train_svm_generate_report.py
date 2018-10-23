import pickle
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import os
import numpy as np
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy import interp
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder as LE

def classification_report_csv(report, csv_file_name):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(csv_file_name, index = False)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

training_embeddings = np.load('CCNA_full_training_embeddings_facenet_model_v1.npy')
training_image_names = np.load('ccna_full_training_image_names_list.npy')
training_labels = np.asarray([image_name.split('/')[-2] for image_name in training_image_names])
training_sku_id_set = list(set(training_labels))
#y_train_biary = label_binarize(np.asarray(training_labels), classes = training_sku_id_set)
y_train = np.asarray(training_labels)
X_train = training_embeddings

testing_embeddings = np.load('CCNA_full_testing_embeddings_facenet_model_v1.npy')
testing_image_names = np.load('ccna_full_testing_image_names_list.npy')
testing_labels = np.asarray([image_name.split('/')[-2] for image_name in testing_image_names])
testing_sku_id_set = list(set(testing_labels))
#y_test_binary = label_binarize(np.asarray(testing_labels), classes = testing_sku_id_set)
y_test = np.asarray(testing_labels)
X_test = testing_embeddings

le = LE()
le = le.fit(np.concatenate((y_test, y_train)))
#print (len(le.classes_))
y_train = le.transform(y_train)
y_test = le.transform(y_test)

clf = LinearSVC(C=50,class_weight='balanced',max_iter=100000)
clf.fit(X_train, y_train)
save_obj(clf, 'LinearSVC_C_50_cls_wt_bal_max_it_100000.pkl')
save_obj(le, 'CCNA_le.pkl')
#y_score = clf.decision_function(X_test)
print ('LinearSVC Score: ', clf.score(X_test, y_test))
predictions = clf.predict(X_test)
report = classification_report(y_test, predictions)
classification_report_csv(report, 'facenet_v1_linear_svc_classification_report.csv')
