from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
dataSet = load_iris()
data = dataSet['data']
label = dataSet['target']
feature = dataSet['feature_names']
target = dataSet['target_names']
df = pd.DataFrame(np.column_stack((data, label)), columns= np.append(feature, 'label'))
StandardScaler().fit_transform(data)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
ss = ShuffleSplit(n_splits = 1,test_size= 0.2)
for tr,te in ss.split(data,label):
    xr = data[tr]
    xe = data[te]
    yr = label[tr]
    ye = label[te]
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    clf.fit(xr, yr)
    predict = clf.predict(xe)
    print(classification_report(ye, predict))