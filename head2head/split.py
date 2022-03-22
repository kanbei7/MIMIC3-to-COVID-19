import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import auc,roc_curve,average_precision_score, confusion_matrix
from groupkfold import groupKfold,groupStratifiedKfold
import copy as cp
import time
from random import shuffle,sample

id_col = 'patientid'
data_file = 'h2hdata.csv'
datadir = 'data'
label_col = 'label_3days'
month_idx = int(sys.argv[1])
month_end_dates = [
	#'2020-07-31',
	#'2020-08-31',
	'2020-09-30',
	'2020-10-31',
	'2020-11-30',
	'2020-12-31',
	'2021-01-31',
	'2020-08-31'			
	]


df = pd.read_csv(os.path.join(datadir, data_file))
df = df.fillna(0)
df = df[df.date<= month_end_dates[month_idx]]
#print(np.min(df.date))
#print(len(df[df[label_col]>0]))
train_kfold,test_kfold = groupStratifiedKfold(df, id_col, label_col, kfold = 3, n_bins = 3)
df_train = df[df[id_col].isin(train_kfold[0])]
df_test = df[df[id_col].isin(test_kfold[0])]

#print(len(df_train[df_train[label_col]>0]))
#print(len(df_test[df_test[label_col]>0]))

df_train.to_csv(os.path.join('data','train.csv') , index = False)
df_test.to_csv(os.path.join('data','test.csv') , index = False)
#print('data loaded.')