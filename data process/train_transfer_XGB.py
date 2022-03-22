import os
import sys
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import auc,roc_curve,average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import copy as cp
import time
import pickle as pk
from groupkfold import groupStratifiedKfold
import xgboost as xgb

label_col = sys.argv[1]
cls_w = int(sys.argv[2])


model_name = '_'.join(['mimic3XGB', label_col, str(cls_w),'.pkl'])
model_path = os.path.join('models_pheno', model_name)

datadir = 'data'
df =  pd.read_csv(os.path.join(datadir,'mimic3_dailyvitals_pheno1.csv'))
df = df.fillna(0)
resdir = 'res'
id_col = 'patientid'

FEATURES = ['Sys_BP_slope',
			'Sys_BP_r2',
			'Sys_BP_mean',
			'Sys_BP_max',
			'Sys_BP_min',
			'Sys_BP_mask',
			'Dia_BP_slope',
			'Dia_BP_r2',
			'Dia_BP_mean',
			'Dia_BP_max',
			'Dia_BP_min',
			'Dia_BP_mask',
			'Heart_Rate_slope',
			'Heart_Rate_r2',
			'Heart_Rate_mean',
			'Heart_Rate_max',
			'Heart_Rate_min',
			'Heart_Rate_mask',
			'Respirations_slope',
			'Respirations_r2',
			'Respirations_mean',
			'Respirations_max',
			'Respirations_min',
			'Respirations_mask',
			'SPO2_slope',
			'SPO2_r2',
			'SPO2_mean',
			'SPO2_max',
			'SPO2_min',
			'SPO2_mask',
			'Temperature_slope',
			'Temperature_r2',
			'Temperature_mean',
			'Temperature_max',
			'Temperature_min',
			'Temperature_mask'
			]


X_all = df[FEATURES]
y_all = list(df[label_col])

model = xgb.XGBClassifier(
		max_depth = 2,
		n_jobs = 6,
		learning_rate = 0.2,
		reg_alpha = 0.0,
		reg_lambda = 0.01,
		objective = 'binary:logistic',
		eval_metric = 'aucpr',
		subsample = 1.0,
		use_label_encoder = False,
		scale_pos_weight = cls_w
		)	

model.fit(X_all,y_all)
pk.dump(model, open(model_path,'wb'))
print('model saved.')

