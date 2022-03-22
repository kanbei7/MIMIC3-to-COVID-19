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
import xgboost as xgb

datadir = 'data'
modeldir = 'models_pheno'
data = pd.read_csv(os.path.join(datadir, 'dataset1_vitals_Jun2020toFeb2021.csv'))
data = data.drop(columns=['index'])

FEATURES = ['sys_bp_slope',
			'sys_bp_r2',
			'sys_bp_mean',
			'sys_bp_max',
			'sys_bp_min',
			'sys_bp_mask',
			'dia_bp_slope',
			'dia_bp_r2',
			'dia_bp_mean',
			'dia_bp_max',
			'dia_bp_min',
			'dia_bp_mask',
			'heart_rate_slope',
			'heart_rate_r2',
			'heart_rate_mean',
			'heart_rate_max',
			'heart_rate_min',
			'heart_rate_mask',
			'respirations_slope',
			'respirations_r2',
			'respirations_mean',
			'respirations_max',
			'respirations_min',
			'respirations_mask',
			'spo2_slope',
			'spo2_r2',
			'spo2_mean',
			'spo2_max',
			'spo2_min',
			'spo2_mask',
			'temperature_slope',
			'temperature_r2',
			'temperature_mean',
			'temperature_max',
			'temperature_min',
			'temperature_mask'
			]


for model_name in os.listdir(modeldir):
	feat_name = model_name.split('.')[0]+'score'
	clf = pk.load(open(os.path.join(modeldir, model_name), 'rb'))
	scores = clf.predict_proba(data[FEATURES])
	data[feat_name] = scores[:,1]

data.to_csv(os.path.join(datadir, 'NewDataset1_vitals_mimic_scores_Jun2020toFeb2021_pheno1.csv'), index = False)

