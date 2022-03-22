import os
import sys
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import auc,roc_curve,average_precision_score, confusion_matrix
from groupkfold import groupStratifiedKfold
from sklearn.linear_model import LogisticRegression
import copy as cp
import time
import xgboost as xgb

data_file = sys.argv[1]
test_file = sys.argv[2]

id_col = 'patientid'
resdir = 'res'
log_file = 'mimicScore_XGBtest1_summary.txt'
datadir = 'data'
label_col = 'label_3days'
cls_w = 1
L2_lambda = 0.01

df_train = pd.read_csv(os.path.join(datadir, data_file))
df_train = df_train.fillna(0)

df_test = pd.read_csv(os.path.join(datadir, test_file))
df_test = df_test.fillna(0)

avg_pr = []
avg_roc = []
avg_precision = []
avg_recall = []
avg_prec_neg = []
avg_recall_neg = []
avg_f1 = []
avg_acc = []

avg_precision_20 = []
avg_recall_20 = []
avg_prec_neg_20 = []
avg_recall_neg_20 = []
avg_f1_20 = []
avg_acc_20 = []

FEATURES = ['icu_flg',
		'n_measures',
		'age_at_admission',
		'gender',
		'R1',
		'R2',
		'R3',
		'R4',
		'E1',
		'E2',
		'E3',
		'log_daySinceAdmit',
		'sys_bp_slope',
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
		'temperature_mask',
		'sofa_slope',
		'sofa_r2',
		'sofa_mean',
		'sofa_max',
		'sofa_min',
		'sofa_mask',
		] + ['mimic3XGB_selectedpheno_label_3days_3_score']	


X_train = df_train[FEATURES]
y_train = list(df_train[label_col])
X_test = df_test[FEATURES]
y_test = list(df_test[label_col])

clf = xgb.XGBClassifier(
	max_depth = 2,
	n_jobs = 7,
	learning_rate = 0.1,
	reg_alpha = 0.0,
	reg_lambda = L2_lambda,
	objective = 'binary:logistic',
	eval_metric = 'aucpr',
	subsample = 1.0,
	use_label_encoder = False,
	scale_pos_weight = cls_w
	)		
clf.fit(X_train,y_train)

#test
probas_ = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

fpr, tpr, thresh = roc_curve(y_test, probas_[:,1])
avg_roc.append(auc(fpr, tpr))
avg_pr.append(average_precision_score(y_test, probas_[:,1]))
#thresh = 0.5
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = 1.0*tp/(tp+fp)
recall = 1.0*tp/(tp+fn)
acc = 1.0*(tn+tp)/len(y_pred)
avg_precision.append(precision )
avg_recall.append(recall )
avg_prec_neg.append(1.0*tn/(tn+fn))
avg_recall_neg.append(1.0*tn/(tn+fp))
avg_f1.append(2.0*precision*recall/(precision+recall+0.00001))
avg_acc.append(acc)

#thresh = 0.2
y_pred = [int(p>=0.2) for p in probas_[:,1]]
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = 1.0*tp/(tp+fp)
recall = 1.0*tp/(tp+fn)
acc = 1.0*(tn+tp)/len(y_pred)
avg_precision_20.append(precision )
avg_recall_20.append(recall )
avg_prec_neg_20.append(1.0*tn/(tn+fn))
avg_recall_neg_20.append(1.0*tn/(tn+fp))
avg_f1_20.append(2.0*precision*recall/(precision+recall+0.00001))
avg_acc_20.append(acc)

output = pd.DataFrame(np.transpose(np.array([df_test.date, df_test.patientid, probas_[:,1], y_test])),columns = ['date','pid','y_pred','y_true'])
output.to_csv('res/MIMICXGB_3d_'+test_file,index=False)

with open(os.path.join(resdir, log_file),'a+') as f:
	f.writelines('data:'+data_file+'\n')
	f.writelines('mimicScore: '+'all2'+'\n')
	f.writelines('L2 Reg: %.4f'%L2_lambda+'\n')
	f.writelines('class weight: %d'%cls_w+'\n')
	f.writelines('label: '+label_col+'\n')

	f.writelines('TEST AUROC: %.8f'%np.mean(avg_roc)+'\n')
	f.writelines('TEST AUPR: %.8f'%np.mean(avg_pr )+'\n')

	f.writelines('TEST Prec: %.8f'%np.mean(avg_precision )+'\n')
	f.writelines('TEST Prec NEG: %.8f'%np.mean(avg_prec_neg )+'\n')
	f.writelines('TEST Recall: %.8f'%np.mean(avg_recall )+'\n')
	f.writelines('TEST Recall NEG: %.8f'%np.mean(avg_recall_neg )+'\n')
	f.writelines('TEST F1: %.8f'%np.mean(avg_f1 )+'\n')
	f.writelines('TEST ACC: %.8f'%np.mean(avg_acc)+'\n')


	f.writelines('TEST Prec 0.2: %.8f'%np.mean(avg_precision_20 )+'\n')
	f.writelines('TEST Prec NEG 0.2: %.8f'%np.mean(avg_prec_neg_20 )+'\n')
	f.writelines('TEST Recall 0.2: %.8f'%np.mean(avg_recall_20 )+'\n')
	f.writelines('TEST Recall NEG 0.2: %.8f'%np.mean(avg_recall_neg_20 )+'\n')
	f.writelines('TEST F1 0.2: %.8f'%np.mean(avg_f1_20 )+'\n')
	f.writelines('TEST ACC 0.2: %.8f'%np.mean(avg_acc_20 )+'\n')

	f.writelines('='*10+'\n')
