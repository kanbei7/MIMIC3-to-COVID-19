import pandas as pd
import numpy as np
import os
import itertools
import math
import datetime as dt
from scipy import stats
import pickle as pk

datadir = os.path.join('data','mimic3')
flst = [x for x in os.listdir(datadir) if x.endswith('_timeseries.csv')]

vital_feat = ['Sys_BP','Dia_BP','Heart_Rate','Respirations','SPO2','Temperature']
mimic_vital_feat = ['Systolic blood pressure','Diastolic blood pressure','Heart Rate',\
'Respiratory rate','Oxygen saturation','Temperature']
MEAN_IMPUTE_ICU = {
	'Systolic blood pressure':120.36,
	'Diastolic blood pressure':69.70,
	'Heart Rate':89.63,
	'Respiratory rate':25.90,
	'Oxygen saturation':93.21,
	'Temperature':98.66
}

Daily_Vitals_Feat_col = ['patientid','day','label'] \
						+ ['_'.join(t) for t in itertools.product(vital_feat, ['slope', 'r2', 'mean', 'max', 'min','mask'])]


def fit(t,x):
	slope, intercept, r_value, p_value, std_err = stats.linregress(t,x)
	return slope, r_value**2

#'slope', 'r2', 'mean', 'max', 'min', 'imp_mask' = 1 if (real) else 0 (imputed)
def tsFeat(data, f):
	df = data[data[f].notnull()]
	imp_mask = 1
	if len(df)==0:
		imp_mask = 0
		key = MEAN_IMPUTE_ICU[f]
		return [0, 1.0] + [key] * 3 + [imp_mask]

	t = list(df.Hours)
	x = list(df[f])
	if len(df)==1:
		return [0, 1.0, x[0], x[0], x[0], imp_mask]
	elif len(df)==2:
		slope, r2 = x[1]-x[0], 0.0
	else:
		slope, r2 = fit(t,x)
	return [slope, r2, np.mean(x), max(x), min(x), imp_mask]

def vitalFeatures(df):
	vec = []
	for f in mimic_vital_feat:
		vec.extend(tsFeat(df,f))
	return vec

#for pa per day
def createDailyVec(day, grp, pid, label):
	grp = pd.DataFrame(grp)

	# slope, sd, mean, max
	# daily feature vec
	return [pid, day, label] + vitalFeatures(grp)

mimic_daily_df = []
i =0 
for fname in flst:

	df = pd.read_csv(os.path.join(datadir, fname))
	if len(df)<10 or max(df.Hours)<48:
		continue
	df['day'] = np.ceil(df.Hours%24)
	df.day = df.day.astype(int)

	pid = fname.split('.')[0]
	label = list(df['label'])[-1]

	res_df = [createDailyVec(d, grp, pid, label) for d, grp in df.groupby('day')]
	res_df = pd.DataFrame(res_df, columns = Daily_Vitals_Feat_col)

	mimic_daily_df.append(res_df)
	i+=1
	if i%500 ==0 :
		print(i)


mimic_daily_df = pd.concat(mimic_daily_df)
mimic_daily_df.sort_values(by = ['patientid', 'day'], inplace = True)
mimic_daily_df.to_csv(os.path.join('data','mimic3_dailyvitals.csv'), index = False)