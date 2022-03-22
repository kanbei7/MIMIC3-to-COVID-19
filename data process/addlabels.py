import os
import pandas as pd
import sys
import numpy as np
import pickle as pk
import math
from datetime import date

datadir = 'data'
MAX_LOS = 48
'''
df1 = pd.read_csv(os.path.join(datadir, 'mimic3_dailyvitals_p1.csv'))
df2 = pd.read_csv(os.path.join(datadir, 'mimic3_dailyvitals_p2.csv'))
df3 = pd.read_csv(os.path.join(datadir, 'mimic3_dailyvitals_p3.csv'))
'''
df = pd.read_csv(os.path.join(datadir, 'mimic3_dailyvitals_pheno1.csv'))
df.drop_duplicates(inplace=True)

resdf=[]
LOS=[]
los_dict = {}
for pid, grp in df.groupby('patientid'):
	if len(grp)<MAX_LOS:
		LOS.append(len(grp))
		los_dict[pid] = len(grp)
		resdf.append(grp)
LOS = pd.Series(LOS)
df=pd.concat(resdf)
print(len(df))
print(LOS.describe())
print('#pa: %d'%len(df.patientid.unique()))
print('#deaths: %d'%len(df[df.label==1].patientid.unique()))
print(df.label.value_counts())


df['daysUntilDischarge'] = [los_dict[a]-b for a,b in zip(df.patientid, df.day)]

def getlabels(lst, thresh):
	return [int(x<=thresh) for x in lst]


res_df = []
for pid, grp in df.groupby('patientid'):
	newdf = pd.DataFrame(grp)
	N = len(newdf)
	if list(newdf.label)[-1]>0:
		newdf['label_all'] = [1]*N
		newdf['label_7days'] = getlabels(newdf.daysUntilDischarge, 7) 
		newdf['label_4days'] = getlabels(newdf.daysUntilDischarge, 6) 
		newdf['label_3days'] = getlabels(newdf.daysUntilDischarge, 5)
		newdf['label_4days'] = getlabels(newdf.daysUntilDischarge, 4) 
		newdf['label_3days'] = getlabels(newdf.daysUntilDischarge, 3)
		newdf['label_2days'] = getlabels(newdf.daysUntilDischarge, 2)  
	else:
		newdf['label_all'] = [0]*N
		newdf['label_7days'] = [0]*N
		newdf['label_6days'] = [0]*N
		newdf['label_5days'] = [0]*N
		newdf['label_4days'] = [0]*N
		newdf['label_3days'] = [0]*N
		newdf['label_2days'] = [0]*N
	res_df.append(newdf)

df = pd.concat(res_df)
df.to_csv(os.path.join(datadir, 'mimic3_dailyvitals_pheno1.csv'),index = False)