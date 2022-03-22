import pandas as pd
import numpy as np
from numpy import take
from numpy.random import shuffle 
from sklearn.model_selection import KFold

'''
input:
int kfold,
dataframe df,
str group_name,
str label (assume binary, {0,1})

return: 
a list of k index lists.
[(train_idx, test_idx), (train_idx, test_idx), ... ]
'''

def bin_strat(data, group_name, kfold, n_bins):
	z = pd.DataFrame(data.groupby(group_name).size())
	z.columns = ['cnt']
	z["bin"] = pd.qcut(z.cnt,n_bins,duplicates = 'drop')
	bins = [grp.index.tolist() for name,grp in z.groupby('bin')]
	random_seeds = np.random.randint(100000000, size = n_bins)

	train_name = [[] for k in range(kfold)]
	test_name = [[] for k in range(kfold)]
	for i in range(len(bins)):
		kf = KFold(n_splits= kfold, shuffle = True, random_state = random_seeds[i])
		k=0
		for train_idx, test_idx in kf.split(bins[i]):
			train_name[k].extend(take(bins[i], train_idx))
			test_name[k].extend(take(bins[i], test_idx))
			k+=1

	return train_name, test_name

#return group name 
def groupKfold(data, group_name, label_name, kfold = 5, n_bins = 5):
	train, test = bin_strat(data, group_name, kfold, n_bins)

	train_kfold = []
	test_kfold = []
	for k in range(kfold):
		tmp_train = train[k]
		shuffle(tmp_train)
		shuffle(tmp_train)
		train_kfold.append(tmp_train)

		tmp_test = test[k] 
		shuffle(tmp_test)
		shuffle(tmp_test)
		test_kfold.append(tmp_test)
	return train_kfold, test_kfold
	
def groupStratifiedKfold(data, group_name, label_name, kfold = 5, n_bins = 5):
	pa_pos = list(data[data[label_name]==1][group_name].unique())
	df_pos = data[data[group_name].isin(pa_pos)] 

	pa_neg = [x for x in data[group_name].unique() if x not in pa_pos]
	df_neg = data[data[group_name].isin(pa_neg)]

	train_pos, test_pos = bin_strat(df_pos, group_name, kfold, n_bins)
	train_neg, test_neg= bin_strat(df_neg, group_name, kfold, n_bins)

	train_kfold = []
	test_kfold = []
	for k in range(kfold):
		tmp_train = train_pos[k] + train_neg[k]
		shuffle(tmp_train)
		shuffle(tmp_train)
		train_kfold.append(tmp_train)

		tmp_test = test_pos[k] + test_neg[k]
		shuffle(tmp_test)
		shuffle(tmp_test)
		test_kfold.append(tmp_test)
	return train_kfold,test_kfold
















