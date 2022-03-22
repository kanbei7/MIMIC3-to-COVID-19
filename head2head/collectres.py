import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import ttest_rel
import seaborn as sns

resdir = 'res'
outdir = 'analysis'

def getAUPRAUROC(fname):
	with open(os.path.join(resdir,fname),'r') as f:
		lines = f.readlines()
	auroc = []
	aupr = []
	for line in lines:
		k,v = line.split(':')
		k,v = k.strip(),v.strip()
		if k=='TEST AUROC':
			auroc.append(float(v))
		if k=='TEST AUPR':
			aupr.append(float(v))
	return np.array(auroc),np.array(aupr)

def ttest_greater(x1,x2):
	_, pv = ttest_rel(x1, x2, alternative = 'greater')
	return pv, np.mean(x1) - np.mean(x2), np.mean(x1-x2), np.mean(x1-x2)/np.mean(x1)

def compare(x1,y1,x2,y2):
	pvalue, md1, md2, rd1 = ttest_greater(x1,x2)
	print('[AUROC] p-value: %.8f, '%pvalue + 'rd1: %.6f,  '%rd1 + 'md1: %.6f, '%md1+ 'md2: %.6f'%md2 )
	pvalue, md1, md2, rd1 = ttest_greater(y1,y2)
	print('[AUPR] p-value: %.8f, '%pvalue + 'rd1: %.6f,  '%rd1 + 'md1: %.6f, '%md1+ 'md2: %.6f'%md2 )

	return

def analyze(N):
	fname1 = 'h2h_base_%d_summary.txt'%N
	fname2 = 'h2h_AllPheno_%d_summary.txt'%N
	fname3 = 'h2h_SelectedPheno_%d_summary.txt'%N

	auroc1, aupr1 = getAUPRAUROC(fname1)
	auroc2, aupr2 = getAUPRAUROC(fname2)
	auroc3, aupr3 = getAUPRAUROC(fname3)

	#3-1
	print('%d : 3-1'%N)
	compare(auroc3, aupr3, auroc1, aupr1)
	print('-'*10)

	#3-2
	print('%d : 3-2'%N)
	compare(auroc3, aupr3, auroc2, aupr2)
	print('-'*10)

	#2-1
	print('%d : 2-1'%N)
	compare(auroc2, aupr2, auroc1, aupr1)
	print('-'*10)

	return


#analyze(0)
analyze(5)

#for i in range(0,5):
#	analyze(i)


