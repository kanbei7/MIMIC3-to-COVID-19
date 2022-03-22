import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os
import sys

N = 100*int(sys.argv[1])

baseline_file = 'baseline_3d_slice%d-%d.csv'%(N,N+200)
method_file = 'MIMICXGB_3d_slice%d-%d.csv'%(N,N+200)

baseline_df = pd.read_csv(os.path.join('res',baseline_file))
method_df = pd.read_csv(os.path.join('res',method_file))

base_y_pred, base_y_true = baseline_df.y_pred , baseline_df.y_true
method_y_pred, method_y_true = method_df.y_pred , method_df.y_true

# plot AUROC
plt.figure()
fpr_lr, tpr_lr, thresholds_lr = roc_curve(base_y_true, base_y_pred)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr,tpr_lr, 'b--', lw=1,label='XGB baseline(AUROC = %.4f)'%roc_auc_lr)

fpr_lr, tpr_lr, thresholds_lr = roc_curve(method_y_true, method_y_pred)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr,tpr_lr, color='red', lw=1,label='XGB+MIMIC3 (AUROC = %.4f)'%roc_auc_lr)

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve, 3 days ahead Mortality Prediction\n (Trained on first %d patients)'%N)
plt.legend(loc="lower right")
plt.savefig(os.path.join('fig','TestAUROC_3d_%d.png'%N))
plt.close()

# plot AUPR


plt.figure()
precision, recall, thresholds = precision_recall_curve(base_y_true, base_y_pred)
aupr_score = average_precision_score(base_y_true, base_y_pred)
plt.plot(recall,precision, 'b--', lw=1,label='XGB baseline(AUPR = %.4f)'%aupr_score)


precision, recall, thresholds = precision_recall_curve(method_y_true, method_y_pred)
aupr_score = average_precision_score(method_y_true, method_y_pred)
plt.plot(recall,precision, color='red', lw=1,label='XGB+MIMIC3 (AUPR = %.4f)'%aupr_score)

plt.plot([0, 1], [1, 0], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('PR Curve, 3 days ahead Mortality Prediction\n (Trained on first %d patients)'%N)
plt.legend(loc="lower right")
plt.savefig(os.path.join('fig','TestAUPR_3d_%d.png'%N))
plt.close()