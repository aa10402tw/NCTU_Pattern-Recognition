import numpy as np
import matplotlib.pyplot as plt
from .confusion_matrix import confusion_matrix
from .auc import AUC
'''
input :
    y: shape (M)
    y_pred_prob: shape(M)

useage:
    y_pred_prob = model.predict_prob(X_test)
    y_pred_prob = y_pred_prob[:, 1].reshape(-1)
    FPR, TPR = ROC(y_test, y_pred_prob, label=str(model))
'''
def ROC(y, y_pred_prob, label='ROC'):
    if len(y) != len(y_pred_prob):
        raise Exception(" len(y) != len(y_pred) [%d != %d]" %(len(y), len(y_pred_prob)))
    thresholds = sorted(list(y_pred_prob))
    FPR, TPR = [], []
    for threshold in thresholds:
        y_pred = y_pred_prob.copy()
        y_pred[y_pred<threshold] = 0
        y_pred[y_pred>=threshold] = 1
        confusion_mat = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = tuple(confusion_mat.reshape(4))
        FPR.append( FP/(FP+TN) )
        TPR.append( TP/(TP+FN) )
    plt.plot([0,1], [0,1], linestyle='dashed')
    plt.plot(FPR, TPR, label='%s (AUC=%.2f)'%(label, AUC(FPR, TPR)))
    plt.legend(loc='best')
    return FPR, TPR

'''
input :
    y: shape (MxC)
    y_pred_prob: shape(MxC)

useage:
    y_pred_prob = model.predict_prob(X_test)
    FPR, TPR = multiclass_ROC(onehot(y_test), y_pred_prob)

'''
def multiclass_ROC(y, y_pred_prob, label='ROC', n_bins=1000):
    M, C = y.shape
    n_bins = n_bins
    FPR = np.linspace(0, 1, n_bins)
    TPR_sum = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for c in range(C):
        labels = y[:, c].reshape(-1)
        preds_ = y_pred_prob[:, c].reshape(-1)
        thresholds = sorted(list(preds_))
        for threshold in thresholds:
            preds = preds_.copy()
            preds[preds<threshold] = 0
            preds[preds>=threshold] = 1
            confusion_mat = confusion_matrix(labels, preds)
            TN, FP, FN, TP = tuple(confusion_mat.reshape(4))
            fpr = FP/(FP+TN)
            tpr = TP/(TP+FN)
            index = min(int(fpr * n_bins), n_bins-1)
            counts[index] += 1
            TPR_sum[index] += tpr
    FPR = FPR[counts!=0]
    TPR_sum = TPR_sum[counts!=0]
    counts = counts[counts!=0]
    TPR = TPR_sum / counts
    plt.plot([0,1], [0,1], linestyle='dashed')
    plt.plot(FPR, TPR, label='%s (AUC=%.2f)'%(label, AUC(FPR, TPR)))
    plt.legend(loc='best')
    return FPR, TPR