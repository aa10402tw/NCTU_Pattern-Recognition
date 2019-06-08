import numpy as np

def binary_confusion_matrix(labels, preds):
    if len(labels) != len(preds):
        raise Exception(" len(lables) != len(preds) [%d != %d]" %(len(labels), len(preds)))

    TP, FP = 0, 0
    FN, TN = 0, 0
    for label, pred in zip(labels, preds):
        if(label==pred and pred==1):
            TP += 1
        elif(label==pred and pred==0):
            TN += 1
        elif(label!=pred and pred==1):
            FP +=1
        elif(label!=pred and pred==0):
            FN += 1
    if(min([FP+TN, TP+FN])) == 0:
        print(TP, FP)
        print(FN, TN)
    return TP, FP, FN, TN


def confusion_matrix(labels, preds, normalzied=False):
    if len(labels) != len(preds):
        raise Exception(" len(lables) != len(preds) [%d != %d]" %(len(labels), len(preds)))

    num_classes = len(set(labels))
    confusion_mat = np.zeros((num_classes, num_classes))
    for (label, pred) in zip(labels, preds):
        confusion_mat[int(label)][int(pred)] += 1
    if normalzied:
        confusion_mat /= np.sum(confusion_mat, axis=1)
    return confusion_mat


