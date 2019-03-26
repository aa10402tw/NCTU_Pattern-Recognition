def AUC(FPR, TPR):
    area = 0
    for i in range(1, len(FPR)):
        area += (TPR[i-1] + TPR[i]) * (FPR[i]-FPR[i-1]) / 2
    return abs(area)