import torch 

import numpy as np 

import torchvision


# This is for the purpose of testing this is why there exists "y"s in the loader
def predict(model,loader,device):
    model.eval()
    probs = []
    labels = [] 
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            prediction_logit = model(x)
            probability = torch.softmax(prediction_logit,dim=1)
            probs.append(probability.cpu())
            labels.append(y)
        probs = torch.cat(probs,dim=0)
        labels = torch.cat(labels,dim=0)
    return probs.numpy(),labels.numpy()

def confusion_matrix(preds,labels,num_classes=10):
    C = np.zeros((num_classes,num_classes),dtype=int)
    for truth,pred in zip(labels,preds):
            C[truth,pred] +=1
    return C 

def per_class_metrics(C):
    TP = np.diag(C)
    FP = C.sum(axis=0) - TP
    FN = C.sum(axis=1) - TP

    precision = np.divide(
    TP,
    TP + FP,
    out=np.zeros_like(TP, dtype=float),
    where=(TP + FP) > 0
                        )
    recall = np.divide(
    TP,
    TP + FN,
    out=np.zeros_like(TP, dtype=float),
    where=(TP + FN) > 0
)
    F1 = np.divide(
    2 * precision * recall,
    precision + recall,
    out=np.zeros_like(precision, dtype=float),
    where=(precision + recall) > 0
)
    return precision, recall, F1


def calibration(probs,labels,n_bins=15):
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (preds == labels)
    bins = np.linspace(0, 1, n_bins + 1)
    counts = np.zeros(n_bins, dtype=int)
    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (conf > lo) & (conf <= hi)
        counts[i] = mask.sum()
        if counts[i] > 0:
             accs[i] = correct[mask].mean()
             confs[i] = conf[mask].mean()
    ece = np.sum((counts / len(labels)) * np.abs(accs - confs))
    return ece, counts, accs, confs
