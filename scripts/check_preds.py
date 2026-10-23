import numpy as np
from cifar_resnet.evaluate import confusion_matrix,per_class_metrics,calibration
d = np.load("artifacts/preds/final_res20.npz")
probs = d["probs"]
labels = d["labels"]

print(probs.shape,labels.shape)
print(np.allclose(probs.sum(axis=1),1))
print((probs.argmax(axis=1)==labels).mean())


C = confusion_matrix(preds=probs.argmax(axis=1),labels=labels,num_classes=10)

print(C.sum(),np.trace(C) / C.sum(), C.sum(axis=1), C)
precision, recall, F1 = per_class_metrics(C)
print(precision, recall, F1)

classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

for name,precision_val,recall_val,F1_val in zip(classes,precision,recall,F1):
    print(f"Class {name:12s} has precision : {precision_val}, recall : {recall_val}, F1 score of {F1_val}")
print(F1.mean())


ece, counts, accs, confs = calibration(probs=probs,labels=labels,n_bins=15)

print(f"ECE:                 {ece:.4f}")
print(f"bin counts:          {counts}")
print(f"total count:         {counts.sum()}")
print(f"weighted acc:        {(counts * accs).sum() / len(labels):.4f}")
print(f"weighted conf:       {(counts * confs).sum() / len(labels):.4f}")
print(f"mean confidence:     {probs.max(axis=1).mean():.4f}")
print(f"accuracy:            {(probs.argmax(axis=1) == labels).mean():.4f}")