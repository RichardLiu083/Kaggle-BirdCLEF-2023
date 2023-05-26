import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score, average_precision_score

__all__= ['Accuracy', 'Mean_Recall', 'AUC', 'padded_cmap']

def Accuracy(all_pred, all_label):
    all_pred= all_pred.argmax(1)
    acc= list((all_label==all_pred)+0).count(1) / len(all_label)
    return acc

def Mean_Recall(all_pred, all_label):
    all_pred= all_pred.argmax(1)
    recall= recall_score(all_label, all_pred, average='macro')
    return recall

def AUC(all_pred, all_label):
    auc= roc_auc_score(all_label, all_pred, multi_class='ovo')
    return auc

def padded_cmap(all_pred, all_label, padding_factor=5):
    all_label= np.eye(264)[all_label]
    
    ## one hot embd
    solution = pd.DataFrame(all_label)
    submission = pd.DataFrame(all_pred)
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score