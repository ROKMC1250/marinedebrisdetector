import torch
from torch import nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, \
    cohen_kappa_score, jaccard_score, accuracy_score

def get_loss(pos_weight=None):
    bcecriterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    def criterion(y_pred, target, mask=None):
        """a wrapper around BCEWithLogitsLoss that ignores no-data
        mask provides a boolean mask on valid data"""
        loss = bcecriterion(y_pred, target)
        if mask is not None:
            return (loss * mask.double()).mean()
        else:
            return loss.mean()
    return criterion

def compound_loss(coe, y_preds, target, common = False):
    bcecriterion = nn.BCEWithLogitsLoss()
    f_coe, c_coe = coe
    target.clamp_(0.01, 0.99)
    multi_loss = []
    if common == False:
        for i, feature in enumerate(y_preds):
            feature = feature.squeeze(1)
            ratio_f = 1 - i / len(y_preds)
            # ratio_c = (i+1) / (len(output_label))

            ihx = bcecriterion(feature, target) * ratio_f * f_coe 
            # if dist.get_rank() == 0:
            #     print(f'ihx: {ihx}, ihy: {ihy}')
            multi_loss.append(ihx)
            # feature_loss.append(torch.dist(output_feature[i], teacher_feature) *  feature_coe)
        # multi_loss.append(cecriterion(output_label[-1], target))
        # print(feature_loss)
        loss = torch.sum(torch.stack(multi_loss), dim=0)
        # +torch.mean(torch.stack(feature_loss), dim=0)
    else:
        ratio_f = 1 / len(y_preds)
        ihx = bcecriterion(y_preds, target) * ratio_f * f_coe 
        multi_loss.append(ihx)
        loss = torch.sum(torch.stack(multi_loss), dim=0)

    return loss 

def calculate_metrics(targets, scores, optimal_threshold):
    print(scores[0])
    predictions = scores > optimal_threshold
    print(optimal_threshold)
    print(predictions[0])
    auroc = roc_auc_score(targets, scores)
    p, r, f, s = precision_recall_fscore_support(y_true=targets,
                                                 y_pred=predictions, zero_division=0, average="binary")
    kappa = cohen_kappa_score(targets, predictions)

    jaccard = jaccard_score(targets, predictions)

    accuracy = accuracy_score(targets, predictions)

    summary = dict(
        auroc=auroc,
        precision=p,
        accuracy=accuracy,
        recall=r,
        fscore=f,
        kappa=kappa,
        jaccard=jaccard,
        threshold=optimal_threshold
    )

    return summary
