"""nDCG@5, MAP and AUC with exact expected treatment of score ties."""
import numpy as np

def ranking_metrics(labels, scores):
    labels=np.asarray(labels,dtype=int)
    scores=np.asarray(scores,dtype=float)
    if labels.ndim!=1 or scores.shape!=labels.shape or not np.isfinite(scores).all():
        raise ValueError('Scores must be a finite one-dimensional array matching labels')
    if not np.isin(labels,[0,1]).all() or not 0<labels.sum()<len(labels):
        raise ValueError('Both binary click labels are required')
    order=np.argsort(-scores,kind='stable')
    y,s=labels[order],scores[order]
    total_positive=int(y.sum()); total_negative=len(y)-total_positive
    idcg=sum(1/np.log2(np.arange(2,min(5,total_positive)+2)))
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    ends=np.r_[starts[1:],len(y)]
    dcg=ap_sum=auc_wins=0.
    positives_before=negatives_before=0
    for start,end in zip(starts,ends):
        size=int(end-start); positive=int(y[start:end].sum()); negative=size-positive
        if positive:
            j=np.arange(1,size+1,dtype=float)
            expected_previous=(j-1)*(positive-1)/(size-1) if size>1 else np.zeros(1)
            ap_sum+=float(np.sum((positive/size)*(positives_before+1+expected_previous)/(start+j)))
            top_end=min(end,5)
            if start<top_end:
                dcg+=(positive/size)*float(np.sum(1/np.log2(np.arange(start+2,top_end+2))))
            lower_negatives=total_negative-negatives_before-negative
            auc_wins+=positive*(lower_negatives+.5*negative)
        positives_before+=positive
        negatives_before+=negative
    return dict(ndcg_at_5=float(dcg/idcg),map=float(ap_sum/total_positive),
                auc=float(auc_wins/(total_positive*total_negative)))
