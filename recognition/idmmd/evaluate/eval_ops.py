import numpy as np
from sklearn.metrics import roc_curve

def evaluate2(gallery_feat, query_feat, labels, fars = [10**-5, 10**-4, 10**-3, 10**-2]):
    query_num = query_feat.shape[0]

    similarity = np.dot(query_feat, gallery_feat.T)
    top_inds = np.argsort(-similarity)
    labels = labels.T
    
    # calculate top1
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0]
        if labels[i, j] == 1:
            correct_num += 1
    top1 = correct_num / query_num
    print("top1 = {:.2%}".format(top1))

    # # calculate top5
    # correct_num = 0
    # for i in range(query_num):
    #     j = top_inds[i, :5]
    #     if any(labels[i, j] == 1.0):
    #         correct_num += 1
    #     # else:
    #     #     print(i,j)
    # top5 = correct_num / query_num
    # print("top5 = {:.4%}".format(top5))

    # # calculate 10
    # correct_num = 0
    # for i in range(query_num):
    #     j = top_inds[i, :10]
    #     if any(labels[i, j] == 1.0):
    #         correct_num += 1
    #     # else:
    #     #     print(i,j)
    # top10 = correct_num / query_num
    # print("top10 = {:.4%}".format(top10))

    labels_ = labels.flatten()
    similarity_ = similarity.flatten()
    fpr, tpr, _ = roc_curve(labels_, similarity_)

    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)
    tpr_fpr_row = []
    for far in fars:
        _, min_index = min(list(zip(abs(fpr - far), range(len(fpr)))))
        tpr_fpr_row.append(tpr[min_index])
        print("TPR {:.2%} @ FAR {:.4%}".format(tpr[min_index], far))
        
    return [top1], tpr_fpr_row
