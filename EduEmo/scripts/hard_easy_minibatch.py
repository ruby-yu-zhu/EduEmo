from sklearn import metrics
import torch
def hard_easy_samples(batch, y_pred,targets):
    hamming_loss_list=[]
    hard_batch=[]
    easy_batch=[]
    hard_edu_batch=[]
    easy_edu_batch=[]
    for i in range(targets.shape[0]):
        hamming_loss = metrics.hamming_loss(targets[i,:].cpu(), y_pred[i,:])
        hamming_loss_list.append(hamming_loss)
    easy_batch_indexs =  [i for i in range(len(hamming_loss_list)) if hamming_loss_list[i]<0.5]
    hard_batch_indexs =  [i for i in range(len(hamming_loss_list)) if hamming_loss_list[i]>=0.5]
    for b in batch[:-1]:
        easy_batch.append(torch.index_select(b,0,index=torch.Tensor(easy_batch_indexs).long()))
        hard_batch.append(torch.index_select(b,0,index=torch.Tensor(hard_batch_indexs).long()))
    for i in easy_batch_indexs:
        easy_edu_batch.append(batch[-1][i])
    for i in hard_batch_indexs:
        hard_edu_batch.append(batch[-1][i])
    easy_batch.append(easy_edu_batch)
    hard_batch.append(hard_edu_batch)
    return easy_batch, hard_batch
