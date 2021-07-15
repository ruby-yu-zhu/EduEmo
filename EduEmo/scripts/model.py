from transformers import BertModel
import torch.nn.functional as F
import torch.nn as nn
import torch

import torch.nn.utils.rnn as R
from torch.autograd import Variable
import numpy as np

from RealFormerLayers import RealFormerEncoderLayer
from hard_easy_minibatch import hard_easy_samples


class BertEncoder(nn.Module):
    def __init__(self, lang='English'):
        """
        :param lang: str, train bert encoder for a given language
        """
        super(BertEncoder, self).__init__()
        if lang == 'English':
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        elif lang == 'Chinese':
            self.bert = BertModel.from_pretrained("bert-base-chinese")

        self.feature_size = self.bert.config.hidden_size
        self.attn_G = torch.nn.Linear(self.feature_size, 1)
    
    def edu2tuple(self, edu_, sen_start=15): # Chinese: sen_start=10
        edu_batch_lists = []
        for item_edu in edu_:
            edu_list = []
            for i,elm in  enumerate(item_edu):
                int_elm = int(elm)
                if i==0 and int_elm>sen_start:
                    edu_list.append((sen_start,int_elm))
                else:
                    if int(item_edu[i-1])<int_elm:
                        edu_list.append((int(item_edu[i-1])+1,int_elm))
            edu_batch_lists.append(edu_list)
        return edu_batch_lists

    def attention_net(self, en_output, src_EDU_lists, device):
        token_EDU_lists = []
        max_EDU_len = 0
        mask_EDU_lists = []
        padding_EDU = torch.zeros([self.feature_size,1],dtype=torch.float)
        
        padding_EDU = padding_EDU.to(device)
        for src_EDU_list in src_EDU_lists:
            if len(src_EDU_list)>max_EDU_len:
                max_EDU_len = len(src_EDU_list)

        for i, src_EDU_list in enumerate(src_EDU_lists):

            token_EDU_list = []
            mask_EDU_list = []
            for locations in src_EDU_list:
                start_index = locations[0]
                end_index = locations[1]
                temp_en_output = en_output[i,start_index:end_index+1,:] #[(end_index-start_index),512]
                if list(temp_en_output.size())[0]==0:
                    continue
                attn = F.softmax(self.attn_G(temp_en_output).permute([1,0]),dim=1) #1,17
                elm_EDU = torch.matmul(temp_en_output.permute([1,0]),attn.permute([1,0])) #768,1
                token_EDU_list.append(elm_EDU)
                mask_elm = torch.tensor(True)
                
                mask_elm = mask_elm.to(device)
                mask_EDU_list.append(mask_elm)
            while len(token_EDU_list)<max_EDU_len:
                token_EDU_list.append(padding_EDU)
                mask_elm = torch.tensor(False)
               
                mask_elm = mask_elm.to(device)
                mask_EDU_list.append(mask_elm)
            sen_EDU = torch.stack(tuple(token_EDU_list),dim=0) #48,768,1
            sen_mask_EDU = torch.stack(tuple(mask_EDU_list),dim=0)
            token_EDU_lists.append(sen_EDU)
            mask_EDU_lists.append(sen_mask_EDU)
        EDU = torch.cat(tuple(token_EDU_lists),dim=-1).permute([2,0,1]) #[32,48,768]
        mask_EDU = torch.stack(tuple(mask_EDU_lists))

        return EDU, mask_EDU


    def forward(self, input_ids, src_EDU_lists, device):
        """
        :param input_ids: list[str], list of tokenised sentences
        :return: last hidden representation, torch.tensor of shape (batch_size, seq_length, hidden_dim)
        """
        src_EDU_lists = self.edu2tuple(src_EDU_lists)
        last_hidden_state, pooler_output = self.bert(input_ids=input_ids) #32, 128(seq_length),768
        output_EDU, output_mask_EDU = self.attention_net(last_hidden_state,src_EDU_lists,device) #32, 5, 768  mask 32,5 
        output_mask_EDU = output_mask_EDU.unsqueeze(1) #32,1,5
        return last_hidden_state, output_EDU, output_mask_EDU


class EduEmo(nn.Module):
    def __init__(self, output_dropout=0.1, lang='English', joint_loss='joint', alpha=0.2,n_layers=6, n_head=12, d_k=64, d_v=64,d_inner=2048):
        """ casting multi-label emotion classification as span-extraction
        :param output_dropout: The dropout probability for output layer
        :param lang: encoder language
        :param joint_loss: which loss to use cel|corr|cel+corr
        :param alpha: control contribution of each loss function in case of joint training
        """
        super(EduEmo, self).__init__()
        self.bert = BertEncoder(lang=lang)
        self.joint_loss = joint_loss
        self.alpha = alpha

        self.layer_stack = nn.ModuleList([
            RealFormerEncoderLayer(self.bert.feature_size, d_inner, n_head, d_k, d_v, dropout=output_dropout)
            for _ in range(n_layers)])
        
        self.ffn = nn.Sequential(
            nn.Linear(self.bert.feature_size, self.bert.feature_size),
            nn.Tanh(),
            nn.Dropout(p=output_dropout),
            nn.Linear(self.bert.feature_size, 1)
        )

    def forward(self, batch, device, pre_attn=None, return_attns=True):
        """
        :param batch: tuple of (input_ids, labels, length, label_indices)
        :param device: device to run calculations on
        :return: loss, num_rows, y_pred, targets
        """
        #prepare inputs and targets
        inputs, targets, lengths, label_idxs, token_EDU_lists = batch  #inputs 32,128; targets 32,11; lengths 32; label_idxs 32;
        
        inputs, num_rows = inputs.to(device), inputs.size(0)
        label_idxs, targets = label_idxs[0].long().to(device), targets.float().to(device)

        #Bert encoder
        last_hidden_state, output_EDU, output_mask_EDU = self.bert(inputs, token_EDU_lists, device) #32, 128, 768 #32, 5, 768
        label_hidden_state = last_hidden_state.index_select(dim=1, index=label_idxs) #32, 11, 768
        label_EDU_state = torch.cat((label_hidden_state, output_EDU), 1) #32,16,768
        label_mask = (torch.ones([label_hidden_state.size(0), 1, label_hidden_state.size(1)])>0.5).to(device) #32,1,11
        label_EDU_mask = torch.cat((label_mask, output_mask_EDU), 2)
        enc_slf_attn_list = []

        #RealFormer encoder
        for enc_layer in self.layer_stack:
            label_EDU_state,  pre_attn = enc_layer(label_EDU_state,  pre_attn, slf_attn_mask=label_EDU_mask)
            enc_slf_attn_list += [pre_attn] if return_attns else []  #enc_output 32, 17, 768
        
        label_idxs_list = [x for x in range(targets.shape[1])]
        label_idxs = torch.tensor(label_idxs_list).to(device)

        # FFN---> 2 linear layers---> linear layer + tanh---> linear layer
        # select span of labels to compare them with ground truth ones
        logits = self.ffn(label_EDU_state).squeeze(-1).index_select(dim=1, index=label_idxs)
        y_pred = self.compute_pred(logits)

        #Hamming loss
        easy_batch, hard_batch = hard_easy_samples(batch, y_pred, targets)

        #Loss Function
        if self.joint_loss == 'joint':
            cel = F.binary_cross_entropy_with_logits(logits, targets).cuda()
            cl = self.corr_loss(logits, targets)
            loss = ((1 - self.alpha) * cel) + (self.alpha * cl)
        elif self.joint_loss == 'cross-entropy':
            loss = F.binary_cross_entropy_with_logits(logits, targets).cuda()
        elif self.joint_loss == 'corr_loss':
            loss = self.corr_loss(logits, targets)

        
        return loss, num_rows, y_pred, targets.cpu().numpy(),easy_batch, hard_batch

    @staticmethod
    def corr_loss(y_hat, y_true, reduction=''):
        """
        :param y_hat: model predictions, shape(batch, classes)
        :param y_true: target labels (batch, classes)
        :param reduction: whether to avg or sum loss
        :return: loss
        """
        loss = torch.zeros(y_true.size(0)).cuda()
        for idx, (y, y_h) in enumerate(zip(y_true, y_hat.sigmoid())):
            y_z, y_o = (y == 0).nonzero(), y.nonzero()
            if y_o.nelement() != 0:
                output = torch.exp(torch.sub(y_h[y_z], y_h[y_o][:, None]).squeeze(-1)).sum()
                num_comparisons = y_z.size(0) * y_o.size(0)
                loss[idx] = output.div(num_comparisons)
        return loss.mean() if reduction == 'mean' else loss.sum()
        
    @staticmethod
    def compute_pred(logits, threshold=0.5):
        """
        :param logits: model predictions
        :param threshold: threshold value
        :return:
        """
        y_pred = torch.sigmoid(logits) > threshold
        return y_pred.float().cpu().numpy()

