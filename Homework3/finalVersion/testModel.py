import json
from transformers import BertModel
import torch

from train import startModel, endModel, transformInput, createAttentionMask

def createWindows(prob_tensor):
    
    n = prob_tensor.size(-1)
    win_list = []
    
    for i in range(0,n,40):
        
        temp = prob_tensor[0,i:i+40].unsqueeze(0)
        win_list.append(temp)
        
    win_tens = torch.cat(win_list, dim=0)
        
    return win_tens


def findWinHitList(s_prob_tens, e_prob_tens):

    s_vals, _ = torch.max(s_prob_tens, dim=-1)
    e_vals, _ = torch.max(e_prob_tens, dim=-1)
    
    score_list = []
    inf_h_list=[]
    
    for i in range(s_vals.size(0)):
        
        score_list.append(s_vals[i].item() + e_vals[i].item())
        inf_h_list.append(0)
        
    max_val = max(score_list)
    
    max_ind = score_list.index(max_val)
    
    inf_h_list[max_ind] = 1
    
    return inf_h_list
    

def findStatistics(start_list,end_list, data):
    
    corr_ones = 0
    true_ones = 0
    pred_ones = 0
    corr_hits = 0
    tot_poss_hits = 0
    
    for ind, item in enumerate(data):
        
        inf_h_list = findWinHitList(start_list[ind], end_list[ind])
        
        curr_ind = 0
        pred_ones+=1
        
        for h in item['win_hit']:
            
            tot_poss_hits += 1
            
            if h == 1:
                
                true_ones += 1
                
                if inf_h_list[curr_ind] == h:
            
                    corr_ones+=1
                
            elif h==0 and inf_h_list[curr_ind]==0:
                
                corr_hits+=1
                
            curr_ind+=1
            
    accuracy = corr_hits/tot_poss_hits
    
    precision = 0
    recall = 0
    f1_score = 0
        
    if pred_ones !=0:
        
        precision = corr_ones/pred_ones
        
    else:
        precision = 0.0
            
    recall = corr_ones/true_ones

    if precision+recall != 0:
        f1_score = 2*((precision*recall)/(precision+recall))
    else:
        f1_score = -1
    

    return accuracy, precision, recall, f1_score
    

def evaluateModel():
    
    with open('word_ind_vocab.json', 'r') as file:
        wi_vocab = json.load(file)
    
    model_name = "bert-base-uncased"
    
    bert_model = BertModel.from_pretrained(model_name)
    
    s_model = startModel(bert_model)
    e_model = endModel(bert_model)
    
    s_model.load_state_dict(torch.load('startModel.pth', weights_only = True))
    e_model.load_state_dict(torch.load('endModel.pth', weights_only = True))
    
    print("////////////////////////////////////////////////////////////////////////////")
    
    with open("newTestingSet.json", 'r') as file:
        data = json.load(file)
    
    s_model.eval()
    e_model.eval()
    
    with torch.no_grad():
        
        inf_start_list = []
        inf_end_list = []
        
        batch_in_list = []
        batch_mask_list = []
        batch_tt_list =[]
    
        for item in data:
            
            q = item['question']
            
            hit_ind = 0

            for w in item['context_win']:
                
                input_tens, tt = transformInput(q, w, wi_vocab)
                mask = createAttentionMask(q, w)
                
                batch_in_list.append(input_tens)
                batch_mask_list.append(mask)
                batch_tt_list.append(tt)
                    
            batch_in_tens = torch.cat(batch_in_list, dim=0)
            batch_mask_tens = torch.cat(batch_mask_list, dim=0)
            batch_tt_tens = torch.cat(batch_tt_list, dim = 0)

            batch_in_list = []
            batch_mask_list = []
            batch_tt_list = []
                
            startProbs = s_model(batch_in_tens, batch_mask_tens, batch_tt_tens)
            endProbs = e_model(batch_in_tens, batch_mask_tens, batch_tt_tens)
            
            s_wins = createWindows(startProbs)
            e_wins = createWindows(endProbs)
                    
            inf_start_list.append(s_wins)
            inf_end_list.append(e_wins)
            
            # print(f"TRUE start: {item['word_start']}")
            # print(f"INF start: {start_ind}")
            # print(f"TRUE end: {item['word_finish']}")
            # print(f"INF start: {end_ind}")
            
        acc, pre, re, f1 = findStatistics(inf_start_list, inf_end_list, data)
        
        # e_acc, e_pre, e_re, e_f1 = findStatistics(inf_end_list, data)               
                    
        print(f"ACCURACY: {acc} \n")
        print(f"PRECISION: {pre} \n")
        print(f"RECALL: {re} \n")
        print(f"F1_SCORE: {f1} \n")
        
#         print(f"END ACCURACY: {e_acc} \n")
#         print(f"END PRECISION: {e_pre} \n")
#         print(f"END RECALL: {e_re} \n")
#         print(f"END F1_SCORE: {e_f1} \n")
                
        
                
        
        

if __name__ == "__main__":

    evaluateModel()