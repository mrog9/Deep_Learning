import json
from transformers import BertModel
import torch
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

def transformInput(q, c, wi_vocab):
    
    ind_list = []
    token_type_list = []
    zero = torch.tensor([0])
    one = torch.tensor([1])
    
    for word in q:
        
        ind = wi_vocab[word]
        ind_tens = torch.tensor([ind])
        ind_list.append(ind_tens)
        token_type_list.append(zero)
        
    for word in c:
        
        ind = wi_vocab[word]
        ind_tens = torch.tensor([ind])
        ind_list.append(ind_tens)
        token_type_list.append(one)
        
    full_ind_tens = torch.stack(ind_list, dim=-1)
    token_type_tens = torch.stack(token_type_list, dim=-1)
    
    return full_ind_tens, token_type_tens

def createAttentionMask(q, c):
    
    qc_len = len(q) + len(c)
    qc_list = q + c
    
    mask_tens = torch.ones(1, qc_len)
    
    for ind in range(qc_len):
        
        if qc_list[ind] == "<PAD>":
            
            mask_tens[0, ind] = 0
            
    return mask_tens



class startModel(torch.nn.Module):
    def __init__(self, bertModel, hidden_size=768, context_len = 40, question_len=35):
        super().__init__()
        
        self.hiddenSize = hidden_size
        self.seq_len = context_len + question_len
        self.bert = bertModel
        self.linear1 = torch.nn.Linear(hidden_size, 250)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(250, 100)
        # self.linear3 = torch.nn.Linear(self.seq_len * 50, 500)
        self.linear3 = torch.nn.Linear(100, 40)
        self.linear4 = torch.nn.Linear(40, 10)
        self.linear5 = torch.nn.Linear(10, 1)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        
    def forward(self, qc_tens, attention_tens, token_type_tens):
        
        inputs = {'input_ids':qc_tens, 'attention_mask':attention_tens, 'token_type_ids':token_type_tens }
        
        outputs = self.bert(**inputs)
        h = outputs.last_hidden_state
        batch_num = h.size(0)
        h = h.reshape(batch_num * self.seq_len, self.hiddenSize)
        x = self.relu(self.linear1(h))
        x = self.relu(self.linear2(x))
        # x = x.reshape(batch_num, self.seq_len * 50)
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.linear5(x)
        x = x.reshape(batch_num, self.seq_len, 1)
        x = x[:,35:,:]
        x = x.reshape(1, batch_num * 40)
        p = self.softmax(x)
       
        return p
    
class endModel(torch.nn.Module):
    def __init__(self, bertModel, hidden_size=768, context_len = 40, question_len=35):
        super().__init__()
        
        self.hiddenSize = hidden_size
        self.seq_len = context_len + question_len
        self.bert = bertModel
        self.linear1 = torch.nn.Linear(hidden_size, 250)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(250, 100)
        # self.linear3 = torch.nn.Linear(self.seq_len * 50, 500)
        self.linear3 = torch.nn.Linear(100, 40)
        self.linear4 = torch.nn.Linear(40, 10)
        self.linear5 = torch.nn.Linear(10, 1)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        
    def forward(self, qc_tens, attention_tens, token_type_tens):
        
        inputs = {'input_ids':qc_tens, 'attention_mask':attention_tens, 'token_type_ids':token_type_tens }
        
        outputs = self.bert(**inputs)
        h = outputs.last_hidden_state
        batch_num = h.size(0)
        h = h.reshape(batch_num * self.seq_len, self.hiddenSize)
        x = self.relu(self.linear1(h))
        x = self.relu(self.linear2(x))
        # x = x.reshape(batch_num, self.seq_len * 50)
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.linear5(x)
        x = x.reshape(batch_num, self.seq_len, 1)
        x = x[:,35:,:]
        x = x.reshape(1, batch_num * 40)
        p = self.softmax(x)
       
        return p
    
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            

def trainBert():
    
    start_time = time.time()
    
    model_name = "bert-base-uncased"
    
    bert_model = BertModel.from_pretrained(model_name)
    
    with open("word_ind_vocab.json", 'r') as file:
        
        wi_vocab = json.load(file)
        
    s_model = startModel(bert_model)
    e_model = endModel(bert_model)
    
    s_model.apply(init_weights)
    e_model.apply(init_weights)
    
    s_optimizer = torch.optim.Adam(s_model.parameters(), lr=0.000001)
    e_optimizer = torch.optim.Adam(e_model.parameters(), lr=0.000001)
    
    s_scheduler = CosineAnnealingLR(s_optimizer, T_max = 100, eta_min = 0.0000001)
    e_scheduler = CosineAnnealingLR(e_optimizer, T_max = 100, eta_min = 0.0000001)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with open("newTrainingSet.json", 'r') as file:

        data = json.load(file)
        
    item_ind = 0

    for item in data:
  
        q = item['question']
        c_list = item['context_win']
            
        ind = 0
        qc_list = []
        tt_list = []
        m_list = []
        
        for c in c_list:
            
            in_tens, token_type = transformInput(q, c, wi_vocab)
            mask_tens = createAttentionMask(q,c)
            
            qc_list.append(in_tens)
            tt_list.append(token_type)
            m_list.append(mask_tens)
            
            ind += 1
            
        qc_batch = torch.cat(qc_list, dim = 0)
        tt_batch = torch.cat(tt_list, dim = 0)
        m_batch = torch.cat(m_list, dim = 0)
            
        inf_start_probs = s_model(qc_batch, m_batch, tt_batch)
        inf_end_probs = e_model(qc_batch, m_batch, tt_batch)
        
        start_tensor = torch.tensor([item['word_start']])
        end_tensor = torch.tensor([item['word_finish']])
        
        s_loss = criterion(inf_start_probs, start_tensor)
        s_loss.backward()
        
        e_loss = criterion(inf_end_probs, end_tensor)
        
        _, s_ind = torch.max(inf_start_probs, dim=-1)
        _, e_ind = torch.max(inf_end_probs, dim=-1)
        
        if s_ind > e_ind:
            
            e_loss *= 2
        
        
        e_loss.backward()

        print(f"index: {item_ind} \n")
        print(f"start_loss: {s_loss.item()}")
        print(f"end_loss: {e_loss.item()}")
        print("\n---------------------------------")
            
        s_optimizer.step()
        e_optimizer.step()
                                  
        if (item_ind + 1)%15 == 0:
                                  
            s_scheduler.step()
            e_scheduler.step()
        
        item_ind += 1
        
    end_time = time.time()
    
    program_time = end_time - start_time
    
    print(f"Program time: {program_time}")
    
    return s_model, e_model
        
                            
if __name__ == '__main__':
            
    s_model, e_model = trainBert()

    torch.save(s_model.state_dict(), 'startModel.pth')
    torch.save(e_model.state_dict(), 'endModel.pth')
