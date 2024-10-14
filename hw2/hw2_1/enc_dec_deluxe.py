import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import json

# ////////////////////////////////////////////////////////////////////////////////////////

def vocab_function():
    
    with open("./avi_data/MLDS_hw2_1_data/training_label.json", 'r') as file:
        
        train_data = json.load(file)
        
    long_str = "<BOS> <EOS> <PAD> <UKN> "
    
    max_word_len = 0
    
    for item in train_data:
        
        sent = item['caption'][0].lower().replace(".","")
        
        if len(sent.split()) > max_word_len:
            
            max_word_len = len(sent.split())
        
        long_str = long_str + sent + ' '
        
    unique_words = set(long_str.split())
    vocab_len = len(unique_words)
    
    vocab = {word : idx for idx, word in enumerate(unique_words)}
    rev_vocab = {idx : word for word, idx in vocab.items()}
    
    return vocab, rev_vocab, vocab_len, max_word_len


# /////////////////////////////////////////////////////////////////////////////////////////

class Pretrained_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv2d = nn.Conv2d(3, 16, 3, 1, 1)
        self.max2d = nn.MaxPool2d(kernel_size = (50,50), stride = (50,50))
        self.conv3d = nn.Conv3d(16, 10, kernel_size = (100, 1, 1), stride = (100, 1, 1))
        
    def forward(self, x):
        
        x = F.relu(self.conv2d(x))
        x = self.max2d(x)
        x = x.unsqueeze(0)
        # print(x.shape)
        x = x.view(1,16,1900,2,2)
        x = F.relu(self.conv3d(x))
        x = x.reshape(1,19,10*2*2)
        x = x.squeeze(0)
        # x = x.unsqueeze(0)
        
        return x     
    
# ///////////////////////////////////////////////////////////////////////////////////////////


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers):
        super().__init__()
    
        self.hid_dim = hid_dim
        
        self.lstm = nn.LSTM(input_dim, hid_dim, num_layers)
        
    def forward(self, x):
        
        h0 = torch.zeros(1, self.hid_dim)
        c0 = torch.zeros(1, self.hid_dim)
        
        out, (hn,cn) = self.lstm(x, (h0, c0))
        
        return out, hn, cn
    
# //////////////////////////////////////////////////////////////////////////////////////////////

class Attention(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, enc_hid_states, dec_hid_st):
        
        attn_scores = torch.matmul(enc_hid_states, dec_hid_st.transpose(0, 1))
        attn_weights = nn.functional.softmax(attn_scores, dim=0)
        context = torch.sum(attn_weights * enc_hid_states, dim=0)
        
        context = context.unsqueeze(0)
        
        return context
    
# ///////////////////////////////////////////////////////////////////////////////////////////////

def getWord(prob_out):
    
    word_tens = torch.zeros(1, prob_out.shape[-1])
    word_clone = word_tens.clone().detach()
    
    max_ind = torch.argmax(prob_out,dim=-1)
    word_clone[0,max_ind] = 1
    word_tens = word_clone
    
    return word_tens
    
# //////////////////////////////////////////////////////////////////////////////////////////////

# returns a (1, seq_len, vocab_len) tensor

def findSequence(dec_self, best_word, hid_st, cont_st, seq_len=3):
    
    seq_list = []
    hn_list = []
    input_word = best_word.unsqueeze(0)
    hn = hid_st
    cn = cont_st
    
    for seq_cnt in range(seq_len):
        
        output, (hn, cn) = dec_self.lstm(input_word, (hn, cn))
        lin_outs = dec_self.classify(output)
        prob_out = F.softmax(lin_outs, dim=-1)
        input_word = getWord(prob_out)
        hn_list.append(hn)
        seq_list.append(prob_out)
        
        
    seq_list_tens = torch.cat(seq_list, dim=0).unsqueeze(0)
    hn_tens = hn_list[-1]

    return seq_list_tens, hn_tens

# /////////////////////////////////////////////////////////////////////////////////////////////

# (3, seq_len, vocab_len) tensor and the indices of chosen words
# seq_list should have 5 (1, seq_len, vocab_len) tensors

def topSequences(seq_list):
    
    best_prev_ind = []
    best_seq_list = []
    seq_scores = []
    
    for seq_ind in range(len(seq_list)):
        
        seq_clone = seq_list[seq_ind].clone().detach()
        
        # hopefully will grab 3 max prob for each word in seq
        
        max_prob, _ = torch.max(seq_clone, dim=-1)
        
        # hopefully will sum each element in max_prob tensor
        
        score = torch.sum(max_prob)
        
        # should have five scores
        
        seq_scores.append(score)
        
    for ind in range(len(seq_scores)):
        focus_score = seq_scores[ind]
        less_than_cnt = 0
        
        for score in seq_scores:
        
            if focus_score < score:
                
                less_than_cnt += 1
                
        if less_than_cnt < 3 and len(best_prev_ind) < 3:
            
            best_prev_ind.append(ind)
            best_seq_list.append(seq_list[ind])
            
    best_seq_tens = torch.cat(best_seq_list, dim = 0)
            
    return best_seq_tens, best_prev_ind
                

# //////////////////////////////////////////////////////////////////////////////////////////////

class Beam_Search(nn.Module):
    
    def __init__(self, best_words_num):
        super().__init__()
        
        self.beam_len = best_words_num
        
    def forward(self, dec_self, best_words_tens, hid_st, cont_st, cur_ind, end_ind):
        
        # print("in BeamSearch:")
        
        best_seq_list = []
        hn_list = []
        fin_hn_list = []
        num_of_best_words = best_words_tens.shape[0]
        best_prev_ind = []
        seq_len = 0
        
        if (end_ind - cur_ind) >= self.beam_len:
            
            seq_len = self.beam_len
            
            seq_list = []
            
            for i in range(num_of_best_words):
                
                # returns a (1, seq_len, vocab_len) tensor. best_words_tens is a (5, vocab_len) tensor
                
                seq_tens, hn_tens= findSequence(dec_self, best_words_tens[i], hid_st, cont_st)
                seq_list.append(seq_tens)
                hn_list.append(hn_tens)
       
            # topSequences function needs to return a (3, seq_len, vocab_len) tensor and the indices of chosen words
            # seq_list should have 5 (1, seq_len, vocab_len) tensors
    
            top_3_seq, best_prev_ind = topSequences(seq_list)
            best_seq_list.append(top_3_seq)
            
            for ind in best_prev_ind:
                
                fin_hn_list.append(hn_list[ind])
                
        else:
            
            seq_len = end_ind - cur_ind
            seq_list = []
            
            for i in range(num_of_best_words):
                
                # returns a (1, seq_len, vocab_len) tensor. best_words_tens is a (5, vocab_len) tensor
                
                seq_tens, hn_tens = findSequence(dec_self, best_words_tens[i], hid_st, cont_st, seq_len)
                seq_list.append(seq_tens)
                hn_list.append(hn_tens)
                
            # topSequences function needs to return a (3, seq_len, vocab_len) tensor and the indices of chosen words
            # seq_list should have 5 (1, seq_len, vocab_len) tensors
                
            top_3_seq, best_prev_ind = topSequences(seq_list)
            best_seq_list.append(top_3_seq)
            
            for ind in best_prev_ind:
                
                fin_hn_list.append(hn_list[ind])
            
        best_seqs_tens = best_seq_list[0]
        
        # print(best_words_tens.shape)
        # print("///////////////////////")
        
        # should be a (3, seq_len, vocab_len) tensor. is also returning a list of best prev indices and hn_list
        
        return best_seqs_tens, best_prev_ind, fin_hn_list
        
        
# //////////////////////////////////////////////////////////////////////////////////////////////

# prob_tens should be (1,vocab_len) tens and this func should return a (5, vocab_len) tens

def getBestWords(prob_tens):
    
    # print("in getBestWords:")
    
    vocab_len = prob_tens.shape[-1]
    
    _ , max_ind = torch.topk(prob_tens, 5, dim = -1)
    
    best_word_list = []
    
    for ind in max_ind[0]:
        
        word_tens = torch.zeros(1, vocab_len)
        word_tens_clone = word_tens.clone().detach()
        word_tens_clone[0, ind.item()] = 1
        word_tens = word_tens_clone
        best_word_list.append(word_tens)
        
    best_word_tens = torch.cat(best_word_list, dim=0)
          
    # print(best_word_tens.shape)
    # print("////////////////////")
    
    return best_word_tens

# /////////////////////////////////////////////////////////////////////////////////////////////////

# addingtoEverythingTensor function is supposed to put best_seqs_tens and the vector for corresponding best_word vector together and then return 
# a (3, seq_len + 1, vocab_len) tensor

def addingToEverythingTensor(best_words_tens, prev_words_ind_lst, best_seqs_tens):
          
    # print("in addingtoEverythingTensor:")
    
    cnt = 0
    new_seq_list = []
    
    
    
    best_words_clone = best_words_tens.clone().detach()
    
    for ind in prev_words_ind_lst:
        
        # print(ind)
        # print("------------")
        
        word_tens = best_words_clone[ind].unsqueeze(0)
        
        # print(word_tens.shape)
        # print("in everything//////////////")
        
        new_seq_tens = torch.cat((word_tens, best_seqs_tens[cnt]), dim = 0)
        new_seq_list.append(new_seq_tens)
        cnt +=1
        
    new_seq_tens = torch.stack(new_seq_list)
    
    return new_seq_tens

# ////////////////////////////////////////////////////////////////////////////////////////////////

# best_3seq_list should be a list of 5 (3, seq_len, vocab_len) tensors, list of prev_words_list should be 5 lists of 3 ind
# should return a (3, seq_len, vocab_len) tensor and a list of 3 ind

def getTop3Sequences(best_3seq_list, list_ind_of_prev_words_list, list_of_hn_lists):
    
    # should return a (15, seq_len, vocab_len) tensor
    
    best_seq_list =[]
    
    for best_3seq in best_3seq_list:
        
        best_seq_list.append(best_3seq.unsqueeze(1))
        
    best_seq_tens = torch.cat(best_seq_list, dim=0)
    
    ind_list = []
    hn_list = []
    
    for list_3 in list_ind_of_prev_words_list:
    
        for ind in list_3:
        
            # ind_tens = torch.tensor([ind])
            ind_list.append(ind)
            
    for hn_3 in list_of_hn_lists:
        
        for hn in hn_3:
            
            hn_list.append(hn)
            
    # hn_list_tens = torch.stack(hn_list)
            
    # ind_list_tens = torch.stack(ind_list)
    
    seq_prob_score_list = []
    
    for two_d_row in best_seq_tens:
        
        max_val_tens, _ = torch.max(two_d_row, dim=-1)
        
        prob_score = torch.sum(max_val_tens)
        
        seq_prob_score_list.append(prob_score)
        
    best_ind_list = []
        
    for ind in range(len(seq_prob_score_list)):
        focus_score = seq_prob_score_list[ind]
        less_than_cnt = 0
        
        for score in seq_prob_score_list:
        
            if focus_score < score:
                
                less_than_cnt += 1
                
        if less_than_cnt < 3 and len(best_ind_list) < 3:
            
            best_ind_list.append(ind_list[ind])
            
        if len(best_ind_list) == 3:
        
            break
            
    top3_seq_list = []
    top3_ind_list = []
    top3_hn_list = []
            
    for ind in best_ind_list:
        
        top3_seq_list.append(best_seq_tens[ind])
        top3_ind_list.append(ind)
        top3_hn_list.append(hn_list[ind])
        
    top3_seq_tens = torch.cat(top3_seq_list)
    
    return top3_seq_tens, top3_ind_list, top3_hn_list
# //////////////////////////////////////////////////////////////////////////////////////////////

def scheduledSampling(temperature):

    if torch.rand(1).item() < temperature:
        
        return True

    else:
        
        return False


# ////////////////////////////////////////////////////////////////////////////////////////////////

class Decoder(nn.Module):
    def __init__(self, vocab_len, hid_dim, num_layers):
        super().__init__()
        
        self.hid_dim = hid_dim

        self.lstm = nn.LSTM(vocab_len, hid_dim, num_layers)
        self.classify = nn.Linear(hid_dim, vocab_len)
        
    def forward(self, enc_hid_sts, final_enc_hid_st, final_enc_con_st, vocab_len, training, targ_sent, temperature):
          
        # print("in decoder:")
        
         # training
        if (training):   
        
            sent_len = enc_hid_sts.shape[0]
            init_input = torch.zeros(1, vocab_len)
            attention = Attention()
            word_prob_list = []

            dec_output, (hn, cn) = self.lstm(init_input, (final_enc_hid_st, final_enc_con_st))
            linear_out = self.classify(dec_output)
            probs_out = F.log_softmax(linear_out, dim=-1)

            word_prob_list.append(probs_out)

            context = attention(enc_hid_sts, hn)

            for word_ind in range(1,sent_len,1):

                useTargWord = scheduledSampling(temperature)
                input_word = torch.zeros(1,vocab_len)

                if not useTargWord:

                    probs_out_clone = probs_out.clone().detach()
                    max_ind = torch.argmax(probs_out_clone)
                    
                    in_word_clone = input_word.clone().detach()
                    in_word_clone[0,max_ind.item()] = 1
                    input_word = in_word_clone

                else:

                    oneD_targ_sent = targ_sent[word_ind-1]
                    
                    input_word = oneD_targ_sent.unsqueeze(0)

                dec_output, (hn, cn) = self.lstm(input_word, (hn,context))
                linear_out = self.classify(dec_output)
                probs_out = F.log_softmax(linear_out, dim=-1)

                word_prob_list.append(probs_out)
                context = attention(enc_hid_sts, hn)
                
            inf_sent = torch.cat(word_prob_list, dim=0)
            
            # should be a (19,vocab_len) tensor
        
            return inf_sent
        
        # testing
        
        else:
        
        
            sent_len = enc_hid_sts.shape[0]
            init_input = torch.zeros(1, vocab_len)
            beamSearch = Beam_Search(3)
            attention = Attention()
            best_words_tens = torch.zeros(5, vocab_len)
            curr_word_ind  = 0
            all_seq_list = []
            everything_tens_ref = torch.zeros(3, 19, vocab_len)
            everything_tens = torch.zeros(3, 19, vocab_len)

            dec_output, (hn, cn) = self.lstm(init_input, (final_enc_hid_st, final_enc_con_st))
            linear_out = self.classify(dec_output)
            probs_out = F.softmax(linear_out, dim=-1)

            best_words_tens = getBestWords(probs_out)
          
            # print(best_words_tens.shape)
            # print("first call///////////////")

            context = attention(enc_hid_sts, hn)

        # Beam search should return indices of previous words for chosen sequences along with best sequences -> (3, seq_len, vocab_len) tensor

            best_seqs_tens, prev_words_ind_lst, hn_list = beamSearch(self, best_words_tens, hn, context, curr_word_ind, sent_len)

            curr_word_ind += best_seqs_tens.shape[1]
            # best_words_list =[]
            # best_words_list.append(best_words_tens)

        # addingtoEverythingTensor function is supposed to put best_seqs_tens and the vector for corresponding best_word vector together and then return 
        # a (3, 4, vocab_len) tensor

            block_tens = addingToEverythingTensor(best_words_tens, prev_words_ind_lst, best_seqs_tens)
            everything_tens_ref[:, :curr_word_ind + 1, :] = block_tens

            while curr_word_ind < sent_len:

                    best_3seq_list = []
                    list_ind_of_prev_words_list = []
                    list_of_hnLists =[]

                    # getting (seq_len, vocab_len) tensor from (3, seq_len, vocab_len) tensor
                    
                    hn_ind = 0

                    for seq in best_seqs_tens:

                        last_seq_word = seq[-1].unsqueeze(0)

                        input_word = torch.zeros(1, vocab_len)

                        input_word = last_seq_word

                        dec_output, (hn, cn) = self.lstm(input_word, (hn_list[hn_ind], cn))
                        linear_out = self.classify(dec_output)
                        probs_out = F.softmax(linear_out, dim=-1)

                        best_words_tens = getBestWords(probs_out)
                        
                        context = attention(enc_hid_sts, hn)
                        
                        if (curr_word_ind + 1) < 19:

        #                   if im finding 5 best words for 3 sequences, then this is a list of 3 tensors of shape(5,vocab_len)

                     # Beam search should return indices of previous words for chosen sequences along with best sequences -> (3, seq_len, vocab_len) tensor

                            new_best_seqs_tens, prev_words_ind_list, new_hn_list = beamSearch(self, best_words_tens, hn_list[hn_ind], context, curr_word_ind, sent_len)
                            list_ind_of_prev_words_list.append(prev_words_ind_list)
                            list_of_hnLists.append(new_hn_list)

                            best_3seq_list.append(new_best_seqs_tens)
                            
                        hn_ind +=1
                    
                    if (curr_word_ind + 1) < 19:

        #                best_3seq_list should be a list of 5 (3, seq_len, vocab_len) tensors, list of prev_words_list should be 5 lists of 3 ind
                        # should return a (3, seq_len, vocab_len) tensor and a list of 3 ind

                        best_seqs_tens, ind_of_prev_words, hn_list= getTop3Sequences(best_3seq_list, list_ind_of_prev_words_list, list_of_hnLists)
                
                        prev_word_ind = curr_word_ind     
                        curr_word_ind = prev_word_ind + best_seqs_tens.shape[1] + 1
                        
                        # print(best_words_tens.shape)
                        # print("sec call/////////////")
          
                        block_tens = addingToEverythingTensor(best_words_tens, ind_of_prev_words, best_seqs_tens)
            
            
                        everything_tens_ref[:, prev_word_ind :curr_word_ind, :] = block_tens
                        
                    else:
                        
                        
                        everything_tens_ref[0, 18, :] = best_words_tens[0]
                        everything_tens_ref[1,18, :] = best_words_tens[1]
                        everything_tens_ref[2, 18, :] = best_words_tens[2]
                        
                        curr_word_ind += 1
                    
            everything_tens = everything_tens_ref

            # is a (3,sent_len, vocab_len) tensor

            return everything_tens
 # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Enc_Dec_Model(nn.Module):
    def __init__(self, pre_cnn, encoder, decoder):
        super().__init__()
        self.pre_cnn = pre_cnn
        self.pre_cnn.eval()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, vid, vocab_len, training=False, targ_sent=[], temperature = 0):
        
        x = self.pre_cnn(vid)
        enc_out, hn, cn = self.encoder(x)
        inf_trainORtest_tens = self.decoder(enc_out, hn, cn, vocab_len, training, targ_sent, temperature)
        
        return inf_trainORtest_tens
        
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def transform_targ_sent(targ_sent, vocab):
    
    targ_len = len(targ_sent.split())
    full_targ_sent = ""
    
    if (targ_len + 2) <= 19:
    
        full_targ_sent = "<BOS> " + targ_sent.lower().replace(".","") + " <EOS>"
        
        if (targ_len + 2) < 19:
            
            n = 19 - (targ_len + 2)
            
            i = 1
            
            while i <= n:
                
                full_targ_sent = full_targ_sent + " <PAD>"
                
                i = i + 1
                
    else:
        
        word_splt = targ_sent.lower().replace(".","").split()
        
        full_targ_sent = "<BOS>"
        
        i = 2
        
        while (i < 19):
            
            full_targ_sent = full_targ_sent + " " + word_splt[i-2]
            
            i = i + 1
        
        full_targ_sent = full_targ_sent + " <EOS>"
        
    targ_sent_arr = []
    
    targ_sent_splt = full_targ_sent.split()
    
    for word in targ_sent_splt:
        
        if word in vocab:
            
            targ_sent_arr.append(vocab[word])
            
        else:
            
            targ_sent_arr.append(vocab["<UKN>"])
            
    targ_tens = torch.zeros(19, len(vocab))
    
    ind = 0
    for word_ind in targ_sent_arr:
        targ_tens[ind, word_ind] = 1
        ind = ind + 1
        
    # is a (19, vocab_len) tensor
        
    return targ_tens

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def padTensor(arr):
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
        
    ])
    
    tensor = torch.from_numpy(arr)
    
    num_samples = arr.shape[0]
    
    pad_depth = 1900 - num_samples
    
    pad_tensor = torch.nn.functional.pad(tensor, (0,0,0,0,0,0,0,pad_depth))
    
    aug_tensor = torch.stack([transform(frame) for frame in pad_tensor])
    
    return aug_tensor

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def coolingSchedule(iteration):
    
    scale = 1/20000
    
    num = iteration*scale
    
    temp = 1 - num
    
    return temp

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def training():
    
    vocab, _ , vocab_len, _ = vocab_function()
    
    cnn_model = torch.load("cnnModel_pretrained_2.pth", weights_only=True)
    pre_cnn = Pretrained_CNN()
    new_model_state_dict = pre_cnn.state_dict()
    filtered_state_dict = {k : v for k, v in cnn_model.items() if k in new_model_state_dict}
    new_model_state_dict.update(filtered_state_dict)
    pre_cnn.load_state_dict(new_model_state_dict)
    
    for param in pre_cnn.parameters():
        param.requires_grad = False
    
    encoder = Encoder(10*2*2, 50, 1)
    decoder = Decoder(vocab_len, 50, 1)
    mainModel = Enc_Dec_Model(pre_cnn, encoder, decoder)
    torch.autograd.set_detect_anomaly(True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mainModel.parameters(), lr = 0.001)
    training = True
    
    iteration = 0
    
    for epoch in range(10):
        
        with open("./avi_data/MLDS_hw2_1_data/training_label.json", 'r') as file:
    
            data = json.load(file)
        
        for item in data:
            
            targ_sent_tens = transform_targ_sent(item['caption'][0], vocab)
            
            file = item['id'].split('.')[0] + '.npy'
            
            file_path = "./avi_data/MLDS_hw2_1_data/training_data/npy_files/" + file
            
            arr = np.load(file_path)
            
            X = padTensor(arr)
            
            temp = coolingSchedule(iteration)
            
            inf_sent_tens = mainModel(X, vocab_len, training, targ_sent_tens,temp)
            
            loss = 0.0
            
            loss = criterion(inf_sent_tens, targ_sent_tens)
            
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            iteration += 1
                
    return mainModel

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
if __name__ == "__main__":
                        

    enc_dec_model = training()




    torch.save(enc_dec_model.state_dict(), "encoder_decoder_model_deluxe.pth")