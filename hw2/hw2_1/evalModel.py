import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import json
import sys
import os

import enc_dec_deluxe


def convert_avi(file_path):
    
    file_path = "./avi_data/MLDS_hw2_1_data/training_data/video/" + file_str
    
    cap = cv2.VideoCapture(file_path)
    frames = []
    
    while cap.isOpened():
        ret,frame = cap.read()
        
        if not ret:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
    cap.release()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100,100)),
        transforms.ToTensor()
    ])
    
    processed_frames = [transform(frame) for frame in frames]
    
    input_tensor = torch.stack(processed_frames)
    
    return input_tensor




def pred_to_sent(pred, rev_vocab):
    
    ind_arr = []
    
    _, inds = torch.max(pred, dim=-1)
    word_lst = []
    
    for ind in inds:
        
        word_lst.append(rev_vocab[ind.item()])
    
    sent = ""

    for word in word_lst:

        sent = sent + word + " "

    return sent



def judge(three_inf_sent_tens):
    
    criterion = nn.CrossEntropyLoss()
    max_sum = 0
    ind = 0
    best_ind = 0
    
    for inf_sent in three_inf_sent_tens:
        
        prob_sum = 0
        
        for word in inf_sent:
            
            prob, _ = torch.max(word, dim=-1)
            
            prob_sum += prob
            
        if prob_sum > max_sum:
            
            best_ind = ind
         
        ind += 1
    
    chosen_tens = three_inf_sent_tens[ind]
    
    return chosen_tens






dir_path = sys.argv[1]
out_file = sys.argv[2]

vocab, rev_vocab, vocab_len, _ = enc_dec_deluxe.vocab_function()

pre_cnn = enc_dec_deluxe.Pretrained_CNN()
encoder = enc_dec_deluxe.Encoder(10*2*2, 50, 1)
decoder = enc_dec_deluxe.Decoder(vocab_len, 50, 1)
mainModel = enc_dec_deluxe.Enc_Dec_Model(pre_cnn, encoder, decoder)

mainModel.load_state_dict(torch.load("encoder_decoder_model_deluxe.pth", weights_only = True))
mainModel.eval()

with open('id.txt', 'r') as file:
    for line in file:
        file_nm = line.strip()
        
        file_path = dir_path + "/" + file_nm
        
        in_tens = convert_avi(file_path)
        
        arr = in_tens.numpy()

        X = enc_dec_deluxe.padTensor(arr)
        
        three_inf_tens = mainModel(X, vocab_len)
        
        pred = judge(three_inf_tens)

        sent = pred_to_sent(pred, rev_vocab)

        with open('out_file', 'w') as file:
            
            out_line = file_nm +',' + sent +'\n'
            
            file.write(out_line)
        
    
       
    
