from transformers import BertTokenizer, BertForMaskedLM, pipeline
import torch
import pickle
import string
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import numpy as np
import datetime
import time 
import spacy
import pandas as pd
import re
import random
from scipy.special import softmax
# from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from networkx.algorithms.shortest_paths.generic import shortest_path_length as spl


# nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english')) 


# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

### Get the predicted words from  BERT model
def decode(tokenizer=None, pred_idx=None, top_clean=5):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return tokens[:top_clean]  ## each line one prediction

### Encode the words by BERT tokenizers
def encode(tokenizer=None, text_sentence='', add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx

###
def get_predictions(model=None,text_sentence='',top_clean=5,tokenizer=None):
    input_ids,mask_idx = encode(tokenizer,text_sentence)
    with torch.no_grad():
        predict= model(input_ids)[0]
    bert = decode(tokenizer,predict[0,mask_idx,:].topk(top_clean).indices.tolist(),top_clean=top_clean)
    return bert


"""
Text pre-processing functions
To split those cut words with come with punctuation e.g. "like.", ". enable"
"""

def punctuation_corr(input_sent):
    ## correct punctuation position
    input_split = input_sent.split()
    for i in range(len(input_split)):
        if not input_split[i]: ## for \t\n char 
            continue
        ## word starts with a punctuation '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' but not @ => @abc , may be a tweet mention
        if input_split[i][0] in string.punctuation and input_split[i][0]!='@':
            input_split[i]= input_split[i][0] + " " + input_split[i][1:]
        
        if input_split[i][-1] in string.punctuation:
            input_split[i]= input_split[i][:-1] + " " + input_split[i][-1]
            if input_split[i][-3] in string.punctuation: ## for ." 
                # orig_punc = input_split[i][-3]
                input_split[i]= input_split[i][:-3] +" "+input_split[i][-3:]
                
        ### account for all CAP words, convert to lower case
        elif input_split[i].upper() == input_split[i] and len(input_split[i])> 1 :  
            input_split[i] = input_split[i].lower()
        
        ## for punct in between the words without space e.g."buy,i"

        try:
            punc_pos = len(re.search('\w+',input_split[i])[0])
            if punc_pos<len(input_split[i])-1:
                if input_split[i][punc_pos] in string.punctuation and input_split[i][punc_pos] !='-' and input_split[i][punc_pos] !="'" :
                    input_split[i] = input_split[i][:punc_pos] + " " +input_split[i][punc_pos]+" "+input_split[i][punc_pos+1:]
        except:
            pass

    input_sent = ' '.join(input_split)
    return input_sent



"""
Based on the cleaned text, mask out interested words for BERT prediction and get the BERT prediction
Two ways to split the sentence, by Spacy or NLTk (Spacy is more advanced and time-consuming)

input_sent (string): the corpus for prediction
top_k (int): number of prediction get from BERT model for each masked word 
useSpacy (boolean): whether the use of Spacy to split words and pos tagging
stop_words (array)
"""

def find_masked_words(input_sent='',top_k=5, useSpacy=True,stop_words=None,spacy_model=None,pred_model=None,tokenizer=None):
    keyword = defaultdict(dict)
    if not useSpacy:
        input_sent = punctuation_corr(input_sent)
        input_split = input_sent.split()
            ## second filter with pos tagging for nltk
        ### POS TAGGGING 
        not_consider_pos = ['PRON','DET','ADP','CONJ','NUM','PRT','.',':','CC','CD','EX','DT','PDT',
                        'IN','LS','MD','NNP','NNPS','PRP','POS','PRP$','TO','UH','WDT','WP$','WP','WRB']
            #  refer to   https://www.learntek.org/blog/categorizing-pos-tagging-nltk-python/ 
        be_do_verb = ['is','are','was','were','did','does','do','not','had','have','has','ever']
        conjunction_words = ['therefore','thus']
        pos_res = nltk.pos_tag(input_sent.split())
        for item_num in range(len(pos_res)):
            if pos_res[item_num][1] not in not_consider_pos:
                ## another level of filtering of words before masking 
                if '@' in pos_res[item_num][0]:
                    continue
                if pos_res[item_num][0].lower() in be_do_verb:
                    continue
                if pos_res[item_num][0].lower() in conjunction_words:
                    continue
                if pos_res[item_num][0][-2:] in ["'s","'t","'r","'d","'l","'v","'m"]:
                    continue
                if 'http' in pos_res[item_num][0]:
                    continue
                if len(pos_res[item_num][0])<3:
                    continue
                if pos_res[item_num][0] in stop_words:
                    continue
                if pos_res[item_num][0][-1] in string.punctuation:
                    pos_res[item_num] = pos_res[item_num][:-1]
                orig = input_split[item_num]
                input_split[item_num]='<mask>'
                input_text_for_pred = ' '.join(input_split)
                input_split[item_num]=orig
                keyword[pos_res[item_num][0]+ "_"+str(item_num)]['prediction']=get_predictions(model=pred_model,text_sentence =input_text_for_pred, top_clean=top_k,tokenizer=tokenizer)
    else:
        # print(input_sent)
        input_sent = punctuation_corr(input_sent)
        doc = spacy_model(input_sent)
        input_split = doc.text.split()
        ### reg_exp to detect punctuation and number in the word splits
        reg_exp= "["+string.punctuation+"0-9]"
        for i in range(len(input_split)):
            
            if len(doc[i].text)<3: ## skip words with length < 3
                continue
            if re.search(reg_exp,doc[i].text): ## skip punctuation and number
                continue
            ### remove words that are definitely not emo-denoting for easier computation
            if not doc[i].is_stop and doc[i].pos_ not in ['SPACE','PUNCT','ADX','CONJ','CCONJ',
                                                        'DET','INTJ','NUM','PRON','PROPN','SCONJ','SYM']:
                orig = input_split[i]
                input_split[i]= "<mask>"
                input_text_for_pred = ' '.join(input_split) ### join the split words together with <mask> for BERT prediction
                input_split[i]= orig
                keyword[doc[i].text+ "_"+str(i)]['prediction']=get_predictions(model=pred_model,text_sentence=input_text_for_pred, top_clean=top_k,tokenizer=tokenizer)
            
        
    return keyword


# network = pd.read_csv(Gelphi_output_file_path)
# network.set_index(network['Label'],inplace=True)

"""
Auxilary functions to find out the prediction from BERT model, change top_k_choic

"""

### match score pertaining to the masked words with the network metrics/score
## match_col : "Authority" , "modularity_class","Weighted Degree","betweenesscentrality"
def self_score(match_col="Authority",pred_out_pf=None,network=None):
    return pred_out_pf['cleaned_index'].map(network[match_col].to_dict())

### match score pertaining to the masked predictions with the network metrics/score
## match_col : "Authority" , "modularity_class","Weighted Degree","betweenesscentrality"
def pred_score(match_col="Authority",pred_out=None,network=None):
    pred_score_output = []
    for item in pred_out['prediction']:
        item = item.lower()
        try:
            pred_score_output.append(network[match_col].to_dict()[item])
        except:
            pred_score_output.append(-1)
    return pred_score_output


### aggregate function
def key_word_predict_with_network_from_sent(input_sent=None,top_k=None, filter_NA_pred=True,network_input=None,stop_words=None,spacy_model=None,tokenizer=None,bert_model=None):
    # print(input_sent)
    keyword_pred_from_bert_output = find_masked_words(input_sent = input_sent, top_k=top_k,stop_words=stop_words,spacy_model=spacy_model,pred_model=bert_model,tokenizer=tokenizer)
    res_out = pd.DataFrame(keyword_pred_from_bert_output).transpose()
    res_out['cleaned_index']= [ item.split('_')[0].lower() for item in res_out.index]
    res_out['Label'] = self_score(match_col="Label",pred_out_pf=res_out,network=network_input)## check whether in the network
    res_out['self_auth'] = self_score(match_col="Authority",pred_out_pf=res_out,network=network_input)
    res_out['self_class'] = self_score(match_col="modularity_class",pred_out_pf=res_out,network=network_input)
    res_out['self_deg'] = self_score(match_col="Weighted Degree",pred_out_pf=res_out,network=network_input)
    res_out['self_betcent'] = self_score(match_col="betweenesscentrality",pred_out_pf=res_out,network=network_input)
    res_out['pred_betcent'] = res_out.apply(lambda row: pred_score(match_col="betweenesscentrality",pred_out=row,network=network_input),axis=1)
    res_out['pred_auth'] = res_out.apply(lambda row: pred_score(match_col="Authority",pred_out=row,network=network_input),axis=1)
    res_out['pred_deg'] = res_out.apply(lambda row: pred_score(match_col="Weighted Degree",pred_out=row,network=network_input),axis=1)
    res_out['pred_class'] = res_out.apply(lambda row: pred_score(match_col="modularity_class",pred_out=row,network=network_input),axis=1)
    return res_out

## calculate average auth score from the BERT predicted words
def cal_avg_pred_score(row):
    count = 0
    sum_ = 0
    for item in row:
        try:
            item = item.strip()
            if item != '-1':
                sum_ += float(item)
                count+= 1
        except:
            if item != -1:
                sum_ +=item
                count+=1
       
    if count==0:
        return 0
    else:
        return sum_/count


def emo_predict(model_file_dir='./emo-prediction_rft_model',bert_output ='',top_cluster_word_dict=dict(),mod_class=[],EmoClusterNetwork=''):
    ## used col in the model for prediciton
    input_col = ['self_auth','self_deg','self_betcent','avg_pred_auth','avg_pred_deg','avg_pred_betcent']
    loaded_model = pickle.load(open(model_file_dir, 'rb'))
    ## load bert_output
    # bert_output =pd.read_csv(bert_output)

        
    input_text = bert_output[['Label','self_auth','self_deg','self_betcent','pred_auth','pred_deg','pred_betcent']]
    for col in ['self_auth','self_deg','self_betcent']:
        input_text.loc[input_text['Label'].isna(),col]=-1
    ### convert string list to float list for further 
    # for col in ['pred_auth','pred_deg','pred_betcent']:
    #     input_text[col] = input_text[col].str.replace("'","",).str.replace("[","").str.replace("]","").str.split(',')
    ### col avg calculation 
    for col in ['pred_auth','pred_deg','pred_betcent']:
        input_text['avg_'+col]= input_text[col].map(lambda x: cal_avg_pred_score(x))
    
    result = loaded_model.predict(input_text[input_col])
  
    input_text[input_col].copy()
    input_text['emo?'] = result
    input_text['word'] = input_text.index
    ### emotion score distribution of all the clusters
    total_emo_res = input_text.apply(lambda x: emo_distribution_cal(top_cluster_word_dict=top_cluster_word_dict,network_Graph=EmoClusterNetwork, row=x,mod_class=mod_class),axis=1)
    
    input_text["emo_dist_prob"] = total_emo_res
    input_text['emo_dist_cluster_order']=input_text.apply(lambda x: emo_cluster_assignment(x,mod_class),axis=1)
    input_text['emo_dist_prob_sorted']=input_text.apply(lambda x:emo_prob_sort(x),axis=1)
    # res['textid']=input_text['from_textid']
    # res.to_csv(output_folder + emo_pred_file_name+".csv",index=False)
    # print(f'Emo-denoting Model prediction results are saved in {output_folder+emo_pred_file_name}.csv')
    res =input_text.loc[input_text['emo?']==1,['emo_dist_cluster_order']].to_dict()['emo_dist_cluster_order']
    # res = input_text.loc[input_text['emo?']==1,'word'].tolist()
    return res


"""
    output softmax function value for the probability of the emotion based on the score 
"""
def emo_distribution_cal(top_cluster_word_dict = dict(),source=None,network_Graph=None,row=None,mod_class=[]):
    emo_score_dist = []
    if row['emo?']:
        source=row['word'].split('_')[0]
    else:
        return 
    
    for key in mod_class:
        top_words = top_cluster_word_dict[key]
        total_distance = 0
        count =0
        ### loop for each top word in one cluster
        for top_word in top_words:
            # print(top_word)
            try: 
                """
                spl retrieves the shortest distance from the source to the top_word in the network
                """
                total_distance += spl(network_Graph,source,top_word)
                count+=1
                # print(count)
            except:
                """
                when either source or top_word in the one cluster not found in the network 
                or they are not reachable => just pass 
                """
                pass
        ### shorter distance between nodes => closer the relationship => 1/ avg_distance of the cluster
        if count !=0 and key not in ['UNKNOWN']:
            avg_distance = total_distance/count
            emo_score_dist.append(1/avg_distance) 
        else:
            emo_score_dist.append(0)

    return softmax(emo_score_dist)


def emo_cluster_assignment(row,mod_class=[]):
    if row['emo?']==1:
        print("testing")
        # print(row['emo_dist_prob'].argsort()[::-1])
        # print(mod_class[np.array(row['emo_dist_prob']).argsort()[::-1]])
        return mod_class[np.array(row['emo_dist_prob']).argsort()[::-1]]
    else:
        return 0

def emo_prob_sort(row):
    if row['emo?']==1:
        return sorted(row['emo_dist_prob'])[::-1]
    else:
        return 0

