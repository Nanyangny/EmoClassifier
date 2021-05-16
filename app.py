import numpy as np
from flask import Flask, render_template,request,flash,redirect,jsonify,url_for
from transformers import BertTokenizer, BertForMaskedLM
import torch
import string
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import datetime
import time 
import spacy
import pandas as pd
import re
import os
import networkx as nx
from bert_predict import key_word_predict_with_network_from_sent,emo_predict




app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system('cmd /c "python -m spacy download en_core_web_sm"')
    nlp = spacy.load("en_core_web_sm")



########################################################################
"""
Input required
"""

## The Gelphi output file to construct the network
Gelphi_output_file_path = "./src/Gelphi_output_updated.csv" 
## trained model from the annotated dataset (can refer to the previous notebook repo link for training details)
emo_prediction_model_path = './src/emo-prediction_rft_model'
## edge list to construct the network (one of the input files for Gelphi)
edge_list_path = "./src/second_iteration_edge_list_updated.csv"


### parameters for the clustering algo

# how many prediction from BERT masked prediction model
top_k_bert_prediction = 5
# how many words within the labelled cluster to represent the cluster in the cluster assignment
top_k_words_in_cluster = 5 
# how many cluster to be displayed in the webapp frontend
top_k_clustering_display = 3

########################################################################




network = pd.read_csv(Gelphi_output_file_path)
network.set_index(network['Label'],inplace=True)

### constructing network using networkx for emo assignment
edge_list = pd.read_csv(edge_list_path)
Graphtype = nx.Graph()
syn_G = nx.from_pandas_edgelist(edge_list, edge_attr='weight', create_using=Graphtype)



network_for_emo_assign = network.set_index(network['index'],inplace=False)
mapping =network_for_emo_assign['Label'].to_dict()
G = nx.relabel_nodes(syn_G, mapping, copy=False)


selected_top = network_for_emo_assign.sort_values('Authority',ascending=False).groupby('modularity_class').head(top_k_words_in_cluster)
## generate dic with cluster number as key and representing words list as the value
selected_top_dic = selected_top.groupby('modularity_class')['Label'].agg(list).to_dict()
modularity_class = selected_top['modularity_class'].unique()


# bert model loading 
## there are other options as well (cased | uncased) refer to https://huggingface.co/transformers/model_doc/bert.html

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()


@app.route('/')
def test():
    return render_template('home.html')


#To use the predict button in our web-app
@app.route('/predict',methods=['GET','POST'])
def predict():
    #For rendering results on HTML GUI
    ## for GET Method 
    input_text = request.form['input_text']
    ## for POST Method
    output_html_array =[]
    # input_text = request.form.get('input_text')
    bert_output = key_word_predict_with_network_from_sent(input_sent = input_text,top_k=top_k_bert_prediction,network_input=network, spacy_model=nlp,tokenizer=bert_tokenizer,bert_model=bert_model)
    # bert_output.to_csv('./src/test_output.csv',index=False)
    emo_pred_res = emo_predict(emo_prediction_model_path,bert_output,top_cluster_word_dict=selected_top_dic,mod_class=modularity_class,EmoClusterNetwork=G)
    # print(emo_pred_res)
    if emo_pred_res:
        # print(emo_pred_res)
        emo_pred_words = list(emo_pred_res.keys())
        input_text_array = input_text.split(' ')
        for input_item in input_text_array:
                if emo_pred_words and input_item.startswith(emo_pred_words[0].split('_')[0]):
                    output_html_array.append("<span class='emo badge bg-secondary'>{}<span class='tooltiptext'>{}</span></span>".format(input_item,emo_pred_res[emo_pred_words[0]][:top_k_clustering_display]))
                    emo_pred_words.pop(0)
                else:
                    output_html_array.append(input_item)
        # print(' '.join(output_html_array))
        return render_template('home.html', prediction_text='Results: {}'.format(' '.join(output_html_array)))
    else:
        return render_template('home.html', prediction_text='Results: {}'.format(input_text))

   

if __name__ == "__main__":
    app.run(debug=True)