import numpy as np
from flask import Flask, render_template,request,flash,redirect,jsonify,url_for
from werkzeug.utils import secure_filename

from transformers import BertTokenizer, BertForMaskedLM
import torch
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
import os
import networkx as nx
from bert_predict import key_word_predict_with_network_from_sent,emo_predict


# UPLOAD_FOLDER = './upload/'
# ALLOWED_EXTENSIONS = {'csv','xlsx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# model = pickle.load(open('model.pkl', 'rb'))

try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system('cmd /c "python -m spacy download en_core_web_sm"')
    nlp = spacy.load("en_core_web_sm")

# stop_words = set(stopwords.words('english')) 
Gelphi_output_file_path = "./src/Gelphi_output_updated.csv" 
emo_prediction_model_path = './src/emo-prediction_rft_model'
edge_list_path = "./src/second_iteration_edge_list_updated.csv"
node_list_path = "./src/second_iteration_nodes_updated.csv"
top_k_words_in_cluster = 5 
top_k_bert_prediction = 5
top_k_clustering = 3




network = pd.read_csv(Gelphi_output_file_path)
network.set_index(network['Label'],inplace=True)

### constructing network using networkx for emo assignment
edge_list = pd.read_csv(edge_list_path)
# node_list = pd.read_csv(node_list_path)
Graphtype = nx.Graph()
syn_G = nx.from_pandas_edgelist(edge_list, edge_attr='weight', create_using=Graphtype)



network_for_emo_assign = network.set_index(network['index'],inplace=False)
mapping =network_for_emo_assign['Label'].to_dict()
G = nx.relabel_nodes(syn_G, mapping, copy=False)


selected_top = network_for_emo_assign.sort_values('Authority',ascending=False).groupby('modularity_class').head(top_k_words_in_cluster)
## generate dic with cluster number as key and representing words list as the value
selected_top_dic = selected_top.groupby('modularity_class')['Label'].agg(list).to_dict()
modularity_class = selected_top['modularity_class'].unique()



bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def test():
    return render_template('home.html')


#To use the predict button in our web-app
@app.route('/predict',methods=['GET','POST'])
def predict():
    #For rendering results on HTML GUI
    # int_features = [x for x in request.form.values()]
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
                    output_html_array.append("<span class='emo badge bg-secondary'>{}<span class='tooltiptext'>{}</span></span>".format(input_item,emo_pred_res[emo_pred_words[0]][:top_k_clustering]))
                    emo_pred_words.pop(0)
                else:
                    output_html_array.append(input_item)
        # print(' '.join(output_html_array))
        return render_template('home.html', prediction_text='Results: {}'.format(' '.join(output_html_array)))
    else:
        return render_template('home.html', prediction_text='Results: {}'.format(input_text))

   



@app.route('/count')
def index_count():
    return render_template('count.html')


@app.route("/<any(plain, jquery, fetch):js>")
def index(js):
    return render_template("{0}.html".format(js), js=js)


@app.route("/add", methods=["POST"])
def add():
    a = request.form.get("a", 0, type=float)
    b = request.form.get("b", 0, type=float)
    return jsonify(result=a + b)


if __name__ == "__main__":
    app.run(debug=True)