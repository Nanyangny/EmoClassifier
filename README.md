# EmoClassifier
A WebApp demo that shows the power of Graph Network and BERT Model in sentiment analysis.


For full documentation on the project, can refer to the [master project repo](https://github.com/Nanyangny/SYN-MLDA_Sentiment-Analysis)


## Instruction to run the webapp demo

1. install relevant python libraries using pip

`pip install -r requirements.txt`

 2. Updates the following inputs in the `app.py` 

`Gelphi_output_file_path` = "./src/Gelphi_output_updated.csv"

`emo_prediction_model_path` = './src/emo-prediction_rft_model'

`edge_list_path` = "./src/second_iteration_edge_list_updated.csv"

`top_k_words_in_cluster` = 5

`top_k_bert_prediction` = 5

`top_k_clustering` = 3

3. Run the webapp 

`python app.py`

## Features in the demo

- Enable user to input a text and predict the emo-denoting words from the text
- Obtain sentiment clustering of emo-denoting words

## Training models

- Emo prediction model

    The model is trained on labelled data (2k words for this case), the detailed training step can refer to the second notebook [here](https://github.com/Nanyangny/SYN-MLDA_Sentiment-Analysis/tree/main/notebooks).

- BERT model

    The current model used in the demo uses the official trained model on wikipedia and IMBD data, there are other options as well (cased | uncased) refer to [https://huggingface.co/transformers/model_doc/bert.html](https://huggingface.co/transformers/model_doc/bert.html).

    Extra training is required if the input text is from other domain (i.e Tweets).

