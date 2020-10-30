# Code for the paper "How Relevant Are Selectional Preferences for Transformer-based Language Models?"

## Prerequisites

* Python 3
* Download the SP-10K corpus: ```git clone https://github.com/HKUST-KnowComp/SP-10K```
* transformers library: ```pip install transformers```

## Contents

* ```extract_sents_from_corpus.py```
Extract sentences from the ukWaC corpus. You don't need to run this script if you already have the sentence files. These sentence files can be shared privately upon request, for copyright reasons.
    **Arguments** (required): 
    - ```--corpus_dir``` the directory where the original corpus is (e.g. ukWaC corpus)
    - ```--sp_dir``` the directory where the SP-10K corpus is downloaded.
    
* ```bert_analyze_sents.py``` 
    Analyze sentences with BERT: probabilities, predictions, attention. You don't need to run this script if there is a _results/[type_phrase]_ directory (you can find it in osirim: https://jupyter-slurm.irit.fr/user/emetheni/lab/tree/BERT_SP10K_experiments/results). 
    **Arguments** (required): 
    - ```--type_phrases``` the type of syntactic relations to create analysis for
    - ```--sent_file``` the file with preprocessed sentences
    - (optional) ```--bert_model``` the model to be used, default is *bert-base-uncased*.
    - (optional) ```--type_attention``` the types of attention mask to be used, defauls is *standard* i.e. no mask. *context* is for blocking everything but the head word, *head* is to block the head word, *control* blocks everything.
    
## Citation

Metheniti, Eleni, Tim Van de Cruys and Nabil Hathout. (2020) How Relevant Are Selectional Preferences for Transformer-based Language Models? 28th International Conference on Computational Linguistics (COLING). (to appear)