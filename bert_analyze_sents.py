import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import os
from utils import bert_tokenize, make_segments_ids, read_phrases
from configure import parse_args
import json
import inflect
from tqdm import tqdm
from scipy.stats import kendalltau
import numpy as np

# Load arguments
args = parse_args()
type_phrases = args.type_phrases
type_attention = args.type_attention

# Load pre-trained model and BERT tokenizer (vocabulary)
modelpath = args.bert_model
tokenizer = BertTokenizer.from_pretrained(modelpath)
model = BertForMaskedLM.from_pretrained(modelpath, output_attentions=True)
model.cuda()
model.eval()

# Initialize softmax for BERT probabilities
sm_vector = torch.nn.Softmax(dim=0)
sm_matrix = torch.nn.Softmax(dim=1)
#torch.set_grad_enabled(False)

# Open all the sents and phrases
sents_file = args.

batch_size = 128
block_size = 100


def makeBatch(data, i, typeMask='standard'):
    currentData = data[i*batch_size:(i+1)*batch_size]
    currentData = [i.split('\t') for i in currentData]
    lefts = [i[1].split(' ')[0] for i in currentData] 
    rights = [i[1].split(' ')[1] for i in currentData] 
    masked_sents = [i[3].split(' ') for i in currentData] 
    scores = [float(i[4]) for i in currentData]
    masked_indices = [i.index("[MASK]") for i in masked_sents]
    head_indices = [int(i[-3]) for i in currentData] 
    indexed_tokens = [tokenizer.convert_tokens_to_ids(i) for i in masked_sents]
    [sequence.extend([tokenizer.pad_token_id] * (block_size - len(sequence))) for sequence in indexed_tokens]
    indexed_tokens = torch.LongTensor(indexed_tokens)
    attention_mask = createMask(indexed_tokens, head_indices, masked_indices, typeMask)

    return indexed_tokens, attention_mask, head_indices, masked_indices, lefts, rights, masked_sents, scores

def createMask(indexed_tokens, head_indices, masked_indices, typeMask):
    attention_mask = torch.ones_like(indexed_tokens)
    idx_pad_tokens = indexed_tokens == tokenizer.pad_token_id
    attention_mask[idx_pad_tokens] = 0
    if typeMask == 'standard':
        pass
    elif typeMask == 'head':
        #head needs to be 0 as well
        attention_mask[range(attention_mask.shape[0]), head_indices] = 0
    elif typeMask == 'context':
        #everything 0 except CLS, SEP, MASK, and head
        for i in range(attention_mask.shape[0]):
            for j in range(attention_mask.shape[1]):
                if not indexed_tokens[i,j] in (tokenizer.cls_token_id, tokenizer.sep_token_id):
                    attention_mask[i,j] = 0
        attention_mask[range(attention_mask.shape[0]), head_indices] = 1
        attention_mask[range(attention_mask.shape[0]), masked_indices] = 1
    elif typeMask == 'control':
        for i in range(attention_mask.shape[0]):
            for j in range(attention_mask.shape[1]):
                if not indexed_tokens[i,j] in (tokenizer.cls_token_id, tokenizer.sep_token_id):
                    attention_mask[i,j] = 0
        attention_mask[range(attention_mask.shape[0]), masked_indices] = 1
    else:
        raise ValueError('unknown attention mask')
    return attention_mask


allList = []
allOrig = []
with open(sents_file, "r", encoding="utf-8") as sentences:
    print("\nProcessing file: ", sents_file)
    allData = sentences.readlines()[1:]
    allOrig.extend(allData)
    n_batches = (len(allData) // batch_size) + 1
    for i in tqdm(range(n_batches)):
        batchx, maskx, head_indices, masked_indices, lefts, rights, masked_sents, scores = makeBatch(allData, i, typeMask=type_attention)
        batchx = batchx.cuda()
        maskx = maskx.cuda()

        with torch.no_grad():
            outputs = model(batchx, attention_mask=maskx)
        predictions = outputs[0]
        rightsi = tokenizer.convert_tokens_to_ids(rights)
        selected_logits = predictions[range(predictions.shape[0]),masked_indices]
        probabilities = sm_matrix(selected_logits)
        #attentions = outputs[-1]
        #attentionsx = get_mask_attention(head_indices, masked_indices, attentions)
        res = probabilities[range(probabilities.shape[0]),rightsi]
        r = list(zip(masked_sents, lefts, rights, res, scores))
        allList.extend(r)

prob_file = open(type_phrases + '_prob_results.txt', 'a')
    
#correlation over all sentences (micro)

real = [i[-1] for i in allList]
prob = [float(i[-2]) for i in allList]
lefts = [i[1] for i in allList]
print(type_phrases + '_' + type_attention + '\n\n')
print('correlation kendall-tau micro', kendalltau(real,prob))
print('average score: ' + str(sum(real)/len(real)) + '\n\n')
prob_file.write('\n\ntype_attention\t' + type_attention + '\n\n')
prob_file.write('correlation kendall-tau\t' + str(list(kendalltau(real,prob))) + '\n')
prob_file.write('average score: ' + str(sum(real)/len(real)) + '\n\n')

#correlation over average sentence probs (macro)
#first assemble in dict
allDict = {}
for i,j in enumerate(allOrig): 
    jList = j.split('\t') 
    if not jList[1] in allDict: 
        allDict[jList[1]] = ( [], float(jList[4])) 
    allDict[jList[1]][0].append(prob[i]) 
#then compute correlation
p_av = []
r_av = []
for k,v in allDict.items(): 
    p_av.append(sum(v[0]) / len(v[0])) 
    r_av.append(v[1]) 

print('average score: ', sum(r_av)/len(r_av))
print('correlation kendall-tau macro', kendalltau(r_av,p_av))
prob_file.write('correlation kendall-tau average\t' + str(list(kendalltau(r_av,p_av))) + '\n')
prob_file.write('average score: ' + str(sum(r_av)/len(r_av)) + '\n\n')
prob_file.flush()

