import json
import numpy as np
import re
import string
from collections import Counter
import argparse


def add_arguments(parser):
    parser.add_argument("--regional_answer", help="path to regional answer", required=True)
    parser.add_argument("--global_answer", help="path to global answer", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
  prediction_tokens = normalize_answer(prediction).split()
  ground_truth_tokens = normalize_answer(ground_truth).split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def cross_f1_max(predictions):
    cross_f1_max = []
    for i in range(len(predictions)):
        index = list(range(len(predictions)))
        index.pop(i)
        cross_f1_max.append(max([f1_score(predictions[i], predictions[j]) for j in index]))
    return cross_f1_max
def cross_f1_mean(predictions):
    cross_f1_mean = []
    for i in range(len(predictions)):
        index = list(range(len(predictions)))
        index.pop(i)
        cross_f1_mean.append(sum([f1_score(predictions[i], predictions[j]) for j in index])/len(index))
    return cross_f1_mean



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    with open(args.regional_answer,'r') as f:
        regional_answer = json.load(f)
    with open(args.global_answer,'r') as f:
        global_answer = json.load(f)
    
    cross_answer = {}
    delta = 0.1
    gamma = 0.8
    for (qid, answer),(_, g_answer) in zip(regional_answer.items(),global_answer.items()):
        score = [i['score']*gamma for i in answer][:10]
        text = [i['text'] for i in answer][:10]
        score1 = [i['score']*(1-gamma) for i in g_answer][:10]
        text1 = [i['text'] for i in g_answer][:10]
        score = score + score1
        text = text + text1
        cross_f1 = cross_f1_mean(text)
        score_list = [delta*i + (1-delta) *j for i,j in zip(score,cross_f1)]
        max_idx = np.argmax(score_list)
        cross_answer[qid] = text[max_idx]
    
    with open(args.output_file,'w') as f:
        json.dump(cross_answer,f)
