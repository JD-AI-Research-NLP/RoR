import json
from collections import defaultdict
import copy
import argparse

def add_arguments(parser):
    parser.add_argument("--regional_answer", help="path to regional_answer", required=True)
    parser.add_argument("--document_file", help="path to train document file", required=True)
    parser.add_argument("--evidence_file", help="path to evidence file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)
   
def get_index(text, item=''):
    return [i.start() for i in re.finditer(item, text)]


def is_overlapping(x1, x2, y1, y2):
    return max(x1, y1) <= min(x2, y2) 

def remove_overlap(positions):
    position_no_overlap = positions[:]
    all_no_overlap = 0
    for i in range(len(position_no_overlap)):
        for j in range(len(position_no_overlap)):
            if i != j and is_overlapping(position_no_overlap[i][0],position_no_overlap[i][1],position_no_overlap[j][0],position_no_overlap[j][1],):
                position_no_overlap.append((min(position_no_overlap[i][0],position_no_overlap[j][0]),max(position_no_overlap[i][1],position_no_overlap[j][1])))
                remove1 = position_no_overlap[i]
                remove2 = position_no_overlap[j]
                position_no_overlap.remove(remove1)
                position_no_overlap.remove(remove2)
                all_no_overlap = 1
                break
        if all_no_overlap == 1:
            break
    if all_no_overlap == 0 or len(positions) == 1:
        return positions
    else:
        return remove_overlap(position_no_overlap)

def pend_position(start,end,source):
    left = len(' '.join(source[:start].split(' ')[-6:])) 
    right = len(' '.join(source[end+1:].split(' ')[:6])) 
    left_pend = max(start-left, 0)
    right_pend = min(end+right, len(source))
    return left_pend, right_pend
    
def answer_to_text(answer, source):
    span_text = []
    start_position, end_position = 0,0
    for ans in answer:
        start_position = source.find(ans)
        if start_position == -1:
            continue
        else:
            end_position = start_position + len(ans) - 1
            
            if len(ans.split(' ')) < 6:
                start_position, end_position = pend_position(start_position, end_position, source)
            break
    

    position = [[start_position, end_position]]
    for ans in answer[1:]:
        start = source.find(ans)
        if start == -1:
            continue
        end = start + len(ans) - 1
        
        if len(ans.split(' ')) < 6:
            start, end = pend_position(start, end, source)
                
        match = 0
        for index in range(len(position)):
            if is_overlapping(start, end, position[index][0], position[index][1]):
                match =1
                if start < position[index][0] and end > position[index][1]:
                    position[index][0] = start
                    position[index][1] = end
                elif start < position[index][0] and end < position[index][1]:
                    position[index][0] = start                  
                elif start > position[index][0] and end > position[index][1]:
                    position[index][1] = end                  
                else :
                    pass
                break
            else:
                pass
        if match == 0:
            position.append([start,end])
    
    if len(position) == 1:
        position_no_overlap = position
    else:
        position_no_overlap = remove_overlap(position)
    
    if len(position_no_overlap) == 1 and position_no_overlap[0] == [0,0]:
        return ''
    
    span_text = [source[i:j+1] for (i,j) in position_no_overlap]
    return ' . '.join(span_text)

def pend_text(answer,source):
    answer = ' '.join(split_token(answer))
    start = source.find(answer)
    if start == -1:
        print(answer)
        print(source)
        raise ValueError("answer must in document")
    end = start + len(answer) - 1
    left_pend = ' '.join(source[:start].split(' ')[-6:])
    left_pend_position = source.find(left_pend)
    right_pend = ' '.join(source[end+1:].split(' ')[:6])
    right_pend_position = end + len(right_pend) 
    pend_answer = source[left_pend_position:right_pend_position+1]
    return pend_answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    with open(args.regional_answer,'r') as f:
        predictions = json.load(f)
    with open(args.document_file,'r') as f:
        train = json.load(f)['data']
    
    new_qid_text = {}
    for qid, pred in list(predictions.items()):
        id_to_text = defaultdict(list)
        for answer in pred:
            ids = answer['ids'].split('--')[1]
            ans = answer['text']
            id_to_text[ids].append(ans)
        converted_text = []
        for idss, anss in id_to_text.items():
            with open(args.evidence_file + idss,'r') as f:
                source_text = f.readlines()
            anss = [i.strip().lower() for i in anss]
            source_text = ' '.join(source_text).replace('\n','').lower()
            text = answer_to_text(anss,source_text)
            converted_text.append(text)
        new_text = ' # '.join(converted_text)
        new_qid_text[qid] = new_text.lower()
        
    new_train = []
    all_qid = []
    for example in train:
        qid = example["paragraphs"][0]['qas'][0]['qid']
    
        if qid not in new_qid_text:
            num1 += 1
            continue
        example["paragraphs"][0]['qas'][0]['answers'] = [{"text":'no answer',"answer_start":0}]
        
        if qid in all_qid:
            continue
        all_qid.append(qid)
        example["paragraphs"][0]['context'] = new_qid_text[qid]
        if new_qid_text[qid] == '':
            continue
        shadow = copy.deepcopy(example)
        new_train.append(shadow)
        
    with open(args.output_file,'w') as f:
        json.dump(new_train,f)
