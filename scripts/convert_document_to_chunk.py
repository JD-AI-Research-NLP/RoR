from transformers import RobertaTokenizer
import json
import torch
import copy
from multiprocessing import Pool
import os
import argparse



def add_arguments(parser):
    parser.add_argument("--document_file", help="path to input file", required=True)
    parser.add_argument("--chunk_file", help="path to output file", required=True)


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def procrss_chunk(index):
    print("%s start processing, qia is %d"%(i,os.getpid()))
    new_example = []
    for example in train[index*batch:min((index+1)*batch,train_length)]:

        for paragraph in example["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                answer_spans = []
                for answer in qa["answers"]:
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    try:
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        token_ids = tokenizer.encode(orig_answer_text)
                    except RuntimeError:
                        print(f'Reading example {idx} failed')
                        start_position = 0
                        end_position = 0
                    answer_spans.append({'start': start_position, 'end': end_position,
                                         'text': orig_answer_text, 'token_ids': token_ids})

                # ===== Given an example, convert it into tensors  =============
                query_tokens = tokenizer.tokenize(question_text)
                query_tokens = query_tokens[:max_question_len]
                tok_to_orig_index = []
                orig_to_tok_index = []
                all_doc_tokens = []
                for (i, token) in enumerate(doc_tokens):
                    orig_to_tok_index.append(len(all_doc_tokens))
                    sub_tokens = tokenizer.tokenize(f'. {token}')[1:] if i > 0 else tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tok_to_orig_index.append(i)
                        all_doc_tokens.append(sub_token)

                all_doc_tokens = all_doc_tokens[:max_doc_len]

                # The -3 accounts for [CLS], [SEP] and [SEP]
                max_tokens_per_doc_slice = max_seq_len - len(query_tokens) - 3
                assert max_tokens_per_doc_slice > 0

    #             stride = max_tokens_per_doc_slice - doc_stride
    #             if stride == 0:

                for slice_start in range(0, len(all_doc_tokens), 4000):
                    slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))
                    doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
                    if len(doc_slice_tokens) + len(query_tokens) + 3 < 20:
                        continue
                    text = tokenizer.convert_tokens_to_string(doc_slice_tokens)
                    #print(text)
                    example["paragraphs"][0]["context"] = text
                    shadow = copy.deepcopy(example)
                    #print(example["paragraphs"][0]["context"])
                    #print(example)
                    new_example.append(shadow)
    return new_example
                


if __name__ == "__main__":

    max_doc_len = 8000
    max_seq_len = 4096
    doc_stride = 96
    max_question_len = 55
    batch = 10000

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    with open(args.document_file,'r') as f: 
        train = json.load(f)['data']
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta.large') # roberta.large model 

    train_length = len(train)

    pool = Pool(processes=12) 
    results = []
    new_results = []
    for i in range(12):
        results.append(pool.apply_async(procrss_chunk, (i, )))
    pool.close() 
    pool.join() 
    print ("Sub-process(es) done.")
    for r in results:
        new_results.extend(r.get())
    with open(args.chunk_file,'w') as f:
        json.dump(new_results,f,indent=4)
