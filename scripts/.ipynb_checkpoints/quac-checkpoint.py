import sys 
sys.path.append("..") 
import os
from collections import defaultdict
import argparse
import json
import string
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, AutoModel, AutoConfig, AutoModelWithLMHead
from scripts.triviaqa_utils import evaluation_utils

import pytorch_lightning as pl
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

from longformer.longformer import Longformer
from longformer.sliding_chunks import pad_to_window_size


class TriviaQADataset(Dataset):
    """
    Largely based on
    https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/reading_comprehension/triviaqa.py
    and
    https://github.com/huggingface/transformers/blob/master/examples/run_squad.py
    """
    def __init__(self, file_path, tokenizer, max_seq_len, max_doc_len, doc_stride,
                 max_num_answers, ignore_seq_with_no_answers, max_question_len):
        assert os.path.isfile(file_path)
        self.file_path = file_path
        with open(self.file_path, "r", encoding='utf-8') as f:
            print(f'reading file: {self.file_path}')
            self.data_json = json.load(f)['data']
            print(f'done reading file: {self.file_path}')
        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.doc_stride = doc_stride
        self.max_num_answers = max_num_answers
        self.ignore_seq_with_no_answers = ignore_seq_with_no_answers
        self.max_question_len = max_question_len

        # A mapping from qid to an int, which can be synched across gpus using `torch.distributed`
        if 'train' not in self.file_path:  # only for the evaluation set
            self.val_qid_string_to_int_map =  \
                {
                    self._get_qid(entry["paragraphs"][0]['qas'][0]['id']): index
                    for index, entry in enumerate(self.data_json)
                }
        else:
            self.val_qid_string_to_int_map = None


    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        entry = self.data_json[idx]
        tensors_list = self.one_example_to_tensors(entry, idx)
        assert len(tensors_list) == 1
        return tensors_list[0]

    
    def _get_question_text(self,
                           history,
                           qas):
        question_tokens = ['<s>'] + qas["question"].split(' ')
        return " ".join(history + [" ".join(question_tokens)])    ####           history + <s> + question
    
    def _get_question_history(self,
                              history,
                              qas,
                              num_turn):
        question_tokens = []
        question_tokens.extend(['<s>'] + qas["question"].split(' '))
        question_tokens.extend(['</s>'] + qas["orig_answer"]["text"].split(' '))   ###      <s> + question + </s> + answer
        
        question_text = " ".join(question_tokens)
        if question_text:
            history.append(question_text)
        
        if num_turn >= 0 and len(history) > num_turn:
            history = history[-num_turn:]
        
        return history
    
    def _get_answer_span(self,
                         context,
                         qas,
                         no_answer):
        orig_text = qas["orig_answer"]["text"].lower()
        answer_start = qas["orig_answer"]["answer_start"]
        
        if no_answer or not orig_text or answer_start < 0:
            return "", -1, -1
        
        answer_end = answer_start + len(orig_text) - 1     # string end !!!!!!!!!! not word length
        answer_text = context[answer_start:answer_end + 1].lower()
        
        assert orig_text == answer_text
        answer_text = context[answer_start:answer_end + 1]
        
        return answer_text, answer_start, answer_end    
    
    
    
    def one_example_to_tensors(self, example, idx):
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False
        tensors_list = []
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
            
            question_history = []
            for qas in paragraph["qas"]:
                
                question_text = self._get_question_text(question_history, qas)
                question_history = self._get_question_history(question_history, qas, 2)

#                 no_answer = (qas["orig_answer"]["text"] == "CANNOTANSWER")
                orig_answer_text, start_position, _ = self._get_answer_span(paragraph_text, qas, no_answer)
                char_answer_length = len(orig_answer_text)
                answer_start_position = char_to_word_offset[start_position]
                answer_end_position = char_to_word_offset[start_position + char_answer_length -1]

                # ===== Given an example, convert it into tensors  =============
                query_tokens = self.tokenizer.tokenize(question_text)
                query_tokens = query_tokens[:self.max_question_len]
                tok_to_orig_index = []
                orig_to_tok_index = []
                all_doc_tokens = []
                for (i, token) in enumerate(doc_tokens):
                    orig_to_tok_index.append(len(all_doc_tokens))
                    # hack: the line below should have been `self.tokenizer.tokenize(token')`
                    # but roberta tokenizer uses a different subword if the token is the beginning of the string
                    # or in the middle. So for all tokens other than the first, simulate that it is not the first
                    # token by prepending a period before tokenizing, then dropping the period afterwards
                    sub_tokens = self.tokenizer.tokenize(f'. {token}')[1:] if i > 0 else self.tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tok_to_orig_index.append(i)
                        all_doc_tokens.append(sub_token)

                all_doc_tokens = all_doc_tokens[:self.max_doc_len]

                # The -3 accounts for [CLS], [SEP] and [SEP]
                max_tokens_per_doc_slice = self.max_seq_len - len(query_tokens) - 3
                assert max_tokens_per_doc_slice > 0
                if self.doc_stride < 0:
                    # negative doc_stride indicates no sliding window, but using first slice
                    self.doc_stride = -100 * len(all_doc_tokens)  # large -ve value for the next loop to execute once
                input_ids_list = []
                input_mask_list = []
                segment_ids_list = []
                start_positions_list = []
                end_positions_list = []
                for slice_start in range(0, len(all_doc_tokens), max_tokens_per_doc_slice - self.doc_stride):
                    slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))

                    doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
                    tokens = [self.tokenizer.cls_token] + query_tokens + [self.tokenizer.sep_token] \
                                                        + doc_slice_tokens + [self.tokenizer.sep_token]
                    segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_slice_tokens) + 1)
                    assert len(segment_ids) == len(tokens)

                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    input_mask = [1] * len(input_ids)

                    if self.doc_stride >= 0:  # no need to pad if document is not strided
                        # Zero-pad up to the sequence length.
                        padding_len = self.max_seq_len - len(input_ids)
                        input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
                        input_mask.extend([0] * padding_len)
                        segment_ids.extend([0] * padding_len)

                        assert len(input_ids) == self.max_seq_len
                        assert len(input_mask) == self.max_seq_len
                        assert len(segment_ids) == self.max_seq_len

                    doc_offset = len(query_tokens) + 2 - slice_start
                    
                    tok_start_position_in_doc = orig_to_tok_index[start_position]
                    not_end_of_doc = int(end_position + 1 < len(orig_to_tok_index))
                    tok_end_position_in_doc = orig_to_tok_index[end_position + not_end_of_doc] - not_end_of_doc                        
                    if tok_start_position_in_doc < slice_start or tok_end_position_in_doc > slice_end:
                    # this answer is outside the current slice
                        continue
                    start_positions = [tok_start_position_in_doc + doc_offset]
                    end_positions = [tok_end_position_in_doc + doc_offset]

                    input_ids_list.append(input_ids)
                    input_mask_list.append(input_mask)
                    segment_ids_list.append(segment_ids)
                    start_positions_list.append(start_positions)
                    end_positions_list.append(end_positions)

                tensors_list.append((torch.tensor(input_ids_list), torch.tensor(input_mask_list),
                                     torch.tensor(segment_ids_list),
                                     torch.tensor(start_positions_list), torch.tensor(end_positions_list),
                                     qas["id"]))  # for eval
        return tensors_list

    def _get_qid(self, qid):
        """all input qids are formatted uniqueID__evidenceFile, but for wikipedia, qid = uniqueID,
        and for web, qid = uniqueID__evidenceFile. This function takes care of this conversion.
        """
        if 'wikipedia' in self.file_path:
            # for evaluation on wikipedia, every question has one answer even if multiple evidence documents are given
            return qid.split('--')[0]
        elif 'web' in self.file_path:
            # for evaluation on web, every question/document pair have an answer
            return qid
        elif 'sample' in self.file_path:
            return qid
        else:
            raise RuntimeError('Unexpected filename')

    @staticmethod
    def collate_one_doc_and_lists(batch):
        num_metadata_fields = 1  # qids and aliases
        fields = [x for x in zip(*batch)]
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        # always use batch_size=1 where each batch is one document
        # will use grad_accum to increase effective batch size
        assert len(batch) == 1
        fields_with_batch_size_one = [f[0] for f in stacked_fields]
        return fields_with_batch_size_one


class TriviaQA(pl.LightningModule):

    def __init__(self, args):
        super(TriviaQA, self).__init__()
        self.args = args
        self.hparams = args

        self.tokenizer = RobertaTokenizer.from_pretrained('/home/user31/notespace/longformer-master/roberta.large')
        self.tokenizer.model_max_length = self.args.max_seq_len
        self.model = self.load_model()
        self.num_labels = 2
        if not self.args.seq2seq:
            self.qa_outputs = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None

    def load_model(self):
        if 'longformer' in self.args.model_path:
            model = Longformer.from_pretrained(self.args.model_path)
            for layer in model.encoder.layer:
                layer.attention.self.attention_mode = self.args.attention_mode
                self.args.attention_window = layer.attention.self.attention_window
        print("Loaded model with config:")
        print(model.config)

        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        return model

    def forward(self, input_ids, attention_mask, segment_ids, start_positions, end_positions):
        if 'longformer' in self.args.model_path:
            question_end_index = self._get_question_end_index(input_ids)
            # Each batch is one document, and each row of the batch is a chunck of the document.
            # Make sure all rows have the same question length.
            assert (question_end_index[0].float() == question_end_index.float().mean()).item()

            # local attention everywhere
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
            # global attention for the question tokens
            attention_mask[:, :question_end_index.item()] = 2

            # sliding_chunks implemenation of selfattention requires that seqlen is multiple of window size
            input_ids, attention_mask = pad_to_window_size(
                input_ids, attention_mask, self.args.attention_window, self.tokenizer.pad_token_id)

            sequence_output = self.model(
                    input_ids,
                    attention_mask=attention_mask)[0]

            # The pretrained TriviaQA model wasn't trained with padding, so remove padding tokens
            # before computing loss and decoding.
            padding_len = input_ids[0].eq(self.tokenizer.pad_token_id).sum()
            if padding_len > 0:
                sequence_output = sequence_output[:, :-padding_len]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            start_positions = start_positions[:, 0:1]
            end_positions = end_positions[:, 0:1]
            start_loss = loss_fct(start_logits, start_positions[:, 0])
            end_loss = loss_fct(end_logits, end_positions[:, 0])

            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


    def training_step(self, batch, batch_nb):
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, qids,  = batch
        output = self.forward(input_ids, input_mask, segment_ids, subword_starts, subword_ends)
        loss = output[0]
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss, 'lr': lr,
                            'input_size': input_ids.numel(),
                            'mem': torch.cuda.memory_allocated(input_ids.device) / 1024 ** 3}
        return {'loss': loss, 'log': tensorboard_logs}


    def _get_question_end_index(self, input_ids):
        eos_token_indices = (input_ids == self.tokenizer.eos_token_id).nonzero()
        assert eos_token_indices.ndim == 2
        assert eos_token_indices.size(0) == 2 * input_ids.size(0)
        assert eos_token_indices.size(1) == 2
        return eos_token_indices.view(input_ids.size(0), 2, 2)[:, 0, 1]

    def decode(self, input_ids, start_logits, end_logits):
        # find beginning of document
        question_end_index = self._get_question_end_index(input_ids)

        # bsz x seqlen => bsz x n_best_size
        start_logits_indices = start_logits.topk(k=self.args.n_best_size, dim=-1).indices
        end_logits_indices = end_logits.topk(k=self.args.n_best_size, dim=-1).indices

        answers = []
        # This loop can't be vectorized, so loop over each example in the batch separetly
        for i in range(start_logits_indices.size(0)):  # bsz
            potential_answers = []
            for start_logit_index in start_logits_indices[i]:  # n_best_size
                for end_logit_index in end_logits_indices[i]:  # n_best_size
                    if start_logit_index <= question_end_index[i]:
                        continue
                    if end_logit_index <= question_end_index[i]:
                        continue
                    if start_logit_index > end_logit_index:
                        continue
                    answer_len = end_logit_index - start_logit_index + 1
                    if answer_len > self.args.max_answer_length:
                        continue
                    potential_answers.append({'start': start_logit_index, 'end': end_logit_index,
                                              'start_logit': start_logits[i][start_logit_index].item(),
                                              'end_logit': end_logits[i][end_logit_index].item()})
            sorted_answers = sorted(potential_answers, key=lambda x: (x['start_logit'] + x['end_logit']), reverse=True)
            if len(sorted_answers) == 0:
                answers.append({'text': 'NoAnswerFound', 'score': -1000000})
            else:
                answer = sorted_answers[0]
                answer_token_ids = input_ids[i, answer['start']: answer['end'] + 1]
                answer_tokens = self.tokenizer.convert_ids_to_tokens(answer_token_ids.tolist())
                text = self.tokenizer.convert_tokens_to_string(answer_tokens)
                score = answer['start_logit'] + answer['end_logit']
                answers.append({'text': text, 'score': score})
        return answers


    def configure_optimizers(self):
        def lr_lambda(current_step):
            if current_step < self.args.warmup:
                return float(current_step) / float(max(1, self.args.warmup))
            return max(0.0, float(self.args.steps - current_step) / float(max(1, self.args.steps - self.args.warmup)))
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @pl.data_loader
    def train_dataloader(self):
        if self.train_dataloader_object is not None:
            return self.train_dataloader_object
        dataset = TriviaQADataset(file_path=self.args.train_dataset, tokenizer=self.tokenizer,
                                  max_seq_len=self.args.max_seq_len, max_doc_len=self.args.max_doc_len,
                                  doc_stride=self.args.doc_stride,
                                  max_num_answers=self.args.max_num_answers,
                                  max_question_len=self.args.max_question_len,
                                  ignore_seq_with_no_answers=self.args.ignore_seq_with_no_answers)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if self.trainer.use_ddp else None
        dl = DataLoader(dataset, batch_size=1, shuffle=(sampler is None),
                        num_workers=self.args.num_workers, sampler=sampler,
                        collate_fn=TriviaQADataset.collate_one_doc_and_lists)
        self.train_dataloader_object = dl
        return self.train_dataloader_object



    def configure_ddp(self, model, device_ids):
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--save_dir", type=str, default='triviaqa')
        parser.add_argument("--save_prefix", type=str, required=True)
        parser.add_argument("--train_dataset", type=str, required=False, help="Path to the training squad-format")
        parser.add_argument("--dev_dataset", type=str, required=True, help="Path to the dev squad-format")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
        parser.add_argument("--gpus", type=int, default=1,
                            help="Number of gpus. 0 for CPU")
        parser.add_argument("--warmup", type=int, default=200, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=0.0001, help="Maximum learning rate")
        parser.add_argument("--val_every", type=float, default=1.0, help="Number of training steps between validations")
        parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
        parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
        parser.add_argument("--max_seq_len", type=int, default=4096,
                            help="Maximum length of seq passed to the transformer model")
        parser.add_argument("--max_doc_len", type=int, default=4096,
                            help="Maximum number of wordpieces of the input document")
        parser.add_argument("--max_num_answers", type=int, default=64,
                            help="Maximum number of answer spans per document (64 => 94%)")
        parser.add_argument("--max_question_len", type=int, default=55,
                            help="Maximum length of the question")
        parser.add_argument("--doc_stride", type=int, default=-1,
                            help="Overlap between document chunks. Use -1 to only use the first chunk")
        parser.add_argument("--ignore_seq_with_no_answers", action='store_true',
                            help="each example should have at least one answer. Default is False")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--n_best_size", type=int, default=20,
                            help="Number of answer candidates. Used at decoding time")
        parser.add_argument("--max_answer_length", type=int, default=30,
                            help="maximum num of wordpieces/answer. Used at decoding time")
        parser.add_argument("--regular_softmax_loss", action='store_true',
                            help="IF true, use regular softmax. Default is using ORed softmax loss")
        parser.add_argument("--test", action='store_true', help="Test only, no training")
        parser.add_argument("--model_path", type=str, required=True,
                            help="Path to the checkpoint directory")
        parser.add_argument("--no_progress_bar", action='store_true', help="no progress bar. Good for printing")
        parser.add_argument("--attention_mode", type=str, choices=['tvm', 'sliding_chunks'],
                            default='sliding_chunks', help='Which implementation of selfattention to use')
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--seq2seq", action='store_true', help="Use an answer generation model")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")


        return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = TriviaQA(args)

    logger = TestTubeLogger(
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=0  # always use version=0
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
        save_top_k=5,
        verbose=True,
        monitor='avg_val_loss',
        # save_last=True,
        mode='min',
        period=-1,
        prefix=''
    )

    print(args)
    train_set_size = 83568  # hardcode dataset size. Needed to compute number of steps for the lr scheduler
    args.steps = args.epochs * train_set_size / (args.batch_size * max(args.gpus, 1))
    print(f'>>>>>>> #steps: {args.steps}, #epochs: {args.epochs}, batch_size: {args.batch_size * args.gpus} <<<<<<<')

    trainer = pl.Trainer(gpus=args.gpus, distributed_backend='ddp' if args.gpus and args.gpus > 1 else None,
                         track_grad_norm=-1, max_epochs=args.epochs, early_stop_callback=None,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.batch_size,
                         val_check_interval=args.val_every,
                         num_sanity_val_steps=2,
                         # check_val_every_n_epoch=2,
                         val_percent_check=args.val_percent_check,
                         test_percent_check=args.val_percent_check,
                         logger=logger if not args.disable_checkpointing else False,
                         checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
                         show_progress_bar=not args.no_progress_bar,
                         use_amp=not args.fp32, amp_level='O2',
                         resume_from_checkpoint=args.resume_ckpt,
                         )
    if not args.test:
        trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="triviaQa")
    parser = TriviaQA.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)
