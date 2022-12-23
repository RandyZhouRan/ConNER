# 
# @author: Allan
#

from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import PreTrainedTokenizer
import collections
import numpy as np
from src.data.data_utils import convert_iobes, build_label_idx, check_all_labels_in_dict

from src.data import Instance

Feature = collections.namedtuple('Feature', 'input_ids attention_mask token_type_ids orig_to_tok_index word_seq_len label_ids align_ids')
Feature.__new__.__defaults__ = (None,) * 7

def convert_instances_to_feature_tensors_max( instances: List[Instance],
                                         tokenizer: PreTrainedTokenizer,
                                         label2idx: Dict[str, int], max_length:int = 128) -> List[Feature]:
    features = []
    new_instances = []
    for idx, inst in enumerate(instances):
        words = inst.ori_words
        labels = inst.labels
        aligns = inst.aligns
        new_words = []
        new_labels = []
        new_aligns = []
        orig_to_tok_index = []
        tokens = []
        label_ids = []
        align_ids = []
        for i, (word, label, align) in enumerate(zip(words, labels, aligns)):
            orig_to_tok_index.append(len(tokens))
            label_ids.append(label2idx[label])
            align_ids.append(align)
            new_words.append(word)
            new_labels.append(label)
            new_aligns.append(align)
            word_tokens = tokenizer.tokenize(" " + word)
            if len(tokens) + len(word_tokens) > max_length:
                
                break # Discard the 2nd half of over-length sentences
                
                input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
                input_mask = [1] * len(input_ids)
                segment_ids = [0] * len(input_ids)
                features.append(Feature(input_ids=input_ids,
                                        attention_mask=input_mask,
                                        orig_to_tok_index=orig_to_tok_index,
                                        token_type_ids=segment_ids,
                                        word_seq_len=len(orig_to_tok_index),
                                        label_ids=label_ids,
                                        align_ids=align_ids))
                new_instances.append(Instance(words=new_words, ori_words=new_words, labels=new_labels, aligns=new_aligns))
                
                new_words = []
                new_labels = []
                new_aligns = []
                orig_to_tok_index = []
                tokens = []
                label_ids = []
                align_ids = []
                
                for sub_token in word_tokens:
                    tokens.append(sub_token)
                orig_to_tok_index.append(len(tokens))
                label_ids.append(label2idx[label])
                align_ids.append(align)
                new_words.append(word)
                new_labels.append(label)
                new_aligns.append(align)
            else:
                for sub_token in word_tokens:
                    tokens.append(sub_token)
        input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)


        features.append(Feature(input_ids=input_ids,
                                attention_mask=input_mask,
                                orig_to_tok_index=orig_to_tok_index,
                                token_type_ids=segment_ids,
                                word_seq_len=len(orig_to_tok_index),
                                label_ids=label_ids,
                                align_ids=align_ids))
        new_instances.append((Instance(words=new_words, ori_words=new_words, labels=new_labels, aligns=new_aligns)))
    return features, new_instances


#def convert_instances_to_feature_tensors(instances: List[Instance],
#                                         tokenizer: PreTrainedTokenizer,
#                                         label2idx: Dict[str, int]) -> List[Feature]:
#    features = []
#    # max_candidate_length = -1
#
#    for idx, inst in enumerate(instances):
#        words = inst.ori_words
#        orig_to_tok_index = []
#        tokens = []
#        for i, word in enumerate(words):
#            """
#            Note: by default, we use the first wordpiece token to represent the word
#            If you want to do something else (e.g., use last wordpiece to represent), modify them here.
#            """
#            orig_to_tok_index.append(len(tokens))
#            ## tokenize the word into word_piece / BPE
#            ## NOTE: adding a leading space is important for BART/GPT/Roberta tokenization.
#            ## Related GitHub issues:
#            ##      https://github.com/huggingface/transformers/issues/1196
#            ##      https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py#L38-L56
#            ##      https://github.com/ThilinaRajapakse/simpletransformers/issues/458
#            word_tokens = tokenizer.tokenize(" " + word)
#            for sub_token in word_tokens:
#                tokens.append(sub_token)
#        labels = inst.labels
#        label_ids = [label2idx[label] for label in labels] if labels else None
#        input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
#        segment_ids = [0] * len(input_ids)
#        input_mask = [1] * len(input_ids)
#
#        features.append(Feature(input_ids=input_ids,
#                                attention_mask=input_mask,
#                                orig_to_tok_index=orig_to_tok_index,
#                                token_type_ids=segment_ids,
#                                word_seq_len=len(orig_to_tok_index),
#                                label_ids=label_ids))
#    return features
#

# Default Dataset
class TransformersNERDataset(Dataset):

    def __init__(self, file: str,
                 tokenizer: PreTrainedTokenizer,
                 is_train: bool, 
                 label2idx: Dict[str, int] = None,
                 number: int = -1):
        """
        Read the dataset into Instance
        """
        ## read all the instances. sentences and labels
        insts = self.read_txt(file=file, number=number)
        self.insts = insts
        if is_train:
            print(f"[Data Info] Using the training set to build label index")
            assert label2idx is None
            ## build label to index mapping. e.g., B-PER -> 0, I-PER -> 1
            idx2labels, label2idx = build_label_idx(insts)
            self.idx2labels = idx2labels
            self.label2idx = label2idx
        else:
            assert label2idx is not None ## for dev/test dataset we don't build label2idx
            self.label2idx = label2idx
            check_all_labels_in_dict(insts=insts, label2idx=self.label2idx)
        #self.insts_ids = convert_instances_to_feature_tensors(insts, tokenizer, label2idx)
        self.insts_ids, self.insts = convert_instances_to_feature_tensors_max(insts, tokenizer, label2idx, max_length=128)
        self.tokenizer = tokenizer

    def read_txt(self, file: str, number: int = -1) -> List[Instance]:
        print(f"[Data Info] Reading file: {file}, labels will be converted to IOBES encoding")
        print(f"[Data Info] Modify src/data/transformers_dataset.read_txt function if you have other requirements")
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            ori_words = []
            labels = []
            aligns = []
            entity_counter = 0
            prev_label = ''
            
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    labels = convert_iobes(labels)
                    insts.append(Instance(words=words, ori_words=ori_words, labels=labels, aligns=aligns))
                    words = []
                    ori_words = []
                    labels = []
                    aligns = []
                    entity_counter = 0
                    prev_label = ''
                    if len(insts) == number:
                        break
                    continue
                ls = line.split()
                word, label = ls[0],ls[-1]
                
                # word alignment for translation
                if len(ls) > 2: # extract alignment if input is more than two column
                    align = int(ls[1])
                elif label == 'O':
                    align = 0
                else:
                    if label[2:] != prev_label[2:]:
                        entity_counter +=1
                    align = entity_counter
                aligns.append(align)
                prev_label = label
                
                ori_words.append(word)
                words.append(word)
                labels.append(label)
        print("number of sentences: {}".format(len(insts)))
        return insts

    def __len__(self):
        return len(self.insts_ids)

    def __getitem__(self, index):
        return self.insts_ids[index]

    def collate_fn(self, batch:List[Feature]):
        word_seq_len = [len(feature.orig_to_tok_index) for feature in batch]
        max_seq_len = max(word_seq_len)
        max_wordpiece_length = max([len(feature.input_ids) for feature in batch])
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature.input_ids)
            input_ids = feature.input_ids + [self.tokenizer.pad_token_id] * padding_length
            mask = feature.attention_mask + [0] * padding_length
            type_ids = feature.token_type_ids + [self.tokenizer.pad_token_type_id] * padding_length
            padding_word_len = max_seq_len - len(feature.orig_to_tok_index)
            orig_to_tok_index = feature.orig_to_tok_index + [0] * padding_word_len
            label_ids = feature.label_ids + [0] * padding_word_len
            align_ids = feature.align_ids + [0] * padding_word_len
            
            batch[i] = Feature(input_ids=np.asarray(input_ids),
                               attention_mask=np.asarray(mask), token_type_ids=np.asarray(type_ids),
                               orig_to_tok_index=np.asarray(orig_to_tok_index),
                               word_seq_len =feature.word_seq_len,
                               label_ids=np.asarray(label_ids),
                               align_ids=np.asarray(align_ids))
        results = Feature(*(default_collate(samples) for samples in zip(*batch)))
        return results

    
# Feature for both unlabeled and translated data
FeatureUDA = collections.namedtuple('FeatureUDA', 'input_ids attention_mask token_type_ids orig_to_tok_index word_seq_len label_ids align_ids input_ids2 attention_mask2 token_type_ids2 orig_to_tok_index2 word_seq_len2 label_ids2 align_ids2')
FeatureUDA.__new__.__defaults__ = (None,) * 14
        
# Dataset for unlabeled data and its translation

class TransformersNERDatasetUDA(Dataset):

    def __init__(self, unlabel_file: str, trans_file: str,
                 tokenizer: PreTrainedTokenizer,
                 is_train: bool,
                 label2idx: Dict[str, int] = None,
                 number: int = -1):
        """
        Read the dataset into Instance
        """
        ## read all the instances. sentences and labels
        unlabel_insts = self.read_txt(file=unlabel_file, number=number)
        trans_insts = self.read_txt(file=trans_file, number=number)
        self.unlabel_insts = unlabel_insts
        self.trans_insts = trans_insts
        if is_train:
            assert False # Never create new idx2label
            print(f"[Data Info] Using the training set to build label index")
            assert label2idx is None
            ## build label to index mapping. e.g., B-PER -> 0, I-PER -> 1
            idx2labels, label2idx = build_label_idx(insts)
            self.idx2labels = idx2labels
            self.label2idx = label2idx
        else:
            assert label2idx is not None ## for dev/test dataset we don't build label2idx
            self.label2idx = label2idx
            check_all_labels_in_dict(insts=unlabel_insts, label2idx=self.label2idx)
            check_all_labels_in_dict(insts=trans_insts, label2idx=self.label2idx)
        #self.insts_ids = convert_instances_to_feature_tensors(insts, tokenizer, label2idx)
        self.unlabel_insts_ids, self.unlabel_insts = convert_instances_to_feature_tensors_max(unlabel_insts, tokenizer, label2idx, max_length=128)
        self.trans_insts_ids, self.trans_insts = convert_instances_to_feature_tensors_max(trans_insts, tokenizer, label2idx, max_length=128)
#         print("len(self.unlabel_insts_ids)", len(self.unlabel_insts_ids))
#         print("len(self.trans_insts_ids)", len(self.trans_insts_ids))
        assert len(self.unlabel_insts_ids) == len(self.trans_insts_ids)
        self.insts_ids = []
        for unlabel, trans in zip(self.unlabel_insts_ids, self.trans_insts_ids):
            self.insts_ids.append(FeatureUDA(
                                input_ids=unlabel.input_ids,
                                attention_mask=unlabel.attention_mask,
                                orig_to_tok_index=unlabel.orig_to_tok_index,
                                token_type_ids=unlabel.token_type_ids,
                                word_seq_len=unlabel.word_seq_len,
                                label_ids=unlabel.label_ids,
                                align_ids=unlabel.align_ids,
                                input_ids2=trans.input_ids,
                                attention_mask2=trans.attention_mask,
                                orig_to_tok_index2=trans.orig_to_tok_index,
                                token_type_ids2=trans.token_type_ids,
                                word_seq_len2=trans.word_seq_len,
                                label_ids2=trans.label_ids,
                                align_ids2=trans.align_ids))
        self.tokenizer = tokenizer

    def read_txt(self, file: str, number: int = -1) -> List[Instance]:
        print(f"[Data Info] Reading file: {file}, labels will be converted to IOBES encoding")
        print(f"[Data Info] Modify src/data/transformers_dataset.read_txt function if you have other requirements")
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            ori_words = []
            labels = []
            aligns = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    labels = convert_iobes(labels)
                    insts.append(Instance(words=words, ori_words=ori_words, labels=labels, aligns=aligns))
                    words = []
                    ori_words = []
                    labels = []
                    aligns = []
                    if len(insts) == number:
                        break
                    continue
                ls = line.split()
                word, label = ls[0],ls[-1]
                
                # word alignment for translation
                if len(ls) > 2: # extract alignment if input is more than two column
                    align = int(ls[1])
                else:
                    align = 0 # default 0 if input is two column
                aligns.append(align)
                
                ori_words.append(word)
                words.append(word)
                labels.append(label)
        print("number of sentences: {}".format(len(insts)))
        return insts

    def __len__(self):
        return len(self.insts_ids)

    def __getitem__(self, index):
        return self.insts_ids[index]

    def collate_fn(self, batch:List[FeatureUDA]):
        # Unlabeled data
        word_seq_len = [len(feature.orig_to_tok_index) for feature in batch]
        max_seq_len = max(word_seq_len)
        max_wordpiece_length = max([len(feature.input_ids) for feature in batch])
        
        # Trans data
        word_seq_len2 = [len(feature.orig_to_tok_index2) for feature in batch]
        max_seq_len2 = max(word_seq_len2)
        max_wordpiece_length2 = max([len(feature.input_ids2) for feature in batch])
        
        for i, feature in enumerate(batch):
            # Unlabeled data
            padding_length = max_wordpiece_length - len(feature.input_ids)
            input_ids = feature.input_ids + [self.tokenizer.pad_token_id] * padding_length
            mask = feature.attention_mask + [0] * padding_length
            type_ids = feature.token_type_ids + [self.tokenizer.pad_token_type_id] * padding_length
            padding_word_len = max_seq_len - len(feature.orig_to_tok_index)
            orig_to_tok_index = feature.orig_to_tok_index + [0] * padding_word_len
            label_ids = feature.label_ids + [0] * padding_word_len
            align_ids = feature.align_ids + [0] * padding_word_len
            
            # Trans data
            padding_length2 = max_wordpiece_length2 - len(feature.input_ids2)
            input_ids2 = feature.input_ids2 + [self.tokenizer.pad_token_id] * padding_length2
            mask2 = feature.attention_mask2 + [0] * padding_length2
            type_ids2 = feature.token_type_ids2 + [self.tokenizer.pad_token_type_id] * padding_length2
            padding_word_len2 = max_seq_len2 - len(feature.orig_to_tok_index2)
            orig_to_tok_index2 = feature.orig_to_tok_index2 + [0] * padding_word_len2
            label_ids2 = feature.label_ids2 + [0] * padding_word_len2
            align_ids2 = feature.align_ids2 + [0] * padding_word_len2
            
            batch[i] = FeatureUDA(input_ids=np.asarray(input_ids),
                               attention_mask=np.asarray(mask), token_type_ids=np.asarray(type_ids),
                               orig_to_tok_index=np.asarray(orig_to_tok_index),
                               word_seq_len =feature.word_seq_len,
                               label_ids=np.asarray(label_ids),
                               align_ids=np.asarray(align_ids),
                               input_ids2=np.asarray(input_ids2),
                               attention_mask2=np.asarray(mask2), token_type_ids2=np.asarray(type_ids2),
                               orig_to_tok_index2=np.asarray(orig_to_tok_index2),
                               word_seq_len2 =feature.word_seq_len2,
                               label_ids2=np.asarray(label_ids2),
                               align_ids2=np.asarray(align_ids2))
        results = FeatureUDA(*(default_collate(samples) for samples in zip(*batch)))
        return results


## testing code to test the dataset
# from transformers import *
# tokenizer = XLMRTokenizer.from_pretrained('bert-base-uncased')
# dataset = TransformersNERDataset(file= "~/workspace/project/uda_ner/code/data/En2De/en.trans.iobes.align",tokenizer=tokenizer, is_train=True)
# from torch.utils.data import DataLoader
# train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2, collate_fn=dataset.collate_fn)
# print(len(train_dataloader))
# for batch in train_dataloader:
#     # print(batch.input_ids.size())
#     print(batch.input_ids)
#     pass
