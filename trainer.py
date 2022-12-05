import argparse
import random
import numpy as np
from src.config import Config, evaluate_batch_insts
import time
from src.model import TransformersCRF
import torch
from typing import List
from termcolor import colored
import os
from src.config.utils import write_results
from src.config.transformers_util import get_huggingface_optimizer_and_scheduler
from src.config import context_models, get_metric
import pickle
import tarfile
from tqdm import tqdm
from collections import Counter
from src.data import TransformersNERDataset, TransformersNERDatasetUDA
from torch.utils.data import DataLoader
import conlleval

import matplotlib.pyplot as plt
import torch.nn.functional as F


def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2','cuda'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--dataset', type=str, default="conll2003_sample")
    parser.add_argument('--optimizer', type=str, default="adamw", help="This would be useless if you are working with transformers package")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="usually we use 0.01 for sgd but 2e-5 working with bert/roberta")
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=30, help="default batch size is 10 (works well for normal neural crf), here default 30 for bert-based crf")
    parser.add_argument('--num_epochs', type=int, default=5, help="Usually we set to 100.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    
    parser.add_argument('--uda_num', type=int, default=-1, help="-1 means all the data")
    
    parser.add_argument('--max_no_incre', type=int, default=30, help="early stop when there is n epoch not increasing on dev")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm, if <=0, means no clipping, usually we don't use clipping for normal neural ncrf")

    ##model hyperparameter
    parser.add_argument('--model_folder', type=str, default="english_model", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=0, help="hidden size of the LSTM, usually we set to 200 for LSTM-CRF")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")

    parser.add_argument('--embedder_type', type=str, default="bert-base-cased",
                        choices=["normal"] + list(context_models.keys()),
                        help="normal means word embedding + char, otherwise you can use 'bert-base-cased' and so on")
    parser.add_argument('--parallel_embedder', type=int, default=0,
                        choices=[0, 1],
                        help="use parallel training for those (BERT) models in the transformers. Parallel on GPUs")

    parser.add_argument('--uda_threshold', type=float, default=0.0)
    parser.add_argument('--rdrop_weight', type=float, required=True)
    parser.add_argument('--uda_weight', type=float, required=True)

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

# Repeating dataloader for UDA
def repeat_dataloader(iterable):
    """ repeat dataloader """
    while True:
        for x in iterable:
            yield x    
            
def tok_score_2_span_prob(batch_align_ids, batch_e_scores, max_entity, config, delta = 1e-20):
    
    list_entity_prob = [[torch.tensor([]) for j in range(max_entity)] for i in range(batch_align_ids.shape[0])] # list (batch) of list (max_entity) of empty tensor
    
    batch_without_entity = True
    for batch_id in range(batch_align_ids.shape[0]):
        
        sent_align_id = batch_align_ids[batch_id]
        sent_e_score = batch_e_scores[batch_id]
        
        for i in range(1, max_entity+1):
            e_score = sent_e_score[sent_align_id == i]
            e_prob = torch.nn.functional.softmax(e_score, dim=-1)
            if e_prob.shape[0] == 0:
                continue
            elif e_prob.shape[0] == 1:
                per_prob = e_prob[0,config.label2idx["S-PER"]] 
                org_prob = e_prob[0,config.label2idx["S-ORG"]] 
                loc_prob = e_prob[0,config.label2idx["S-LOC"]] 
                misc_prob = e_prob[0,config.label2idx["S-MISC"]] 
                o_prob = e_prob[0,config.label2idx["O"]] 
                
                # Numerical Stability                
                per_prob = per_prob + delta if per_prob < delta else per_prob
                org_prob = org_prob + delta if org_prob < delta else org_prob
                loc_prob = loc_prob + delta if loc_prob < delta else loc_prob
                misc_prob = misc_prob + delta if misc_prob < delta else misc_prob
                o_prob = o_prob + delta if o_prob < delta else o_prob
              
                rest_prob = 1 - (per_prob + org_prob + loc_prob + misc_prob + o_prob)
                if rest_prob <= 0:
                    print('probs:', per_prob, org_prob, loc_prob, misc_prob, o_prob)
                assert rest_prob > 0 
                
                assert per_prob.requires_grad
                assert rest_prob.requires_grad
  
                entity_prob = torch.stack([per_prob, org_prob, loc_prob, misc_prob, o_prob, rest_prob], dim=0)
                list_entity_prob[batch_id][i-1] = entity_prob
                
                batch_without_entity = False
            else:
                per_prob = e_prob[0,config.label2idx["B-PER"]]
                per_prob = per_prob * e_prob[-1,config.label2idx["E-PER"]]
                for i_prob in e_prob[1:-1,config.label2idx["I-PER"]]:
                    per_prob = per_prob * i_prob
                    
                org_prob = e_prob[0,config.label2idx["B-ORG"]]
                org_prob = org_prob * e_prob[-1,config.label2idx["E-ORG"]]
                for i_prob in e_prob[1:-1,config.label2idx["I-ORG"]]:
                    org_prob = org_prob * i_prob

                loc_prob = e_prob[0,config.label2idx["B-LOC"]]
                loc_prob = loc_prob * e_prob[-1,config.label2idx["E-LOC"]]
                for i_prob in e_prob[1:-1,config.label2idx["I-LOC"]]:
                    loc_prob = loc_prob * i_prob

                misc_prob = e_prob[0,config.label2idx["B-MISC"]]
                misc_prob = misc_prob * e_prob[-1,config.label2idx["E-MISC"]]
                for i_prob in e_prob[1:-1,config.label2idx["I-MISC"]]:
                    misc_prob = misc_prob * i_prob

                o_prob = 1.0
                for prob in e_prob[:,config.label2idx["O"]]:
                    o_prob = o_prob * prob
                    
                # Numerical Stability                
                per_prob = per_prob + delta if per_prob < delta else per_prob
                org_prob = org_prob + delta if org_prob < delta else org_prob
                loc_prob = loc_prob + delta if loc_prob < delta else loc_prob
                misc_prob = misc_prob + delta if misc_prob < delta else misc_prob
                o_prob = o_prob + delta if o_prob < delta else o_prob

                rest_prob = 1 - (per_prob + org_prob + loc_prob + misc_prob + o_prob)
                if rest_prob <= 0:
                    print('probs:', per_prob, org_prob, loc_prob, misc_prob, o_prob)
                assert rest_prob > 0 
                
                entity_prob = torch.stack([per_prob, org_prob, loc_prob, misc_prob, o_prob, rest_prob], dim=0)
                list_entity_prob[batch_id][i-1] = entity_prob
                
                batch_without_entity = False
                
    if not batch_without_entity:
        return list_entity_prob
    else:
        return None
            
def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    p_loss = p_loss.sum(dim=-1)
    q_loss = q_loss.sum(dim=-1)
    
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    p_loss = p_loss.sum() / (~pad_mask).sum() #mean
    q_loss = q_loss.sum() / (~pad_mask).sum() #mean

    loss = (p_loss + q_loss) / 2
    return loss            
            
            

def train_model(config: Config, epoch: int, train_loader: DataLoader, dev_loader: DataLoader, test_loader, uda_dataloader: DataLoader):
    ### Data Processing Info
    train_num = len(train_loader)
    print(f"[Data Info] number of training instances: {train_num}")

    print(
        colored(f"[Model Info]: Working with transformers package from huggingface with {config.embedder_type}", 'red'))
    print(colored(f"[Optimizer Info]: You should be aware that you are using the optimizer from huggingface.", 'red'))
    print(colored(f"[Optimizer Info]: Change the optimier in transformers_util.py if you want to make some modifications.", 'red'))
    model = TransformersCRF(config)
    optimizer, scheduler = get_huggingface_optimizer_and_scheduler(config, model, num_training_steps=len(train_loader) * epoch,
                                                                   weight_decay=0.0, eps = 1e-8, warmup_step=0)
    print(colored(f"[Optimizer Info] Modify the optimizer info as you need.", 'red'))
    print(optimizer)

    model.to(config.device)

    best_dev = [-1,-1,-1]
    best_test = [-1,-1,-1]

    model_folder = config.model_folder
    res_folder = "results"
    if os.path.exists("model_files/" + model_folder):
        raise FileExistsError(
            f"The folder model_files/{model_folder} exists. Please either delete it or create a new one "
            f"to avoid override.")
    model_path = f"model_files/{model_folder}/lstm_crf.m"
    config_path = f"model_files/{model_folder}/config.conf"
    dev_path_en = f"{res_folder}/{model_folder}.dev_results_en"
    res_path = f"{res_folder}/{model_folder}.results"
    print("[Info] The model will be saved to: %s.tar.gz" % (model_folder))
    os.makedirs(f"model_files/{model_folder}", exist_ok= True) ## create model files. not raise error if exist
    os.makedirs(res_folder, exist_ok=True)
    no_incre_dev = 0
    print(colored(f"[Train Info] Start training, you have set to stop if performace not increase for {config.max_no_incre} epochs",'red'))
    
    # Convert to repeating dalaloader for UDA
    uda_dataloader_repeat = repeat_dataloader(uda_dataloader)
    
    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        label_kl_epoch_loss = 0
        unlabel_kl_epoch_loss = 0
        uda_epoch_loss = 0
        start_time = time.time()
        uda_time = 0
        model.zero_grad()
        model.train()
        
        num_uda = 0
        total_uda = 1e-8
        
        skipped_uda_batch = 0
        unlabel_skip = 0
        trans_skip = 0
        missing_skip = 0
        
        unlabel_maxprobs = np.array([])
        trans_maxprobs = np.array([])
        unlabel_maxprobs_legit = np.array([])
        trans_maxprobs_legit = np.array([])
              
        
        for iter, batch in tqdm(enumerate(train_loader, 1), desc="--training batch", total=len(train_loader)):
            optimizer.zero_grad()
            loss, label_e_scores, label_start_tok_mask = model(words = batch.input_ids.to(config.device), 
                                                               word_seq_lens = batch.word_seq_len.to(config.device),
                                                               orig_to_tok_index = batch.orig_to_tok_index.to(config.device), 
                                                               input_mask = batch.attention_mask.to(config.device),
                                                               labels = batch.label_ids.to(config.device), output_emission=True)
            
            # dropout-based consistency training
            _, label_e_scores_2, _ = model(words = batch.input_ids.to(config.device), 
                                           word_seq_lens = batch.word_seq_len.to(config.device),
                                           orig_to_tok_index = batch.orig_to_tok_index.to(config.device), 
                                           input_mask = batch.attention_mask.to(config.device),
                                           labels = batch.label_ids.to(config.device), 
                                           output_emission=True)            
            
            if config.rdrop_weight > 0.0:
                label_logits = label_e_scores.view([-1, label_e_scores.shape[-1]])
                label_logits2 = label_e_scores_2.view([-1, label_e_scores_2.shape[-1]])
                label_pad_mask = ~label_start_tok_mask.bool().view([-1]) # 1 where pad token and 0 otherwise

                label_kl_loss = compute_kl_loss(label_logits, label_logits2, pad_mask=label_pad_mask)
                
                loss = loss + config.rdrop_weight * label_kl_loss
                label_kl_epoch_loss += label_kl_loss.item()
            
            
            # Translation-based consistency training            
            uda_batch = [t.to(config.device) for t in next(uda_dataloader_repeat)]
            input_ids, attention_mask, token_type_ids, orig_to_tok_index, word_seq_len, label_ids, align_ids, \
            input_ids2, attention_mask2, token_type_ids2, orig_to_tok_index2, word_seq_len2, label_ids2, align_ids2 = uda_batch
            
            if config.uda_weight > 0.0:
                _, unlabel_e_scores, _ = model(words=input_ids, word_seq_lens=word_seq_len, orig_to_tok_index=orig_to_tok_index,
                                           input_mask=attention_mask, labels=label_ids, output_emission=True)
                _, trans_e_scores, _ = model(words=input_ids2, word_seq_lens=word_seq_len2, orig_to_tok_index=orig_to_tok_index2,
                                           input_mask=attention_mask2, labels=label_ids2, output_emission=True)

                max_entity = 10
                unlabel_entity_probs = tok_score_2_span_prob(batch_align_ids=align_ids, batch_e_scores=unlabel_e_scores, 
                                                             max_entity=max_entity, config=config)
                trans_entity_probs = tok_score_2_span_prob(batch_align_ids=align_ids2, batch_e_scores=trans_e_scores, 
                                                           max_entity=max_entity, config=config)

                if (unlabel_entity_probs is not None) and (trans_entity_probs is not None):

                    unlabel_entity_probs_list = []
                    trans_entity_probs_list = []
                    for batch_id in range(len(unlabel_entity_probs)):                    
                        # Discard ENTIRE SENTENCE if entity is missing
                        unlabel_prob_sent = []
                        trans_prob_sent = []
                        no_missing = True
                        for entity_id in range(max_entity):
                            if unlabel_entity_probs[batch_id][entity_id].nelement() != 0 and trans_entity_probs[batch_id][entity_id].nelement() != 0:
                                unlabel_prob_sent.append(unlabel_entity_probs[batch_id][entity_id])
                                trans_prob_sent.append(trans_entity_probs[batch_id][entity_id])
                            elif unlabel_entity_probs[batch_id][entity_id].nelement() != trans_entity_probs[batch_id][entity_id].nelement():
                                missing_skip +=1
                                no_missing = False
                        if no_missing:
                            unlabel_entity_probs_list.extend(unlabel_prob_sent)
                            trans_entity_probs_list.extend(trans_prob_sent)

                    unlabel_entity_probs = torch.stack(unlabel_entity_probs_list)
                    trans_entity_probs = torch.stack(trans_entity_probs_list)
                    assert unlabel_entity_probs.shape[0] == trans_entity_probs.shape[0]

                    trans_entity_log_probs = torch.log(trans_entity_probs)

                    # UDA loss masking
                    uda_loss_mask = torch.max(unlabel_entity_probs[:,:-1], dim=-1)[0] > config.uda_threshold
                    uda_loss_mask = uda_loss_mask.type(torch.float32)
                    uda_loss_mask = uda_loss_mask.to(config.device)

                    # KL
                    KL = torch.nn.KLDivLoss(reduction='none') # use batchmean instead of mean to align with math definition
                    # Use bidirectional KL
                    uda_loss = (torch.sum(KL(trans_entity_log_probs, unlabel_entity_probs), dim=-1) \
                              + torch.sum(KL(torch.log(unlabel_entity_probs), trans_entity_probs), dim=-1)) / 2
                    
                    uda_loss = torch.sum(uda_loss * uda_loss_mask, dim=-1) / torch.max(torch.sum(uda_loss_mask, dim=-1), torch.tensor(1.).to(config.device))

                    loss = loss + uda_loss * config.uda_weight

                    num_uda += torch.sum(uda_loss_mask, dim=-1).item()
                    total_uda += uda_loss_mask.shape[0]

                    uda_epoch_loss += uda_loss.item()
                else:
                    skipped_uda_batch += 1
                    if unlabel_entity_probs is None:
                        unlabel_skip += 1
                    if trans_entity_probs is None:
                        trans_skip += 1
            
            epoch_loss += loss.item()            
            
            loss.backward()

            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            model.zero_grad()
        end_time = time.time()
    
        print(f"unlabel skip: {unlabel_skip}, trans_skip:{trans_skip}, missing_skip: {missing_skip}")
        print("Epoch %d: loss %.5f, label kl loss %.5f, unlabel kl loss %.5f, uda loss %.5f, uda ratio is %.2f, skipped uda batch %d / %d, Time is %.2fs" % (i, epoch_loss, label_kl_epoch_loss, unlabel_kl_epoch_loss, uda_epoch_loss, num_uda/total_uda, skipped_uda_batch, len(train_loader), end_time - start_time), flush=True)

        model.eval()
        evaluate_model(config, model, dev_loader, "dev", dev_loader.dataset.insts)
        
        write_results(dev_path_en, dev_loader.dataset.insts)
        dev_metrics, res_dev = evaluate_model_conlleval(dev_path_en)
        print("dev precision results: ",dev_metrics[0])
        print("dev recall results: ",dev_metrics[1])
        print("dev f1 results: ",dev_metrics[2])
        if dev_metrics[2] > best_dev[2]:
            print("saving the best model...")
            no_incre_dev = 0
            best_dev = dev_metrics
            torch.save(model.state_dict(), model_path)
            # Save the corresponding config as well.
            f = open(config_path, 'wb')
            pickle.dump(config, f)
            f.close()
            evaluate_model(config, model, test_loader, "test", test_loader.dataset.insts)
            write_results(res_path, test_loader.dataset.insts)
            test_metrics, res_test = evaluate_model_conlleval(res_path)
            best_test = test_metrics
            print("Test Set Result:")
            print(conlleval.report(res_test))
            
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            print("early stop because there are %d epochs not increasing f1 on dev"%no_incre_dev)
            break

    print("The best dev: %.4f" % (best_dev[2]))
    print("The corresponding test: %.4f" % (best_test[2]))
    print("Final testing.")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    evaluate_model(config, model, test_loader, "test", test_loader.dataset.insts)
    write_results(res_path, test_loader.dataset.insts)
    test_metrics, res_test = evaluate_model_conlleval(res_path)
    print("Test Set Result:")
    print(conlleval.report(res_test))



def evaluate_model_conlleval(fpath):
    file_context = open(fpath).read().splitlines()
    res = conlleval.evaluate(file_context)
    fscore = res['overall']['chunks']['evals']['f1']
    precision = res['overall']['chunks']['evals']['prec']
    recall = res['overall']['chunks']['evals']['rec']
    return [precision, recall, fscore], res

def evaluate_model(config: Config, model: TransformersCRF, data_loader: DataLoader, name: str, insts: List, print_each_type_metric: bool = False):
    ## evaluation
    p_dict, total_predict_dict, total_entity_dict = Counter(), Counter(), Counter()
    batch_size = data_loader.batch_size
    with torch.no_grad():
        for batch_id, batch in tqdm(enumerate(data_loader, 0), desc="--evaluating batch", total=len(data_loader)):
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = model.decode(words= batch.input_ids.to(config.device),
                    word_seq_lens = batch.word_seq_len.to(config.device),
                    orig_to_tok_index = batch.orig_to_tok_index.to(config.device),
                    input_mask = batch.attention_mask.to(config.device))
            batch_p , batch_predict, batch_total = evaluate_batch_insts(one_batch_insts, batch_max_ids, batch.label_ids, batch.word_seq_len, config.idx2labels)
            p_dict += batch_p
            total_predict_dict += batch_predict
            total_entity_dict += batch_total
            batch_id += 1 

    return

def main():
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    conf = Config(opt)

    set_seed(opt, conf.seed)

    print(colored(f"[Data Info] Tokenizing the instances using '{conf.embedder_type}' tokenizer", "blue"))
    tokenizer = context_models[conf.embedder_type]["tokenizer"].from_pretrained(conf.embedder_type)
    print(colored(f"[Data Info] Reading dataset from: \n{conf.train_file}\n{conf.dev_file}\n{conf.test_file_en}\n{conf.test_file_de}\n{conf.test_file_es}\n{conf.test_file_nl}", "blue"))
    train_dataset = TransformersNERDataset(conf.train_file, tokenizer, number=conf.train_num, is_train=True)
    conf.label2idx = train_dataset.label2idx
    conf.idx2labels = train_dataset.idx2labels

    dev_dataset = TransformersNERDataset(conf.dev_file, tokenizer, number=conf.dev_num, label2idx=train_dataset.label2idx, is_train=False)
    test_dataset = TransformersNERDataset(conf.test_file, tokenizer, number=conf.test_num, label2idx=train_dataset.label2idx, is_train=False)
    
    uda_dataset = TransformersNERDatasetUDA(conf.unlabel_file, conf.trans_file, tokenizer, number=conf.uda_num, label2idx=train_dataset.label2idx, is_train=False)
    
    num_workers = 8
    conf.label_size = len(train_dataset.label2idx)
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
                                  collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
                                  collate_fn=test_dataset.collate_fn)
    
    uda_dataloader = DataLoader(uda_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=uda_dataset.collate_fn)
    
    train_model(conf, conf.num_epochs, train_dataloader, dev_dataloader, test_dataloader, uda_dataloader)


if __name__ == "__main__":
    main()
