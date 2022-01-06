
import os
import math
# from CONSTANT import REPORT_DATA_ARGS,REPORT_MODEL_ARGS,REPORT_TRAINING_ARGS

from dataclasses import asdict

import time
import numpy as np
import json
# import editdistance
import torch.nn as nn
import torch
import nltk
# import rouge
from sklearn.metrics import f1_score, balanced_accuracy_score
from tqdm import tqdm
from collections import OrderedDict

# from CONSTANT import GENERATION_INPUTS_FOR_MODELS
from torch.nn.parallel import DistributedDataParallel as DDP

def weight_adaptation():
    new_dict = OrderedDict()
    bart_state_dict = torch.load('../pretrained_model/bart_base/pytorch_model.bin')
    for k,v in bart_state_dict.items():
        if 'encoder' in k and  'decoder' not in k:
            t = k.split('.')
            t.insert(2,'text_encoder')
            t = '.'.join(t)
            new_dict[t] = bart_state_dict[k]
        else:
            new_dict[k] = v
    return new_dict

def get_edit_sim(src,tm):
    
    
    a = src.split()
    b = tm.split()
    edit_distance = editdistance.eval(a, b)

    edit_sim = 1 - edit_distance / max(len(src), len(tm))

    return edit_sim

def get_cos_sim(src,tm):

    sim_fn = nn.CosineSimilarity(dim=-1)
    return sim_fn(src,tm)

def get_json(f):
    
    return [json.loads(x) for x in open(f).readlines()]

def generate_and_classify(model,data_loader,model_args,tokenizer):#,progress,eval_task):
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    
    generation_preds = []
    generation_labels = []
    rating_preds = []
    rating_labels = []
    all_loss = []
    lm_loss = []
    rating_loss = []

    model.eval()
    gen_kwargs = {
        'max_length':model_args.max_trg_len,
        'num_beams':model_args.num_beams,
    }        
    for step,batch in enumerate(data_loader):
        with torch.no_grad():
            

            additional_kwargs = {}
            for k in GENERATION_INPUTS_FOR_MODELS[model_args.model_type]:
                additional_kwargs[k] = batch.get(k,None)

            # get generated tokens and rating representation
            if isinstance(model,DDP):
                generated_tokens = model.module.generate(
                    batch['input_ids'],
                    attention_mask = batch['attention_mask'],
                    **gen_kwargs,
                    **additional_kwargs,
                )
            else:
                generated_tokens = model.generate(
                    batch['input_ids'],
                    attention_mask = batch['attention_mask'],
                    **gen_kwargs,
                    **additional_kwargs,
                )

            # get loss
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs['loss'].mean().detach().item()
                all_loss.append(loss)
                lm_loss.append(outputs['lm_loss'])
                rating_loss.append(outputs['rating_loss'])
                rating_pred = outputs['rating_pred'].cpu().numpy()


            # # detokenize preds and labels 
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, model_args.pad_token_id)
            generated_tokens = generated_tokens.cpu().numpy()
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            for pred,label in zip(decoded_preds,decoded_labels):
                generation_preds.append(pred)
                generation_labels.append(label)
            
            rating_label = batch['rating_labels'].squeeze().cpu().numpy() # bs
            rating_preds.append(rating_pred)
            rating_labels.append(rating_label)
            
    model.train()
    return generation_preds,generation_labels,np.concatenate(rating_preds,axis=0),np.concatenate(rating_labels,axis=0),sum(all_loss)/len(all_loss),sum(lm_loss)/len(lm_loss),sum(rating_loss)/len(rating_loss)

def get_rouge_score(hyps,refs):
    import rouge
    hyps,refs = postprocess_text(hyps,refs) # add \n

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=True,
                            apply_best=False,
                            alpha=0.5,  # Default F1_score
                            stemming=True)
    py_rouge_scores = evaluator.get_scores(hyps, refs)
    
    ## *100
    for k,v in py_rouge_scores.items():
        for _k,_v in v.items():
            py_rouge_scores[k][_k] = round(_v*100,4)

    return py_rouge_scores

def generate(model,data_loader,model_args,tokenizer):

    
    generation_preds = []
    generation_labels = []
    all_loss = []

    model.eval()
    gen_kwargs = {
        'max_length':model_args.max_trg_len,
        'num_beams':model_args.num_beams,
    }
    # with Progress(transient=True) as progress:
        # eval_task = progress.add_task('[green]evaluating...',total = len(data_loader))
    for step,batch in enumerate(data_loader):
        with torch.no_grad():
            
            additional_kwargs = {}
            if model_args.use_image or model_args.use_keyword:
                additional_kwargs['extra'] = batch['extra']

            if isinstance(model,DDP):
                generated_tokens = model.module.generate(
                    batch['input_ids'],
                    attention_mask = batch['attention_mask'],
                    **gen_kwargs,
                    **additional_kwargs,)
            else:
                generated_tokens = model.generate(
                    batch['input_ids'],
                    attention_mask = batch['attention_mask'],
                    **gen_kwargs,
                    **additional_kwargs,
                )

            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs['loss'].mean().detach().item()
                all_loss.append(loss)


            # # detokenize preds and labels 
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, model_args.pad_token_id)
            generated_tokens = generated_tokens.cpu().numpy()
            decoded_preds = tokenizer.decode_batch(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.decode_batch(labels, skip_special_tokens=True)
            for pred,label in zip(decoded_preds,decoded_labels):
                generation_preds.append(pred)
                generation_labels.append(label)
            # progress.update(eval_task,advance=1)
    model.train()
    return generation_preds,generation_labels#,sum(all_loss)/len(all_loss)

def balanced_acc_and_macro_f1(preds, labels):
    acc = round(balanced_accuracy_score(y_true=labels, y_pred=preds)*100,4)
    f1 = round(f1_score(y_true=labels, y_pred=preds, average='macro')*100,4)
    return {
        "balanced_accuracy": acc,
        "macro_f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def get_time_stamp():
    t = time.localtime(time.time())
    return f"{t.tm_mon}-{t.tm_mday}-{t.tm_hour}:{t.tm_min}:{t.tm_sec}"

def save_numpy(f,n):
    np.save(f,n)

def load_numpy(f):
    return np.load(f)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def report_args(data_args,model_args,training_args):

    def get_content(k,v):
        return f"[b]{k}[/b]\n[yellow]{v}"

    console = Console()

    console.rule('[bold red]Data Args')
    renderables = [Panel(get_content(k,v)) for k,v in sorted(asdict(data_args).items()) if k in REPORT_DATA_ARGS ]
    console.print(Columns(renderables))

    console.rule('[bold red]Model Args')
    renderables = [Panel(get_content(k,v)) for k,v in sorted(vars(model_args).items()) if k in REPORT_MODEL_ARGS ]
    renderables.insert(0,Panel(get_content('model_type',model_args.model_type)))
    console.print(Columns(renderables))

    console.rule('[bold red]Training Args')
    renderables = [Panel(get_content(k,v)) for k,v in sorted(asdict(training_args).items()) if k in REPORT_TRAINING_ARGS ]
    console.print(Columns(renderables))

def s2hm(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    # return  "%02d:%02d:%02d" % (h, m, s)
    return  "%02d:%02d" % (h, m)

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

def save_config(data_args,model_args,training_args):
    ret = {
        "data_args":{},
        "model_args":{},
        "training_args":{},
    }
    for k,v in asdict(data_args).items():
        if k in REPORT_DATA_ARGS:
            ret["data_args"][k] = v

    for k,v in vars(model_args).items():
        if k in REPORT_MODEL_ARGS:
            ret["model_args"][k] = str(v)
    
    ret['model_args']['model_type'] = model_args.model_type

    for k,v in asdict(training_args).items():
        if k in REPORT_TRAINING_ARGS:
            ret["training_args"][k] = v

    # ret['training_args']['logging_strategy'] = ret['training_args']['logging_strategy'].value
    # ret['training_args']['evaluation_strategy'] = ret['training_args']['evaluation_strategy'].value
    # ret['training_args']['lr_scheduler_type'] = ret['training_args']['lr_scheduler_type'].value
    # ret['training_args']['save_strategy'] = ret['training_args']['save_strategy'].value
    # return ret
    # print(ret)
    with open(os.path.join(training_args.output_dir,'config.json'),'w') as f:
        json.dump(ret,f,indent=4)

import socket
def port_is_used(port,ip='127.0.0.1'):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    try:
        s.connect((ip,port))
        s.shutdown(2)
        return True
    except:
        return False

def get_bleu_score(src,trg):
    import sacrebleu
    # trg = [[x] for x in trg]
    return (sacrebleu.corpus_bleu(src, [trg],force=True,lowercase=False,tokenize='none')).score
    # bleu = BLEU()
    # return round((bleu.corpus_scores(src,trg)).score,2)

