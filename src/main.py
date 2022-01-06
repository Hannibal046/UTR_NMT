from data import get_txt,Dataset,build_dataset
import pickle

import transformers
from data import get_dataset,DataCollator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import datetime
import os
import warnings
from tqdm import tqdm
from transformers import set_seed

warnings.filterwarnings('ignore')

# my own module
from config import DataConfig,TrainingConfig,check_config,ModelConfig
from utils import (
    generate,
    get_bleu_score,
    s2hm
)
from model import MarianMTModel
from optim import Adam,get_inverse_sqrt_schedule_with_warmup

lang2file = {
    'de':['test_2016_flickr.lc.norm.tok','test_2017_flickr.lc.norm.tok','test_2018_flickr.lc.norm.tok','test_2017_mscoco.lc.norm.tok'],
    'cs':['test_2016_flickr.lc.norm.tok','test_2018_flickr.lc.norm.tok'],
    'fr':['test_2016_flickr.lc.norm.tok','test_2017_flickr.lc.norm.tok','test_2018_flickr.lc.norm.tok','test_2017_mscoco.lc.norm.tok'],
}

def report(data_args):
    p  = f"train_src_dir:--->{data_args.train_src_dir.split('/')[-1]}\n"
    p += f"train_trg_dir:--->{data_args.train_trg_dir.split('/')[-1]}\n"
    p += f"dev_src_dir:----->{data_args.dev_src_dir.split('/')[-1]}\n"
    p += f"dev_trg_dir:----->{data_args.dev_trg_dir.split('/')[-1]}\n"
    p += "*"*50
    print(p)

def main(data_args,model_args,training_args):
    
    set_seed(training_args.seed)
    os.makedirs('../.cache',exist_ok=True)
    print("Loading Dataset...")
    if data_args.use_cache:
        train_dataset,dev_dataset,src_tokenizer,trg_tokenizer,tfidf_vec,image_lookup_table,keyword_net = pickle.load(open('../.cache/cache.pkl','rb'))
    else:
        train_dataset,dev_dataset,src_tokenizer,trg_tokenizer,tfidf_vec,image_lookup_table,keyword_net = get_dataset(data_args)
        pickle.dump([train_dataset,dev_dataset,src_tokenizer,trg_tokenizer,tfidf_vec,image_lookup_table,keyword_net],open('../.cache/cache.pkl','wb'))

    model_args.src_vocab_size,model_args.trg_vocab_size = len(src_tokenizer),len(trg_tokenizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_collator = DataCollator(device=device)

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        shuffle=True,
        batch_size = training_args.per_device_train_batch_size,)

    dev_dataloader = DataLoader(
        dev_dataset,
        collate_fn=data_collator,
        batch_size = training_args.per_device_eval_batch_size,
        shuffle=False,)

    
    model = MarianMTModel(model_args).to(device)
    optimizer = Adam([{'params':[v for k,v in model.named_parameters() if k != "model.encoder.image_embedding.weight"],
                       'lr': model_args.d_model**-0.5}], betas=(0.9, 0.98), eps=1e-9)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (training_args.gradient_accumulation_steps))
    training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    total_batch_size = training_args.per_device_train_batch_size * training_args.world_size * training_args.gradient_accumulation_steps
    lr_scheduler = get_inverse_sqrt_schedule_with_warmup(optimizer, training_args.num_warmup_steps, training_args.max_train_steps)

    completed_steps = 0
    best_bleu = -1
    best_epoch = 0
    last_bleu = -1
    start_time = datetime.datetime.now()


    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {training_args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    print(f"  World Size = {training_args.world_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {training_args.max_train_steps}")

    for epoch in range(training_args.num_train_epochs):

        for step,batch in enumerate(train_dataloader):

            outputs = model(**batch)
            loss = outputs.loss / training_args.gradient_accumulation_steps
            loss.backward()

            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                nn.utils.clip_grad_norm_(model.parameters(),training_args.max_grad_norm,)
                optimizer.step()
                completed_steps += 1
                lr_scheduler.step()
                optimizer.zero_grad()
        
        # translate
        generation_preds,generation_labels = generate(model,dev_dataloader,model_args,trg_tokenizer)
        bleu = get_bleu_score(generation_preds,generation_labels)
        last_bleu = bleu
        if bleu>best_bleu:
            best_bleu = bleu
            best_epoch = epoch
            torch.save(model.state_dict(),'best_model_.ckpt')
        
        log
        loss = loss.item() * training_args.gradient_accumulation_steps
        log = f"Epoch[{epoch:>2}/{training_args.num_train_epochs}] "
        log += f"Loss:{loss:.2f} "
        log += f"best_bleu:{best_bleu:.2f} "
        log += f"best_epoch:{best_epoch} "
        log += f"last_bleu:{last_bleu:.2f} "
        r = s2hm(((training_args.max_train_steps-completed_steps)/completed_steps)\
                                    *(datetime.datetime.now()-start_time).seconds)
        log += f"time:{r}"
        print(log)

    ## test
    model.load_state_dict(torch.load('best_model_.ckpt'))
    for file in lang2file[data_args.trg_lang]:
        data_args.test_src_dir = os.path.join(data_args.dataset_prefix,file+'.en')
        data_args.test_trg_dir = os.path.join(data_args.dataset_prefix,file + '.'+ data_args.trg_lang)
        test_src = get_txt(data_args.test_src_dir)
        test_trg = get_txt(data_args.test_trg_dir)
        test_dataset = Dataset(build_dataset(test_src,test_trg,src_tokenizer,trg_tokenizer,data_args,\
                                            tfidf_vec,\
                                            image_lookup_table=image_lookup_table,\
                                            keyword_net=keyword_net))
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=data_collator,
            batch_size = training_args.per_device_eval_batch_size,
            shuffle=False,)

        generation_preds,generation_labels = generate(model,test_dataloader,model_args,trg_tokenizer)
        bleu = get_bleu_score(generation_preds,generation_labels)
        print(f"Bleu of [{data_args.test_trg_dir.split('/')[-1]}]:{bleu:.2f}")

if __name__ == '__main__':
        
    data_args,model_args,training_args = check_config(DataConfig(),ModelConfig(),TrainingConfig())
    for lang in ['de','cs','fr']:
        data_args.trg_lang = lang
        data_args.train_trg_dir = os.path.join(data_args.dataset_prefix,'train.lc.norm.tok.'+data_args.trg_lang)
        data_args.dev_trg_dir = os.path.join(data_args.dataset_prefix,"val.lc.norm.tok."+data_args.trg_lang)
        data_args,model_args,training_args = check_config( data_args,model_args,training_args)
        main(data_args,model_args,training_args)


