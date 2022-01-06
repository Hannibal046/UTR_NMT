from data import get_dataset,DataCollator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import datetime
import os
import warnings
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


data_args,model_args,training_args = check_config(DataConfig(),ModelConfig(),TrainingConfig())
train_dataset,dev_dataset,test_dataset,src_tokenizer,trg_tokenizer = get_dataset(data_args)

model_args.src_vocab_size,model_args.trg_vocab_size = len(src_tokenizer),len(trg_tokenizer)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_collator = DataCollator(device)

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
    
test_dataloader = DataLoader(
    test_dataset,
    collate_fn=data_collator,
    batch_size = training_args.per_device_eval_batch_size,
    shuffle=False,)  
model = MarianMTModel(model_args)
optimizer = Adam([{'params':model.parameters(), 'lr': model_args.d_model**-0.5}], betas=(0.9, 0.98), eps=1e-9)
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (training_args.gradient_accumulation_steps))
training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
total_batch_size = training_args.per_device_train_batch_size * training_args.world_size * training_args.gradient_accumulation_steps
lr_scheduler = get_inverse_sqrt_schedule_with_warmup(optimizer, training_args.num_warmup_steps, training_args.max_train_steps)
completed_steps = 0
best_bleu = -1
best_step = -1
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
        
        if completed_steps%training_args.logging_steps==0:
            
            loss = loss.item() * training_args.gradient_accumulation_steps
            log = f"Step[{completed_steps:>6}/{training_args.max_train_steps}] "
            log += f"Loss:{loss:.2f} "
            log += f"rouge1-f:{best_bleu:.2f} "
            log += f"bstep:{best_step} "
            r = s2hm(((training_args.max_train_steps-completed_steps)/completed_steps)\
                                        *(datetime.datetime.now()-start_time).seconds)
            log += f"time:{r}"
            print(log)

    generation_preds,generation_labels = generate(model,dev_dataloader,src_tokenizer,trg_tokenizer)
    bleu = get_bleu_score(generation_preds,generation_labels)
    if bleu>best_bleu:
        best_bleu = bleu
        best_step = completed_steps
        torch.save(model.state_dict(),os.path.join(training_args.output_dir,'best_model.ckpt'))

