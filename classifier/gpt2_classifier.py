#!/home/yeemanchoi/anaconda3/envs/tf/bin/python3
import pandas as pd
import numpy as np
import re

import sklearn
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

import transformers
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
workspace_path = "/home/yeemanchoi/ECE496"

dataset_1 = pd.read_csv(workspace_path + "/datasets/mbti_1.csv")
print("Dataset information:")
dataset_1.info()

labels = np.unique(dataset_1['type'])
print("Labels:", labels)

def text_preprocess(df):

    processed_df = df.copy()

    #strip quotes at beinging/end
    processed_df["posts"] = processed_df["posts"].apply(lambda x: x.strip('\''))

    #remove links
    processed_df["posts"] = processed_df["posts"].apply(lambda x: re.sub(r'https?:.*?[\s+]', '', x))

    #convert to lowercase
    processed_df["posts"] = processed_df["posts"].apply(lambda x: x.lower())

    #remove non-words, keep puntuation and numbers in word
    processed_df["posts"] = processed_df["posts"].apply(lambda x: re.sub(r'[^a-zA-Z0-9\'\.\?!\s\-]',' ',x))
    processed_df["posts"] = processed_df["posts"].apply(lambda x: re.sub(r'[^a-zA-z0-9]\.',' ',x))
    processed_df["posts"] = processed_df["posts"].apply(lambda x: re.sub(r'[^a-zA-z0-9]\?',' ',x))
    processed_df["posts"] = processed_df["posts"].apply(lambda x: re.sub(r'[^a-zA-z0-9]\!',' ',x))
    processed_df["posts"] = processed_df["posts"].apply(lambda x: re.sub(r'[^a-zA-z\s][0-9]*[^a-zA-z\s]',' ',x))

    #replace multiple spaces with one
    processed_df["posts"] = processed_df["posts"].apply(lambda x: re.sub(r'\s+',' ',x))

    return processed_df

def preprocess(df):
  processed_ds = text_preprocess(df)
  encoder = LabelEncoder()
  processed_ds['encoded_type'] = encoder.fit_transform(processed_ds['type'])
  print("encode mapping: ")
  print({x:encoder.inverse_transform([x])[0] for x in range(16)})
  return processed_ds

processed_ds = preprocess(dataset_1)
processed_ds.head(10)

model_config = GPT2Config.from_pretrained('gpt2', num_labels=16)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
print("tokenizer embeddings", len(tokenizer))

model = GPT2ForSequenceClassification.from_pretrained('gpt2', config=model_config)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
print(model.get_input_embeddings)

#store dataset in a iterable class for constructing batches later
class MBTI_Dataset(Dataset):
  def __init__(self, ds):
    self.posts = ds.posts.to_list()
    self.types = ds.encoded_type.to_list()
    self.n_examples = len(self.posts)
    return

  def __len__(self):
    return self.n_examples

  def __getitem__(self, item):
    return {'posts':self.posts[item], 'types':self.types[item]}

def collate(batch):
  #max sequence length allowed in model = 1024, need to pad/truncate sequence length
  ds = tokenizer(text=[each['posts'] for each in batch], return_tensors="pt", padding=True, truncation=True, max_length=1024)
  ds.update({'labels':torch.tensor([each['types'] for each in batch])})
  return ds

total_data = len(processed_ds)
train_ds = MBTI_Dataset(processed_ds.loc[:int(0.9*total_data)])
test_ds = MBTI_Dataset(processed_ds.loc[int(0.9*total_data):])
train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)
test_dataloader = DataLoader(test_ds, batch_size=32, shuffle=True, collate_fn=collate)

def train(model, dataloader, optimizer_, scheduler_):

  true_labels = []
  predicted_labels = []
  total_loss = 0

  model.train()

  for batch in dataloader:

    true_labels += batch['labels'].numpy().flatten().tolist()

    model.zero_grad()

    outputs = model(**batch)

    loss = outputs.loss
    logits = outputs.logits

    total_loss += loss.item()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()

    predicted_labels += logits.argmax(axis=-1).flatten().tolist()

  avg_epoch_loss = total_loss / len(dataloader)

  return true_labels, predicted_labels, avg_epoch_loss

#try different optimizer/prarmeters
optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # default is 5e-5.
                  eps = 1e-8 # default is 1e-8.
                  )

# Total number of training steps is number of batches * number of epochs.
epochs = 4
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

all_loss = {'train_loss':[], 'val_loss':[]}
all_acc = {'train_acc':[], 'val_acc':[]}

for epoch in range(epochs):

  train_labels, train_predict, train_loss = train(model, train_dataloader, optimizer, scheduler)
  train_acc = accuracy_score(train_labels, train_predict)

  valid_labels, valid_predict, val_loss = validation(valid_ds)
  val_acc = accuracy_score(valid_labels, valid_predict)

  print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))

  all_loss['train_loss'].append(train_loss)
  all_loss['val_loss'].append(val_loss)
  all_acc['train_acc'].append(train_acc)
  all_acc['val_acc'].append(val_acc)
