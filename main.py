#!/usr/bin/env python
# coding: utf-8

import subprocess
import sys

def install(package, force_reinstall=False, file=False):
    if force_reinstall:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", package])
    elif file:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", package])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('scikit-learn')
install('keras')
install('torchtext')
install('dropbox')

from zipfile import ZipFile
import dropbox
import os

dropbox_access_token= "Di46rAxP7HgAAAAAAAAke27OIfaQP5cynND1iaJIksmgLFuyC8faJwmJ6oEcnWnX"    #Enter your own access token

def download_unzip(folder):
    client = dropbox.Dropbox(dropbox_access_token)
    with open(os.path.join(folder, "tmp.zip"), "wb") as f:
        metadata, res = client.files_download(path="/deeplearning_needed.zip")
        f.write(res.content)

    with ZipFile(os.path.join(folder, "tmp.zip"), 'r') as zipObj:
       # Extract all the contents of zip file in current directory
       zipObj.extractall()

    os.unlink(os.path.join(folder, "tmp.zip"))

if not os.path.exists('deeplearning_needed'):
    download_unzip('')

install('transformers')
install('deeplearning_needed/transformers-master/examples/requirements.txt', file=True)
install('transformers==2.5.1', force_reinstall=True)


import glob
import logging
import os
import re
import random
import json
import timeit
from tqdm.notebook import tqdm, trange
from importlib import reload

import numpy as np

import time
import datetime

import copy
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from transformers import (XLNetTokenizer, AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer,
                          squad_convert_examples_to_features, AdamW, get_linear_schedule_with_warmup, 
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from torch.utils.tensorboard import SummaryWriter
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import XLNetForQuestionAnswering, AdamW, XLNetConfig# Load BertForSequenceClassification, the pretrained BERT model with a single 
from transformers import get_linear_schedule_with_warmup# Number of training epochs (authors recommend between 2 and 4)
import random# This training code is based on the `run_glue.py` script here:
from transformers.data.processors.squad import SquadResult, SquadV2Processor

import torchtext
from torchtext import data
from torchtext import datasets


def to_list(tensor):
    return tensor.detach().cpu().tolist()

def set_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

def load_and_cache_examples(tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_dir = "deeplearning_needed/SQUAD_data"
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, model_type.split("/"))).pop(),
            str(384),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file):
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        processor = SquadV2Processor()
        if evaluate:
            examples = processor.get_dev_examples(input_dir, filename='dev-v2.0.json')
        else:
            examples = processor.get_train_examples(input_dir, filename='train-v2.0.json')

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=not evaluate,
            return_dataset="pt",
            threads=1,
        )
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset

def train(train_dataset, model, tokenizer, n_epochs=2, eval_every=2500, save_every=5000, output_folder='SQUAD_data', checkpoint='-1', bs=2, w_checkpoint=True,
          tensordir='runs', acc_steps=1):
    """ Train the model """
    tb_writer = SummaryWriter(tensordir)
    train_batch_size = bs
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    t_total = len(train_dataloader) // acc_steps * n_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer, num_warmup_steps=0, num_training_steps=t_total
#     )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(output_folder, checkpoint, "optimizer.pt")):# and os.path.isfile(
#         os.path.join(output_folder, checkpoint, "scheduler.pt")
#     ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(output_folder, checkpoint, "optimizer.pt")))
#         scheduler.load_state_dict(torch.load(os.path.join(output_folder, checkpoint, "scheduler.pt")))
        print('Optimizer and scheduler found !\n')

    # Train!
    print("***** Running training *****")
    print("  Num examples = %s" % len(train_dataset))
    print("  Num Epochs = %s" % n_epochs)
    print("  Instantaneous batch size per GPU = %s" % bs)
    print("  Total optimization steps = %s" % t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(os.path.join(output_folder, checkpoint)):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            if checkpoint == '':
                t = glob.glob(os.path.join(output_folder, '*.txt'))[0]
                global_step = int(t[len(output_folder) + 1:-4])
            else:
                checkpoint_suffix = checkpoint.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // acc_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // acc_steps)

            print("  Continuing training from checkpoint, will skip to saved global_step")
            print("  Continuing training from epoch %d" % epochs_trained)
            print("  Continuing training from global step %d" % global_step)
            print("  Will skip the first %d steps in the first epoch" % steps_trained_in_current_epoch)
        except ValueError:
            print("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    optimizer.zero_grad()
    train_iterator = trange(
        epochs_trained, n_epochs, desc="Epoch", disable=False
    )
    # Added here for reproductibility
    set_seed()

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }


            inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
            inputs.update({"is_impossible": batch[7]})
            if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                inputs.update(
                    {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(device)}
                )

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            if acc_steps > 1:
                loss = loss / acc_steps
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
#                 scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
                global_step += 1

                #if global_step % 5000 == 0: drive.mount("/content/gdrive", force_remount=True)

                # Log metrics
                if (global_step % eval_every == 0) and (eval_every != -1):
                    # Only evaluate when single GPU otherwise metrics may not average well
                    output_dir = os.path.join(output_folder, "checkpoint-{}".format(global_step)) if w_checkpoint else output_folder
                    results = evaluate(model, tokenizer, output_dir, bs=train_batch_size)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
#                     tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / eval_every, global_step)
                    logging_loss = tr_loss
                    with open(os.path.join(output_dir, 'results_{}.json'.format(global_step)), 'w') as f:
                        json.dump(results, f)

                # Save model checkpoint
                if (global_step % save_every == 0) and (save_every != -1):
                    output_dir = os.path.join(output_folder, "checkpoint-{}".format(global_step)) if w_checkpoint else output_folder
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    #torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    print("Saving model checkpoint to %s" % output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
#                     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    print("Saving optimizer and scheduler states to %s" % output_dir)
                    if checkpoint == '':
                        if global_step == save_every: pass
                        else:
                            t = glob.glob(os.path.join(output_dir, '*.txt'))[0]
                            os.remove(t)
                        with open(os.path.join(output_dir, f'{global_step}.txt'), 'w') as f:
                            f.write(' ')
        if save_every == -1:
            output_dir = os.path.join(output_folder, "checkpoint-{}".format(global_step)) if w_checkpoint else output_folder
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            #torch.save(args, os.path.join(output_dir, "training_args.bin"))
            print("Saving model checkpoint to %s" % output_dir)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
#             torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            print("Saving optimizer and scheduler states to %s" % output_dir)
            if checkpoint == '':
                if global_step == save_every: pass
                else:
                    t = glob.glob(os.path.join(output_dir, '*.txt'))[0]
                    os.remove(t)
                with open(os.path.join(output_dir, f'{global_step}.txt'), 'w') as f:
                    f.write('')
        if eval_every == -1:
            # Only evaluate when single GPU otherwise metrics may not average well
            output_dir = os.path.join(output_folder, "checkpoint-{}".format(global_step)) if w_checkpoint else output_folder
            results = evaluate(model, tokenizer, output_dir, bs=train_batch_size)
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
#             tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
            tb_writer.add_scalar("loss", (tr_loss - logging_loss) / 500, global_step)
            logging_loss = tr_loss
            with open(os.path.join(output_dir, f'results_{global_step}.json'), 'w') as f:
                json.dump(results, f)

    tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(model, tokenizer, output_dir, prefix="", bs=2):
    dataset, examples, features = load_and_cache_examples(tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eval_batch_size = bs

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d" % len(dataset))
    print("  Batch size = %d" % eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            example_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
            # for lang_id-sensitive xlm models
            if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                inputs.update(
                    {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(device)}
                )

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    print("  Evaluation done in total %f secs (%f sec per example)" % (evalTime, evalTime / len(dataset)))

    # Compute predictions
    output_prediction_file = os.path.join(output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(output_dir, "nbest_predictions_{}.json".format(prefix))

    output_null_log_odds_file = os.path.join(output_dir, "null_odds_{}.json".format(prefix))

    # XLNet and XLM use a more complex post-processing procedure
    start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
    end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

    predictions = compute_predictions_log_probs(
        examples,
        features,
        all_results,
        20,
        30,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        start_n_top,
        end_n_top,
        True,
        tokenizer,
        False,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


if __name__ == '__main__':
    device = 'cuda'
    model_type = 'xlnet-large-cased'

    config = AutoConfig.from_pretrained(model_type)

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_type,
        from_tf=False,
        config=config,
        cache_dir=None
    )

    tokenizer = AutoTokenizer.from_pretrained(model_type,
        do_lower_case=False,
        cache_dir=None
    )

    _ = model.to(device)
    train_dataset = load_and_cache_examples(tokenizer, evaluate=False, output_examples=False)
    global_step, tr_loss = train(train_dataset, model, tokenizer, eval_every=1000, save_every=5000, output_folder='/artifacts/large_XLNet', bs=4, n_epochs=4,
                             tensordir='/storage/runs_large_XLNet', acc_steps=2)
