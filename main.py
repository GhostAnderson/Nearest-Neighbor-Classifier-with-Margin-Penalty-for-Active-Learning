"""
The main file which runs our active learning experiments. The experiment results are saved in pickle files that we later
analyze over many experiments to produce the plots in our blog.
"""

import enum
import pickle
import os
import sys
import argparse
import torch
import torch.utils.data as data
from marginbert import BertWithArcLinear, NCEBert
from transformers import AdamW, get_linear_schedule_with_warmup
from custom_dataset import IMDb, AG_NEWS, Yelp_Polarity, Telecom
from query_methods_sub import *

import numpy as np
import pandas as pd
import tqdm
import random

import gc

import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

BATCH_SIZE = 10
epochs = 10

accs = []

def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, help="data type (IMDb/cifar10/cifar100)")
    p.add_argument('--output', type=str, help="output name of the final result")
    p.add_argument('--sampler', type=str, help="sampler type (discrim/random/contrast)")
    p.add_argument('--device', type=str, help='torch.device("?")')
    args = p.parse_args()
    return args

def get_one_hot_label(labels=None, num_classes=10, device = 'cpu'):

    return torch.zeros(labels.shape[0],
                       num_classes
                       ).to(torch.device(device)).scatter_(1, labels.view(-1, 1), 1)

def main(args):

    acc = []

    # parse the arguments:
    device = torch.device(args.device)
    # load the dataset:
    if args.data == 'imdb':
        train_dataset = IMDb()
        test_dataset = IMDb(mode='test')
        test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=True, pin_memory=True, drop_last=False, num_workers=8)

        args.num_examples = 25000
        args.budget = 20
        args.initial_budget = 100
        args.num_classes = 2

        args.task_epoches = 5
        args.discrim_epoches = 5
    elif args.data == 'ag':
        train_dataset = AG_NEWS()
        test_dataset = AG_NEWS(mode='test')
        test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)

        args.num_examples = 120000
        args.budget = 10
        args.initial_budget = 50
        args.num_classes = 4

        args.task_epoches = 10
        args.discrim_epoches = 200
    elif args.data == 'yelp':
        train_dataset = Yelp_Polarity()
        test_dataset = Yelp_Polarity(mode='test')
        test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=True, pin_memory=True, drop_last=False)

        args.num_examples = 560000
        args.budget = 20
        args.initial_budget = 100
        args.num_classes = 2

        args.task_epoches = 5
        args.discrim_epoches = 20025
    elif args.data == 'telecom':
        train_dataset = Telecom()
        test_dataset = Telecom(mode='test')
        test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=True, pin_memory=True, drop_last=False)

        args.num_examples = len(train_dataset)
        args.budget = 20
        args.initial_budget = 100
        args.num_classes = train_dataset.class_num

        args.task_epoches = 5
        args.discrim_epoches = 200

    if args.sampler == 'discrim':
        Sampler = DiscriminativeSampling
    elif args.sampler == 'random':
        Sampler = RandomSampling
    elif args.sampler == 'uncertainty-entropy':
        Sampler = UncertaintyEntropySampling
    elif args.sampler == 'egl-word':
        Sampler = EGLWordsSampling
    elif args.sampler == 'uncertainty':
        Sampler = UncertaintySampling
    elif args.sampler == 'bayesian-normal':
        Sampler = BayesianUncertaintySampling
    elif args.sampler == 'bayesian-entropy':
        Sampler = BayesianUncertaintyEntropySampling
    elif args.sampler == 'arcal':
        Sampler = ArcBertCosDistanceSampling
    elif args.sampler == 'arcun':
        Sampler = ArcBertCosUncertaintySampling
    elif args.sampler == 'arcrej':
        Sampler = ArcBertRejectSampling
    elif args.sampler == 'arcconf':
        Sampler = ArcBertConfusionSampling
    elif args.sampler == 'arcmargin':
        Sampler = ArcMarginSampling
    elif args.sampler == 'coreset':
        Sampler = CoreSetSampling
    elif args.sampler == 'arcmax':
        Sampler = ArcMaxSamping
    else:
        exit(0)

    all_indices = set(np.arange(args.num_examples))
    initial_indices = random.sample(list(all_indices), args.initial_budget)
    dataset_sampler = data.SubsetRandomSampler(initial_indices)
    
    splits = list(range(1, 51))

    current_indices = list(initial_indices)
    unlabeled_indices = list(all_indices - set(current_indices))

    print('begin training...')
    for split in splits:
        print('current sample time {}'.format(split))

        task_model = BertWithArcLinear.from_pretrained('bert-base-chinese', num_labels=args.num_classes, output_attentions=False, output_hidden_states=False, return_dict=False)
        # task_model = BertWithArcLinear.from_pretrained('bert-base-uncased', num_labels=args.num_classes, output_attentions=False, output_hidden_states=False, return_dict=False)
        # task_model = NCEBert.from_pretrained('bert-base-uncased', num_labels=args.num_classes, output_attentions=False, output_hidden_states=False, return_dict=False)
        # task_model = NCEBert.from_pretrained('bert-base-chinese', num_labels=args.num_classes, output_attentions=False, output_hidden_states=False, return_dict=False)
        task_model.to(device)
        
        optimizer = AdamW(task_model.parameters(), lr=2e-5)
        # crit = torch.nn.CrossEntropyLoss()
        crit = torch.nn.BCELoss()

        trainloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=dataset_sampler, drop_last=False)
        task_model.train()
        print('training task model...')
        for epoch in tqdm.trange(args.task_epoches):
            for i, data_ in enumerate(trainloader):
                X, Y, _ = data_
                X, Y = X.long().to(device), Y.long().to(device)

                Y_onehot = get_one_hot_label(Y, num_classes=args.num_classes, device=args.device)

                logits = task_model(X, token_type_ids=None, attention_mask=(X>0), labels=Y)
                # loss = crit(logits, Y)
                loss = crit(logits.sigmoid(), Y_onehot)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        task_model.eval()

        with torch.no_grad():
            print('evaluating task model...')
            correct = 0
            total = 0
            for i, data_ in enumerate(test_loader):
                X, Y, _ = data_
                X, Y = X.to(device), Y.to(device)

                feature, logits = task_model.predict(X, token_type_ids=None, attention_mask=(X>0), labels=Y)
                logits = logits.detach()

                preds = logits.argmax(dim=1)
                correct += torch.eq(preds, Y).sum().item()
                total += len(logits)

            
            print('///with {} examples labeled, total {}, correct {}, current acc {}///'.format(len(current_indices),total,correct, correct/total))
            acc.append(correct/total)
            with open('./output/{}.txt'.format(args.output), 'a') as f:
                f.write('{} {}\n'.format(len(current_indices), correct/total))
        
        print('training discriminator')
        sampler = Sampler(task_model, train_dataset, current_indices, unlabeled_indices, budget=args.budget, encoding_size=768, dis_epoch=args.discrim_epoches, device=device)
        print('sampler created')
        # sampler.train()

        query_pool_indices = sampler.query()

        current_indices += query_pool_indices
        unlabeled_indices = list(all_indices - set(current_indices))
        dataset_sampler = torch.utils.data.SubsetRandomSampler(current_indices)
        del(sampler)
        del(task_model)

        gc.collect()
    
    accs.append(acc)


if __name__ == '__main__':
    import time
    args = parse_input()
    for seed in range(114510, 114520):

        t1 = time.time()
        print('current seed: {}'.format(seed))
        with open('./output/{}.txt'.format(args.output), 'a') as f:
            f.write('{}\n'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        main(args)
        # break

        t2 = time.time()

        print('Seed {} complete, total time {}s.'.format(seed, t2-t1))

    accs = torch.Tensor(accs)
    accs_mean = accs.mean(0).tolist()

    with open('./output/{}.txt'.format(args.output), 'a') as f:
        list(map(lambda x: f.write(str(x)+' '), accs_mean))