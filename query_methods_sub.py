import numpy as np
import pandas as pd

import torch
from torch import optim
from torch.nn import parameter
import torch.nn.functional as F

import discriminator

import tqdm
import math
import random
import gc

from scipy.spatial import distance_matrix


def get_one_hot_label(labels=None, num_classes=10, device = 'cpu'):

    return torch.zeros(labels.shape[0],
                       num_classes
                       ).to(torch.device(device)).scatter_(1, labels.view(-1, 1), 1)

class DiscrimDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, current_indices, task_model, device):
        super().__init__()

        self.task_model = task_model
        self.dataset = dataset
        self.device = device
        self.current_indices = current_indices
        self.neg_indices = list(set(range(len(self.dataset))) - set(current_indices))
        self.neg_indices = random.sample(self.neg_indices, np.min([len(self.current_indices) * 10, len(self.neg_indices)]))

        self.indices = self.current_indices + self.neg_indices
        self.Ys = [1] * len(self.current_indices) + [0] * len(self.neg_indices)

        self.encode()
    
    def encode(self):
        self.Xs = []
        self.task_model.eval()
        self.embeddings = []
        for i in self.indices:
            X, _, _ = self.dataset[i]
            self.Xs.append(X)

            with torch.no_grad():
                x = self.task_model.bert(X.unsqueeze(0).to(self.device), token_type_ids=None, attention_mask=(X>0).unsqueeze(0).to(self.device))[1].squeeze()
                self.embeddings.append(x)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.Xs[index], self.embeddings[index], self.Ys[index]


class DiscriminativeSampling():

    def __init__(self, task_model, dataset, current_indices, unlabeled_indices, budget, device, encoding_size=300, dis_epoch = 1):
        self.task_model = task_model
        self.encoding_size = encoding_size
        self.dis_epoch = dis_epoch
        self.budget = budget
        self.device = device

        self.dataset = dataset
        self.current_indices = current_indices
        self.unlabeled_indices = unlabeled_indices

        self.sub_sample_time = 5
        self.sub_batch_size = int(self.budget/self.sub_sample_time)
    
    def train(self):
        pass

    def query(self):
        
        final_query_res = []

        for i in range(self.sub_sample_time):

            self.discriminator = discriminator.Discriminator(task_model = self.task_model, width=self.encoding_size, input_size=self.encoding_size).to(self.device)
            self.discriminator.train()

            discrim_dataset = DiscrimDataset(self.dataset, self.current_indices + final_query_res, self.task_model, self.device)
            discrim_dataloader = torch.utils.data.DataLoader(discrim_dataset, batch_size=100, drop_last=False, shuffle=True)

            crit = torch.nn.CrossEntropyLoss(weight=torch.Tensor([10,1]).to(self.device))
            optim = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

            for e in tqdm.trange(200):
            # for e in tqdm.trange(self.dis_epoch):
    
                total = 0
                correct = 0

                losses = []

                for index, data in enumerate(discrim_dataloader):
                    x, emb, y = data
                    x, emb, y = x.to(self.device), emb.to(self.device), torch.LongTensor(y).to(self.device)

                    y_ = self.discriminator(emb)
                    loss = crit(y_, y)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    losses.append(loss.cpu().detach().item())

                    preds = y_.argmax(dim=1)
                    correct += torch.eq(preds, y).sum()
                    total += len(y)
                
                print('Epoch {}: Average Loss: {}'.format(e, sum(losses)/len(losses)))

                acc = correct / total
                if acc >= 0.95:
                    print('Early stop triggered! Current acc {}'.format(acc))
                    break
                else:
                    print('epoch {}: current acc {}'.format(e, acc))

            candidates = discrim_dataset.neg_indices
            sampler = torch.utils.data.sampler.SubsetRandomSampler(candidates)
            loader = torch.utils.data.DataLoader(self.dataset, batch_size=None, sampler=sampler)

            all_indices = []
            all_preds = []

            self.discriminator.eval()

            for i, data in enumerate(loader):
                x, _, index = data
                x = x.to(self.device).unsqueeze(0)
                x = self.task_model.bert(x, token_type_ids=None, attention_mask=(x>0))[1]
                y_ = F.softmax(self.discriminator(x))[0].detach().cpu()[0].item()

                all_preds.append(y_)
                all_indices.append(index)
            
            _, temp_pos = torch.topk(torch.Tensor(all_preds), k=self.sub_batch_size)
            query_pool_indices = torch.Tensor(all_indices)[temp_pos].long().tolist()
            final_query_res += query_pool_indices

            del(self.discriminator)
            gc.collect()

        return final_query_res

class RandomSampling():
    def __init__(self, task_model, dataset, current_indices, unlabeled_indices, budget, encoding_size=300, dis_epoch = 1, device=None):
        self.task_model = task_model
        self.encoding_size = encoding_size
        self.dis_epoch = dis_epoch
        self.budget = budget

        self.dataset = dataset
        self.current_indices = current_indices
        self.unlabeled_indices = unlabeled_indices

    def train(self):
        #do nothing
        return "我真的训练完了！"

    def query(self):
        query_pool_indices = random.sample(self.unlabeled_indices, self.budget)
        return query_pool_indices

class UncertaintyEntropySampling():
    '''
    The basic uncertainty sampling query strategy, querying the examples with the top entropy.
    '''
    def __init__(self, task_model, dataset, current_indices, unlabeled_indices, budget, device, encoding_size=300, dis_epoch = 1, beta=0, autoencoder_epoch=5):
        
        self.task_model = task_model
        self.dataset = dataset
        self.current_indices = current_indices
        self.unlabeled_indices = unlabeled_indices
        self.budget = budget
        self.device = device
    
    def train(self):
        pass
    
    def query(self):
        
        dataset_sampler = torch.utils.data.SubsetRandomSampler(self.unlabeled_indices)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=200, sampler=dataset_sampler, drop_last=False)
        
        all_indices = []
        all_logits = []

        for i, data in tqdm.tqdm(enumerate(dataloader)):
            X, Y, index = data
            X, Y = X.to(self.device), Y.to(self.device)

            with torch.no_grad():
                loss, logits = self.task_model(X, token_type_ids=None, attention_mask=(X>0).to(self.device), labels=Y)
                logits = torch.softmax(logits, dim=-1)
            
            all_logits.append(logits)
            all_indices.append(index)
        
        all_indices = torch.hstack(all_indices)
        all_logits = torch.cat(all_logits, dim=0).detach().cpu()
        all_entropies = torch.sum(all_logits*torch.log(all_logits+1e-10), dim=-1)
        temp_pos = np.argpartition(all_entropies, self.budget)[:self.budget].tolist()

        selected_indices = all_indices[temp_pos]

        return selected_indices.long().tolist()

class EGLWordsSampling():
    '''
    EGL-words.
    An implementation of the EGL query strategy.
    '''
    def __init__(self, task_model, dataset, current_indices, unlabeled_indices, budget, device, encoding_size=300, dis_epoch = 1, beta=0, autoencoder_epoch=5, n_classes=2):

        self.task_model = task_model
        self.dataset = dataset
        self.current_indices = current_indices
        self.unlabeled_indices = unlabeled_indices
        self.budget = budget
        self.device = device
        self.n_classes = n_classes

    def train(self):
        pass
    
    def query(self):

        candidate_indices = random.sample(self.unlabeled_indices, np.min([len(self.current_indices) * 20, len(self.unlabeled_indices)]))
        dataset_sampler = torch.utils.data.SubsetRandomSampler(candidate_indices)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, sampler=dataset_sampler, drop_last=False)
        
        gradient_lengths = []
        all_indices = []
        self.task_model.to(self.device)
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            gradient_length_single = 0
            for c in range(self.n_classes):
                Y = torch.LongTensor([c]).to(self.device)
                X,_, index = data
                X = X.to(self.device)

                loss, logits = self.task_model(X, token_type_ids=None, attention_mask=(X>0).to(self.device), labels=Y)
                predictions = torch.softmax(logits, dim=-1)[0]
                
                loss.backward()
                gradient_length_single += torch.sum(torch.pow(self.task_model.bert.embeddings.word_embeddings.weight.grad, 2)) * predictions[c]
                self.task_model.zero_grad()

            gradient_lengths.append(gradient_length_single.detach().cpu().item())
            all_indices.append(index)

        gradient_lengths = np.array(gradient_lengths)
        temp_pos = np.argpartition(-gradient_lengths, self.budget)[:self.budget].tolist()
        selected_indices = torch.Tensor(all_indices)[temp_pos]
        return selected_indices.long().tolist()

class UncertaintySampling():
    '''
    The basic uncertainty sampling query strategy, querying the examples with the least confidence.
    '''
    def __init__(self, task_model, dataset, current_indices, unlabeled_indices, budget, device, encoding_size=300, dis_epoch = 1, beta=0, autoencoder_epoch=5):
        
        self.task_model = task_model
        self.dataset = dataset
        self.current_indices = current_indices
        self.unlabeled_indices = unlabeled_indices
        self.budget = budget
        self.device = device
    
    def train(self):
        pass
    
    def query(self):
        
        candidate_indices = random.sample(self.unlabeled_indices, np.min([len(self.current_indices) * 100, len(self.unlabeled_indices)]))
        dataset_sampler = torch.utils.data.SubsetRandomSampler(candidate_indices)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=256, sampler=dataset_sampler, drop_last=False)
        
        all_indices = []
        all_logits = []

        for i, data in tqdm.tqdm(enumerate(dataloader)):
            X, Y, index = data
            X, Y = X.to(self.device), Y.to(self.device)

            with torch.no_grad():
                logits = self.task_model.predict(X, token_type_ids=None, attention_mask=(X>0).to(self.device), labels=Y)
                logits = torch.softmax(logits, dim=-1)
            
            all_logits.append(torch.max(logits, dim=-1)[0])
            all_indices.append(index)
        
        all_indices = torch.hstack(all_indices)
        all_logits = torch.hstack(all_logits).detach().cpu()
        temp_pos = np.argpartition(all_logits, self.budget)[:self.budget].tolist()

        selected_indices = all_indices[temp_pos]

        return selected_indices.long().tolist()

class BayesianUncertaintySampling():
    """
    An implementation of the Bayesian active learning method, using minimal top confidence as the decision rule.
    """
    def __init__(self, task_model, dataset, current_indices, unlabeled_indices, budget, device, encoding_size=300, dis_epoch = 1, beta=0, autoencoder_epoch=5):
    
        self.task_model = task_model
        self.dataset = dataset
        self.current_indices = current_indices
        self.unlabeled_indices = unlabeled_indices
        self.budget = budget
        self.device = device
        self.T = 20

        self.task_model.train()
    
    def train(self):
        pass
    
    def query(self):
        
        candidate_indices = random.sample(self.unlabeled_indices, np.min([len(self.current_indices) * 10, len(self.unlabeled_indices)]))
        dataset_sampler = torch.utils.data.SubsetRandomSampler(candidate_indices)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=200, sampler=dataset_sampler, drop_last=False)
        
        all_outputs = []
        all_indices = []
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            X, Y, index = data
            X, Y = X.to(self.device), Y.to(self.device)
            all_indices.append(index)

            temp_batch_output = []
            for i in range(self.T):
                with torch.no_grad():
                    logits = self.task_model(X, token_type_ids=None, attention_mask=(X>0).to(self.device), labels=Y)
                    logits = torch.softmax(logits, dim=-1).unsqueeze(0)
                    temp_batch_output.append(logits.detach().cpu())
            
            temp_batch_output = torch.cat(temp_batch_output, dim=0)
            all_outputs.append(temp_batch_output)
        
        all_indices = torch.hstack(all_indices)
        all_outputs = torch.cat(all_outputs, dim=1)
        print(all_indices.shape, all_outputs.shape)

        all_avg_outputs = torch.mean(all_outputs, dim=0)
        print(all_avg_outputs.shape)
        all_predictions = torch.max(all_avg_outputs, dim=-1)[0]

        temp_pos = np.argpartition(all_predictions, self.budget)[:self.budget].tolist()

        selected_indices = all_indices[temp_pos]

        return selected_indices.long().tolist()

class BayesianUncertaintyEntropySampling():
    """
    An implementation of the Bayesian active learning method, using minimal top confidence as the decision rule.
    """
    def __init__(self, task_model, dataset, current_indices, unlabeled_indices, budget, device, encoding_size=300, dis_epoch = 1, beta=0, autoencoder_epoch=5):
    
        self.task_model = task_model
        self.dataset = dataset
        self.current_indices = current_indices
        self.unlabeled_indices = unlabeled_indices
        self.budget = budget
        self.device = device
        self.T = 10

        self.task_model.train()
    
    def train(self):
        pass
    
    def query(self):
        
        candidate_indices = random.sample(self.unlabeled_indices, np.min([len(self.current_indices) * 10, len(self.unlabeled_indices)]))
        dataset_sampler = torch.utils.data.SubsetRandomSampler(candidate_indices)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=200, sampler=dataset_sampler, drop_last=False)
        
        all_outputs = []
        all_indices = []
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            X, Y, index = data
            X, Y = X.to(self.device), Y.to(self.device)
            all_indices.append(index)

            temp_batch_output = []
            for i in range(self.T):
                with torch.no_grad():
                    logits = self.task_model(X, token_type_ids=None, attention_mask=(X>0).to(self.device), labels=Y)
                    logits = torch.softmax(logits, dim=-1).unsqueeze(0)
                    temp_batch_output.append(logits.detach().cpu())
            
            temp_batch_output = torch.cat(temp_batch_output, dim=0)
            all_outputs.append(temp_batch_output)
        
        all_indices = torch.hstack(all_indices)
        all_outputs = torch.cat(all_outputs, dim=1)

        all_avg_outputs = torch.mean(all_outputs, dim=0)
        all_predictions = torch.sum(all_avg_outputs * torch.log(all_avg_outputs + 1e-10), dim=1)

        temp_pos = np.argpartition(all_predictions, self.budget)[:self.budget].tolist()

        selected_indices = all_indices[temp_pos]

        return selected_indices.long().tolist()

class ArcBertCosDistanceSampling():
    """
    An implementation of the Bayesian active learning method, using minimal top confidence as the decision rule.
    """
    def __init__(self, task_model, dataset, current_indices, unlabeled_indices, budget, device, encoding_size=300, dis_epoch = 1, beta=0, autoencoder_epoch=5):
    
        self.task_model = task_model
        self.dataset = dataset
        self.current_indices = current_indices
        self.unlabeled_indices = unlabeled_indices
        self.budget = budget
        self.device = device
        self.T = 100

        self.task_model.train()
    
    def train(self):
        pass
    
    def query(self):
        
        candidate_indices = random.sample(self.unlabeled_indices, np.min([len(self.current_indices) * 10, len(self.unlabeled_indices)]))
        dataset_sampler = torch.utils.data.SubsetRandomSampler(candidate_indices)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=128, sampler=dataset_sampler, drop_last=False)

        crit = torch.nn.MSELoss(reduction='none')
        
        all_outputs = []
        all_indices = []
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            X, Y, index = data

            all_indices+=index.tolist()
            X = X.to(self.device)

            centroids = F.normalize(self.task_model.classifier.weight.data)
            centroids = F.normalize(torch.mean(centroids, dim=0), dim=0).unsqueeze(0).repeat(X.shape[0],1 )

            with torch.no_grad():
                feature = self.task_model.encode(X, token_type_ids=None, attention_mask=(X>0).to(self.device))
                feature = F.normalize(feature)

                all_outputs.append(torch.sum(crit(feature, centroids), dim=-1).cpu())

        all_outputs = torch.cat(all_outputs).cpu()
        temp_pos = torch.topk(all_outputs, k=self.budget)[1].tolist()

        selected_indices = torch.Tensor(all_indices)[temp_pos]

        return selected_indices.long().tolist()

class ArcBertCosUncertaintySampling():
    """
    An implementation of the Bayesian active learning method, using minimal top confidence as the decision rule.
    """
    def __init__(self, task_model, dataset, current_indices, unlabeled_indices, budget, device, encoding_size=300, dis_epoch = 1, beta=0, autoencoder_epoch=5):
    
        self.task_model = task_model
        self.dataset = dataset
        self.current_indices = current_indices
        self.unlabeled_indices = unlabeled_indices
        self.budget = budget
        self.device = device
        self.T = 100

        self.task_model.train()
    
    def train(self):
        pass
    
    def query(self):
        
        candidate_indices = random.sample(self.unlabeled_indices, np.min([len(self.current_indices) * 10, len(self.unlabeled_indices)]))
        dataset_sampler = torch.utils.data.SubsetRandomSampler(candidate_indices)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=128, sampler=dataset_sampler, drop_last=False)
        
        all_outputs = []
        all_indices = []
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            X, Y, index = data

            all_indices+=index.tolist()
            X = X.to(self.device)

            centroids = F.normalize(self.task_model.classifier.weight.data)

            with torch.no_grad():
                feature = self.task_model.encode(X, token_type_ids=None, attention_mask=(X>0).to(self.device))
                feature = F.normalize(feature)

                coses = F.linear(feature, centroids)
                all_outputs.append(torch.abs(coses[:,0]-coses[:,1]))

        all_outputs = torch.cat(all_outputs).cpu()
        temp_pos = torch.topk(all_outputs, k=self.budget)[1].tolist()

        selected_indices = torch.Tensor(all_indices)[temp_pos]

        return selected_indices.long().tolist()

class ArcBertRejectSampling():
    """
    An implementation of the Bayesian active learning method, using minimal top confidence as the decision rule.
    """
    def __init__(self, task_model, dataset, current_indices, unlabeled_indices, budget, device, encoding_size=300, dis_epoch = 1, beta=0, autoencoder_epoch=5):
    
        self.task_model = task_model
        self.dataset = dataset
        self.current_indices = current_indices
        self.unlabeled_indices = unlabeled_indices
        self.budget = budget
        self.device = device
        self.T = 100

        self.task_model.train()
    
    def train(self):
        pass
    
    def query(self):
        
        candidate_indices = random.sample(self.unlabeled_indices, np.min([len(self.current_indices) * 10, len(self.unlabeled_indices)]))
        dataset_sampler = torch.utils.data.SubsetRandomSampler(candidate_indices)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=128, sampler=dataset_sampler, drop_last=False)
        
        all_outputs = []
        all_indices = []
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            X, Y, index = data

            all_indices+=index.tolist()
            X = X.to(self.device)

            centroids = F.normalize(self.task_model.classifier.weight.data)

            with torch.no_grad():
                feature = self.task_model.encode(X, token_type_ids=None, attention_mask=(X>0).to(self.device))
                feature = F.normalize(feature)

                coses = F.linear(feature, centroids)
                all_outputs.append((1 - coses.sigmoid()).sum(-1))

        all_outputs = torch.cat(all_outputs).cpu()
        temp_pos = torch.topk(all_outputs, k=self.budget)[1].tolist()

        selected_indices = torch.Tensor(all_indices)[temp_pos]

        return selected_indices.long().tolist()

class ArcMaxSamping():

    def __init__(self, task_model, dataset, current_indices, unlabeled_indices, budget, device, encoding_size=300, dis_epoch = 1, beta=0, autoencoder_epoch=5):
    
        self.task_model = task_model
        self.dataset = dataset
        self.current_indices = current_indices
        self.unlabeled_indices = unlabeled_indices
        self.budget = budget
        self.device = device
        self.T = 100

        self.task_model.train()
    
    def train(self):
        pass

    def query(self):
        candidate_indices = random.sample(self.unlabeled_indices, np.min([len(self.current_indices) * 10, len(self.unlabeled_indices)]))
        dataset_sampler = torch.utils.data.SubsetRandomSampler(candidate_indices)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=128, sampler=dataset_sampler, drop_last=False)

        all_outputs = []
        all_indices = []
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            X, Y, index = data

            all_indices+=index.tolist()
            X = X.to(self.device)

            centroids = F.normalize(self.task_model.classifier.weight.data)

            with torch.no_grad():
                feature = self.task_model.encode(X, token_type_ids=None, attention_mask=(X>0).to(self.device))
                feature = F.normalize(feature)

                coses = F.linear(feature, centroids)
                largest_prob = torch.max(-coses, dim=-1)[0]
                all_outputs.append(largest_prob)

        all_outputs = torch.cat(all_outputs).cpu()
        temp_pos = torch.topk(all_outputs, k=self.budget)[1].tolist()

        selected_indices = torch.Tensor(all_indices)[temp_pos]

        return selected_indices.long().tolist()


class ArcMarginSampling():
    """
    An implementation of the Bayesian active learning method, using minimal top confidence as the decision rule.
    """
    def __init__(self, task_model, dataset, current_indices, unlabeled_indices, budget, device, encoding_size=300, dis_epoch = 1, beta=0, autoencoder_epoch=5):
    
        self.task_model = task_model
        self.dataset = dataset
        self.current_indices = current_indices
        self.unlabeled_indices = unlabeled_indices
        self.budget = budget
        self.device = device
        self.T = 100

        self.task_model.train()
    
    def train(self):
        pass
    
    def query(self):
        
        candidate_indices = random.sample(self.unlabeled_indices, np.min([len(self.current_indices) * 10, len(self.unlabeled_indices)]))
        dataset_sampler = torch.utils.data.SubsetRandomSampler(candidate_indices)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=128, sampler=dataset_sampler, drop_last=False)
        
        all_outputs = []
        all_indices = []
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            X, Y, index = data

            all_indices+=index.tolist()
            X = X.to(self.device)

            centroids = F.normalize(self.task_model.classifier.weight.data)

            with torch.no_grad():
                feature = self.task_model.encode(X, token_type_ids=None, attention_mask=(X>0).to(self.device))
                feature = F.normalize(feature)

                coses = F.linear(feature, centroids)
                top_1_2, _ = torch.topk(torch.abs(torch.sigmoid(coses)), 2)
                margin = top_1_2[:, 0] - top_1_2[:, 1]
                all_outputs.append(-margin)

        all_outputs = torch.cat(all_outputs).cpu()
        temp_pos = torch.topk(all_outputs, k=self.budget)[1].tolist()

        selected_indices = torch.Tensor(all_indices)[temp_pos]

        return selected_indices.long().tolist()

class ArcBertConfusionSampling():
    """
    An implementation of the Bayesian active learning method, using minimal top confidence as the decision rule.
    """
    def __init__(self, task_model, dataset, current_indices, unlabeled_indices, budget, device, encoding_size=300, dis_epoch = 1, beta=0, autoencoder_epoch=5):
    
        self.task_model = task_model
        self.dataset = dataset
        self.current_indices = current_indices
        self.unlabeled_indices = unlabeled_indices
        self.budget = budget
        self.device = device
        self.T = 100

        self.task_model.eval()
    
    def train(self):
        pass
    
    def query(self):
        
        candidate_indices = random.sample(self.unlabeled_indices, np.min([len(self.current_indices) * 10, len(self.unlabeled_indices)]))
        dataset_sampler = torch.utils.data.SubsetRandomSampler(candidate_indices)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=128, sampler=dataset_sampler, drop_last=False)
        
        all_outputs = []
        all_indices = []
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            X, Y, index = data

            all_indices+=index.tolist()
            X = X.to(self.device)

            # centroids = F.normalize(self.task_model.protocol)
            centroids = F.normalize(self.task_model.classifier.weight.data)

            with torch.no_grad():
                feature = self.task_model.encode(X, token_type_ids=None, attention_mask=(X>0).to(self.device))
                feature = F.normalize(feature)

                prob = F.linear(feature, centroids).sigmoid()
                # logits, prob = self.task_model.predict(X, token_type_ids=None, attention_mask=(X>0).to(self.device))
                # prob = prob.sigmoid()
                # print(prob)
                all_outputs.append((1 + prob - torch.broadcast_to(prob.max(-1)[0], (prob.shape[-1], prob.shape[0])).T).sum(-1))

        all_outputs = torch.cat(all_outputs).cpu()
        temp_pos = torch.topk(all_outputs, k=self.budget)[1].tolist()

        selected_indices = torch.Tensor(all_indices)[temp_pos]

        return selected_indices.long().tolist()

class CoreSetSampling():
    """
    An implementation of the greedy core set query strategy.
    """

    def __init__(self, task_model, dataset, current_indices, unlabeled_indices, budget, device, encoding_size=300, dis_epoch = 1, beta=0, autoencoder_epoch=5):
        self.task_model = task_model
        self.dataset = dataset
        self.current_indices = current_indices
        self.unlabeled_indices = unlabeled_indices
        self.budget = budget
        self.device = device
        self.T = 100

        self.task_model.train()

    def greedy_k_center(self, labeled, unlabeled, amount):

        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount-1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices)

    def query(self):

        labeled_idx = self.current_indices
        unlabeled_idx = random.sample(self.unlabeled_indices, np.min([len(self.current_indices) * 10, len(self.unlabeled_indices)]))
        amount = self.budget

        # use the learned representation for the k-greedy-center algorithm:
        labeled_datasampler = torch.utils.data.SubsetRandomSampler(labeled_idx)
        labeled_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=128, sampler=labeled_datasampler, drop_last=False)
        labeled_reps = []
        with torch.no_grad():
            for index, data in enumerate(labeled_dataloader):
                X, Y, index = data
                X = X.to(self.device)
                rep = self.task_model.encode(X, token_type_ids=None, attention_mask=(X>0).to(self.device)).detach().cpu()
                labeled_reps.append(rep)
        labeled_reps = torch.cat(labeled_reps, dim=0)

        unlabeled_datasampler = torch.utils.data.SubsetRandomSampler(unlabeled_idx)
        unlabeled_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=128, sampler=unlabeled_datasampler, drop_last=False)
        unlabeled_reps = []
        with torch.no_grad():
            for index, data in enumerate(unlabeled_dataloader):
                X, Y, index = data
                X = X.to(self.device)
                rep = self.task_model.encode(X, token_type_ids=None, attention_mask=(X>0).to(self.device)).detach().cpu()
                unlabeled_reps.append(rep)
        unlabeled_reps = torch.cat(unlabeled_reps, dim=0)


        new_indices = self.greedy_k_center(labeled_reps, unlabeled_reps, amount)
        return new_indices.tolist()
