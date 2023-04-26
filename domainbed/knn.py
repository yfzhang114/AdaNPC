import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import faiss

class MomentumQueueClass(nn.Module):
    def __init__(self, feature_dim, queue_size, temperature, k, classes):
        super(MomentumQueue, self).__init__()
        self.queue_size = queue_size
        self.index = 0
        self.temperature = temperature
        self.k = k
        self.classes = classes

        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)
        memory_label = torch.zeros(self.queue_size).long()
        self.register_buffer('memory_label', memory_label)
    
    def update_queue(self, k_all, k_label_all):
        with torch.no_grad():
            k_all = F.normalize(k_all)
            all_size = k_all.shape[0]
            out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
            self.memory.index_copy_(0, out_ids, k_all)
            self.memory_label.index_copy_(0, out_ids, k_label_all)
            self.index = (self.index + all_size) % self.queue_size
    
    def forward(self, x, test=False):
        dist = torch.mm(F.normalize(x), self.memory.transpose(1, 0))    # B * Q, memory already normalized
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_weight, sim_indices = torch.topk(dist, k=self.k)
        sim_labels = torch.gather(self.memory_label.expand(x.size(0), -1), dim=-1, index=sim_indices)
        # sim_weight = (sim_weight / self.temperature).exp()
        if not test:
            sim_weight = F.softmax(sim_weight / self.temperature, dim=1)
        else:
            sim_weight = F.softmax(sim_weight / 0.1, dim=1)

        # counts for each class
        one_hot_label = torch.zeros(x.size(0) * self.k, self.classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(x.size(0), -1, self.classes) * sim_weight.unsqueeze(dim=-1), dim=1)
        
        pred_scores = (pred_scores + 1e-5).clamp(max=1.0)
        return pred_scores


class MomentumQueue(nn.Module):
    def __init__(self, feature_dim, queue_size, temperature, k, classes, eps_ball=1.1):
        super(MomentumQueue, self).__init__()
        self.queue_size = queue_size
        self.index = 0
        self.temperature = temperature
        self.k = k
        self.classes = classes
        self.eps_ball = eps_ball

        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)
        memory_label = torch.zeros(self.queue_size).long()
        self.register_buffer('memory_label', memory_label)
    
    def update_queue(self, k_all, k_label_all):
        with torch.no_grad():
            k_all = F.normalize(k_all)
            all_size = k_all.shape[0]
            out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
            self.memory.index_copy_(0, out_ids, k_all)
            self.memory_label.index_copy_(0, out_ids, k_label_all)
            self.index = (self.index + all_size) % self.queue_size

    def extend_test(self, k_all, k_label_all):
        with torch.no_grad():
            k_all = F.normalize(k_all)
            self.memory = torch.cat([self.memory, k_all], dim=0)
            self.memory_label = torch.cat([self.memory_label, k_label_all], dim=0)
            
    def forward(self, x, test=False):
        dist = torch.mm(F.normalize(x), self.memory.transpose(1, 0))    # B * Q, memory already normalized
        max_batch_dist, _ = torch.max(dist, 1)
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        if self.eps_ball <= 1 and max_batch_dist.min() > self.eps_ball:
            sim_weight = torch.where(dist >= self.eps_ball, dist, torch.tensor(float("-inf")).float().to(x.device))
            sim_labels = self.memory_label.expand(x.size(0), -1)
            sim_weight = F.softmax(sim_weight / self.temperature, dim=1)
            # counts for each class
            one_hot_label = torch.zeros(x.size(0) * self.memory_label.shape[0], self.classes, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.contiguous().view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
        else:
            sim_weight, sim_indices = torch.topk(dist, k=self.k)
            sim_labels = torch.gather(self.memory_label.expand(x.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = F.softmax(sim_weight / self.temperature, dim=1)

            one_hot_label = torch.zeros(x.size(0) * self.k, self.classes, device=sim_labels.device)
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(x.size(0), -1, self.classes) * sim_weight.unsqueeze(dim=-1), dim=1)
        
        pred_scores = (pred_scores + 1e-5).clamp(max=1.0)
        return pred_scores

class Model_with_Predictor(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, args, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(Model_with_Predictor, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        if args.mlp:
            self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True), # first layer
                                            nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True), # second layer
                                            self.encoder.fc,
                                            nn.BatchNorm1d(dim, affine=False)) # output layer
            self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True), # hidden layer
                                       nn.Linear(pred_dim, dim)) # output layer


    def forward(self, x):
        z = self.encoder(x)
        p = self.predictor(z)
        return p, z

def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.module.encoder.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

