# The code is modified from domainbed.algorithms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np

from domainbed.algorithms import Algorithm


ALGORITHMS = [
    'DRM',
    'DRMFull',
    'T3A', 
    'TentFull', 
    'TentNorm',  
    'TentPreBN',  # Tent-BN in the paper
    'TentClf',  # Tent-C in the paper
    'PseudoLabel', 
    'PLClf', 
    'SHOT', 
    'SHOTIM',
    'AdaNPC',
    'AdaNPCBN'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class T3A(Algorithm):
    """
    Test Time Template Adjustments (T3A)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = algorithm.featurizer
        self.classifier = algorithm.classifier

        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = hparams['filter_K']
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x, adapt=False):
        if not self.hparams['cached_loader']:
            z = self.featurizer(x)
        else:
            z = x
        if adapt:
            # online adaptation
            p = self.classifier(z)
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
        
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s))))
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data


class TentFull(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.model, self.optimizer = self.configure_model_optimizer(algorithm, alpha=hparams['alpha'])
        self.steps = hparams['gamma']
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False
    
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                if self.hparams['cached_loader']:
                    outputs = self.forward_and_adapt(x, self.model.classifier, self.optimizer)
                else:
                    self.model.featurizer.eval()
                    outputs = self.forward_and_adapt(x, self.model, self.optimizer)
                    self.model.featurizer.train()
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        # adapt
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        optimizer.step()
        return outputs

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        adapted_algorithm.featurizer = configure_model(adapted_algorithm.featurizer)
        params, param_names = collect_params(adapted_algorithm.featurizer)
        optimizer = torch.optim.Adam(
            params, 
            lr=algorithm.hparams["lr"]*alpha,
            weight_decay=algorithm.hparams['weight_decay']
        )
        # adapted_algorithm.classifier.predict = lambda self, x: self(x)
        return adapted_algorithm, optimizer

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


class TentNorm(TentFull):
    def forward(self, x, adapt=False):
        if self.hparams['cached_loader']:
            outputs = self.model.classifier(x)
        else:
            outputs = self.model(x)
        return outputs


class TentPreBN(TentFull):
    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        adapted_algorithm.classifier = PreBN(adapted_algorithm.classifier, adapted_algorithm.featurizer.n_outputs)
        adapted_algorithm.network = torch.nn.Sequential(adapted_algorithm.featurizer, adapted_algorithm.classifier)
        optimizer = torch.optim.Adam(
            adapted_algorithm.classifier.bn.parameters(), 
            lr=algorithm.hparams["lr"] * alpha,
            weight_decay=algorithm.hparams['weight_decay']
        )
        return adapted_algorithm, optimizer


class TentClf(TentFull):
    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            adapted_algorithm.classifier.parameters(), 
            lr=algorithm.hparams["lr"]  * alpha,
            weight_decay=algorithm.hparams['weight_decay']
        )
        adapted_algorithm.classifier.predict = lambda self, x: self(x)
        return adapted_algorithm, optimizer


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None   
    return model


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=False)
    optimizer.load_state_dict(optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class PreBN(torch.nn.Module):
    def __init__(self, m, num_features, **kwargs):
        super().__init__()
        self.m = m
        self.bn = torch.nn.BatchNorm1d(num_features, **kwargs)
        self.bn.requires_grad_(True)
        self.bn.track_running_stats = False
        self.bn.running_mean = None
        self.bn.running_var = None
        
    def forward(self, x):
        x = self.bn(x)
        return self.m(x)

    def predict(self, x):
        return self(x)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


class PseudoLabel(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        gamma (int) : number of updates
        """
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.model, self.optimizer = self.configure_model_optimizer(algorithm, alpha=hparams['alpha'])
        self.beta = hparams['beta']
        self.steps = hparams['gamma']
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False
    
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                if self.hparams['cached_loader']:
                    outputs = self.forward_and_adapt(x, self.model.classifier, self.optimizer)
                else:
                    self.model.featurizer.eval()
                    outputs = self.forward_and_adapt(x, self.model, self.optimizer)
                    self.model.featurizer.train()
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        # adapt
        py, y_prime = F.softmax(outputs, dim=-1).max(1)
        flag = py > self.beta
        
        loss = F.cross_entropy(outputs[flag], y_prime[flag])
        loss.backward()
        optimizer.step()
        return outputs

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            adapted_algorithm.parameters(), 
            lr=algorithm.hparams["lr"]  * alpha,
            weight_decay=algorithm.hparams['weight_decay']
        )
        return adapted_algorithm, optimizer

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


class PLClf(PseudoLabel):
    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            adapted_algorithm.classifier.parameters(), 
            lr=algorithm.hparams["lr"]  * alpha,
            weight_decay=algorithm.hparams['weight_decay']
        )
        return adapted_algorithm, optimizer

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


class SHOT(Algorithm):
    """
    "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation"
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        theta (float) : clf coefficient
        gamma (int) : number of updates
        """
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.model, self.optimizer = self.configure_model_optimizer(algorithm, alpha=hparams['alpha'])
        self.beta = hparams['beta']
        self.theta = hparams['theta']
        self.steps = hparams['gamma']
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False
    
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                if self.hparams['cached_loader']:
                    outputs = self.forward_and_adapt(x, self.model.classifier, self.optimizer)
                else:
                    self.model.featurizer.eval()
                    outputs = self.forward_and_adapt(x, self.model, self.optimizer)
                    self.model.featurizer.train()
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        
        loss = self.loss(outputs)
        loss.backward()
        optimizer.step()
        return outputs
    
    def loss(self, outputs):
        # (1) entropy
        ent_loss = softmax_entropy(outputs).mean(0)

        # (2) diversity
        softmax_out = F.softmax(outputs, dim=-1)
        msoftmax = softmax_out.mean(dim=0)
        ent_loss += torch.sum(msoftmax * torch.log(msoftmax + 1e-5))

        # (3) pseudo label
        # adapt
        py, y_prime = F.softmax(outputs, dim=-1).max(1)
        flag = py > self.beta
        clf_loss = F.cross_entropy(outputs[flag], y_prime[flag])

        loss = ent_loss + self.theta * clf_loss
        return loss

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            adapted_algorithm.featurizer.parameters(), 
            # adapted_algorithm.classifier.parameters(), 
            lr=algorithm.hparams["lr"]  * alpha,
            weight_decay=algorithm.hparams['weight_decay']
        )
        return adapted_algorithm, optimizer

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


class SHOTIM(SHOT):    
    def loss(self, outputs):
        # (1) entropy
        ent_loss = softmax_entropy(outputs).mean(0)

        # (2) diversity
        softmax_out = F.softmax(outputs, dim=-1)
        msoftmax = softmax_out.mean(dim=0)
        ent_loss += torch.sum(msoftmax * torch.log(msoftmax + 1e-5))

        return ent_loss

class AdaNPC(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        gamma (int) : number of updates
        """
        super().__init__(input_shape, num_classes, num_domains, hparams)
        from domainbed.knn import MomentumQueue

        self.beta = hparams['beta']
        self.model = algorithm
        self.classifier = MomentumQueue(self.model.featurizer.n_outputs, 1, temperature=hparams['temperature'], k=self.hparams['k'], classes=num_classes)

    def forward(self, x, adapt=False):
        if adapt:
            outputs = self.forward_and_adapt(x)
        else:
            outputs = self.model.classifier(x)

        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        p = self.classifier(x)
        confidences, predict = p.softmax(1).max(1)
        predict = predict[confidences >= self.beta]
        if predict.shape[0] > 0:
            self.classifier.extend_test(x[confidences >= self.beta], predict)
        return p

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.classifier.memory = self.classifier.memory[:self.classifier.queue_size,:]
        self.classifier.memory_label = self.classifier.memory_label[:self.classifier.queue_size]

    def reset_params(self, hparams):
        self.beta = hparams['beta']
        self.classifier.k = hparams['k']
        self.classifier.temperature = hparams['temperature']
        self.reset()

class AdaNPCBN(AdaNPC):
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        gamma (int) : number of updates
        """
        super().__init__(input_shape, num_classes, num_domains, hparams, algorithm)
        from domainbed.knn import MomentumQueue

        self.beta = 0.1
        self.model = algorithm
        self.bn, self.optimizer = self.configure_model_optimizer(algorithm, alpha=0.01)
        self.steps = 3
        self.classifier = MomentumQueue(self.model.featurizer.n_outputs, 1, temperature=0.01, k=self.hparams['k'], classes=num_classes, eps_ball=1.1)
        self.model_state, self.optimizer_state =  copy_model_and_optimizer(self.model, self.optimizer)

    def reset_params(self, hparams):
        self.beta = hparams['beta']
        self.classifier.k = hparams['k']
        self.classifier.temperature = hparams['temperature']
        self.steps = hparams['gamma']
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.bn, self.optimizer = self.configure_model_optimizer(self.model, alpha=hparams['alpha'])
        self.reset()

    def forward(self, x, adapt=False):
        if adapt:
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x)
        else:
            outputs = self.model.classifier(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        x = self.bn(x)
        p = self.classifier(x)
        confidences, predict = p.softmax(1).max(1)
        predict = predict[confidences >= self.beta]
        if predict.shape[0] > 0:
            self.classifier.extend_test(x[confidences >= self.beta], predict)
        self.optimizer.zero_grad()
        loss = softmax_entropy(p).mean(0)
        loss.backward()
        self.optimizer.step()
        return p

    def configure_model_optimizer(self, algorithm, alpha):
        bn = nn.BatchNorm1d(algorithm.featurizer.n_outputs).cuda()
        optimizer = torch.optim.Adam(
            bn.parameters(), 
            lr=algorithm.hparams["lr"]*alpha,
            weight_decay=algorithm.hparams['weight_decay']
        )
        # adapted_algorithm.classifier.predict = lambda self, x: self(x)
        return bn, optimizer

class DRMFull(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        gamma (int) : number of updates
        """
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.model, self.optimizer = self.configure_model_optimizer(algorithm, alpha=hparams['alpha'])
        self.beta = hparams['beta']
        self.steps = hparams['step']
        self.gamma = hparams['gamma']
        self.label = hparams['label']
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False
    
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                if self.hparams['cached_loader']:
                    self.model.featurizer.eval()
                    outputs = self.forward_and_adapt(x, self.model, self.optimizer, cached_loader=self.hparams['cached_loader'])
                    self.model.featurizer.train()
                else:
                    outputs = self.forward_and_adapt(x, self.model, self.optimizer, cached_loader=self.hparams['cached_loader'])
                    
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer, cached_loader=False):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        if not cached_loader:
            x = model.featurizer(x)
        if self.label == 'own':
            outputs = self.entropy_predict_label_individual_entropy(x, model, optimizer)
        elif self.label == 'last':
            outputs = self.entropy_predict_label_by_final(x, model, optimizer)
        elif self.label == 'uniform':
            outputs = self.entropy_predict_label_uniform(x, model, optimizer)
        elif self.label == 'drm':
            outputs = self.entropy_predict_label_drm(x, model, optimizer)
        else:
            raise NotImplementedError

        return outputs

    def entropy_predict(self, logits, model, optimizer):
        entropy = torch.tensor(1e10)
        result = None
        ents, y_hats = [], []
        loss = torch.zeros(1)
        for i in range(model.num_domains + 1):
            y_hat = model.classifier_list[i](logits)
            confidences, predict = y_hat.softmax(1).max(1)
            predict = predict[confidences >= self.beta]
            if predict.shape[0] > 0:
                if loss == 0:
                    loss = F.cross_entropy(y_hat[confidences >= self.beta], predict)
                else:
                    loss += F.cross_entropy(y_hat[confidences >= self.beta], predict)
            ent = model.softmax_entropy(y_hat).mean()
            ents.append(ent.item())

            y_hats.append(torch.nn.functional.normalize(y_hat, dim=0))
            if  ent < entropy:
                entropy = ent
                result = y_hat

        if loss > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.gamma >=0 :
            com_result = torch.zeros(y_hat.shape[0], y_hat.shape[1]).cuda()
            weight = 1.0 / ( np.array(ents) ** self.gamma)
            weight /= np.sum(weight)
            
            for i in range(model.num_domains):
                com_result += weight[i] * y_hats[i]
            return com_result

        return result

    def entropy_predict_label_drm(self, logits, model, optimizer):
        result = None
        entropy, y_hats, y_hats_pre = torch.zeros((model.num_domains + 1, logits.shape[0])).cuda(), torch.zeros(model.num_domains + 1, logits.shape[0], model.num_class).cuda(), []
        loss = torch.zeros(1)
        for i in range(model.num_domains + 1):
            y_hat = model.classifier_list[i](logits)
            
            y_hats[i] = torch.nn.functional.softmax(y_hat, dim=1)
            y_hats_pre.append(y_hat)
            entropy[i] = model.softmax_entropy(y_hat)

        com_result = torch.zeros(y_hat.shape[0], y_hat.shape[1]).cuda()
        if self.gamma >=0 :
            weight = 1.0 / ( entropy ** self.gamma)
            weight = torch.nn.functional.normalize(weight, p=1, dim=0)
            for i in range(model.num_domains):
                com_result += torch.mul( y_hats[i].T, weight[i]).T
        else:
            for i in range(y_hats.shape[1]):
                idx = entropy[:,i].argmin()
                com_result[i] = y_hats[idx, i]

        confidences, predict = com_result.softmax(1).max(1)
        predict = predict[confidences >= self.beta]
        for i in range(model.num_domains + 1):
            if predict.shape[0] > 0:
                if loss == 0:
                    loss = F.cross_entropy(y_hats_pre[i][confidences >= self.beta], predict)
                else:
                    loss += F.cross_entropy(y_hats_pre[i][confidences >= self.beta], predict)
        if loss > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        return com_result

    def entropy_predict_label_uniform(self, logits, model, optimizer):
        result = None
        entropy, y_hats, y_hats_pre = torch.zeros((model.num_domains + 1, logits.shape[0])).cuda(), torch.zeros(model.num_domains + 1, logits.shape[0], model.num_class).cuda(), []
        loss = torch.zeros(1)
        for i in range(model.num_domains + 1):
            y_hat = model.classifier_list[i](logits)
            
            y_hats[i] = torch.nn.functional.softmax(y_hat, dim=1)
            y_hats_pre.append(y_hat)
            entropy[i] = model.softmax_entropy(y_hat)
    
        y = y_hats.mean(dim=0)
        for i in range(model.num_domains + 1):
            confidences, predict = y.softmax(1).max(1)
            predict = predict[confidences >= self.beta]
            if predict.shape[0] > 0:
                if loss == 0:
                    loss = F.cross_entropy(y_hats_pre[i][confidences >= self.beta], predict)
                else:
                    loss += F.cross_entropy(y_hats_pre[i][confidences >= self.beta], predict)
        if loss > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        com_result = torch.zeros(y_hat.shape[0], y_hat.shape[1]).cuda()
        if self.gamma >=0 :
            weight = 1.0 / ( entropy ** self.gamma)
            weight = torch.nn.functional.normalize(weight, p=1, dim=0)
            for i in range(model.num_domains):
                com_result += torch.mul( y_hats[i].T, weight[i]).T
        else:
            for i in range(y_hats.shape[1]):
                idx = entropy[:,i].argmin()
                com_result[i] = y_hats[idx, i]
        
        return com_result

    def entropy_predict_label_by_final(self, logits, model, optimizer):
        result = None
        entropy, y_hats = torch.zeros((model.num_domains + 1, logits.shape[0])).cuda(), torch.zeros(model.num_domains + 1, logits.shape[0], model.num_class).cuda()
        loss = torch.zeros(1)
        for i in range(model.num_domains + 1):
            y_hat = model.classifier_list[i](logits)
            y = model.classifier_list[-1](logits)
            confidences, predict = y.softmax(1).max(1)
            predict = predict[confidences >= self.beta]
            if predict.shape[0] > 0:
                if loss == 0:
                    loss = F.cross_entropy(y_hat[confidences >= self.beta], predict)
                else:
                    loss += F.cross_entropy(y_hat[confidences >= self.beta], predict)
            y_hats[i] = torch.nn.functional.softmax(y_hat, dim=1)
            entropy[i] = model.softmax_entropy(y_hat)

        if loss > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        com_result = torch.zeros(y_hat.shape[0], y_hat.shape[1]).cuda()
        if self.gamma >=0 :
            weight = 1.0 / ( entropy ** self.gamma)
            weight = torch.nn.functional.normalize(weight, p=1, dim=0)
            for i in range(model.num_domains):
                com_result += torch.mul( y_hats[i].T, weight[i]).T
        else:
            for i in range(y_hats.shape[1]):
                idx = entropy[:,i].argmin()
                com_result[i] = y_hats[idx, i]
        
        return com_result

    def entropy_predict_label_individual_entropy(self, logits, model, optimizer):
        result = None
        entropy, y_hats = torch.zeros((model.num_domains + 1, logits.shape[0])).cuda(), torch.zeros(model.num_domains + 1, logits.shape[0], model.num_class).cuda()
        loss = torch.zeros(1)
        for i in range(model.num_domains + 1):
            y_hat = model.classifier_list[i](logits)
            confidences, predict = y_hat.softmax(1).max(1)
            predict = predict[confidences >= self.beta]
            if predict.shape[0] > 0:
                if loss == 0:
                    loss = F.cross_entropy(y_hat[confidences >= self.beta], predict)
                else:
                    loss += F.cross_entropy(y_hat[confidences >= self.beta], predict)
            y_hats[i] = torch.nn.functional.softmax(y_hat, dim=1)
            entropy[i] = model.softmax_entropy(y_hat)

        if loss > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        com_result = torch.zeros(y_hat.shape[0], y_hat.shape[1]).cuda()
        if self.gamma >=0 :
            weight = 1.0 / ( entropy ** self.gamma)
            weight = torch.nn.functional.normalize(weight, p=1, dim=0)
            for i in range(model.num_domains):
                com_result += torch.mul( y_hats[i].T, weight[i]).T
        else:
            for i in range(y_hats.shape[1]):
                idx = entropy[:,i].argmin()
                com_result[i] = y_hats[idx, i]
        
        return com_result

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            adapted_algorithm.parameters(), 
            lr=algorithm.hparams["lr"]  * alpha,
            weight_decay=algorithm.hparams['weight_decay']
        )
        return adapted_algorithm, optimizer

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

class DRM(DRMFull):
    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            adapted_algorithm.classifier_list.parameters(), 
            lr=algorithm.hparams["lr"]  * alpha,
            weight_decay=algorithm.hparams['weight_decay']
        )
        return adapted_algorithm, optimizer

    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()
            for _ in range(self.steps):
                self.model.featurizer.eval()
                outputs = self.forward_and_adapt(x, self.model, self.optimizer, cached_loader=self.hparams['cached_loader'])
                self.model.featurizer.train()
                    
        else:
            outputs = self.model.classifier(x)
        return outputs