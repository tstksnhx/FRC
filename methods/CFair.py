import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import tools
import mathtool
import numpy as np


class GradReverse(Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural
    network.
    The forward part is the identity function while the backward part is the negative
    function.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


class CFairNet(torch.nn.Module):
    """
    Multi-layer perceptron with adversarial training for conditional fairness.
    """

    def __init__(self, input_size, **args):
        """

        :param configs: num_classes: 敏感类别数目
        input_dim
        hidden_layers
        """
        super(CFairNet, self).__init__()
        hidden = 32
        configs = {"num_classes": 2, "num_groups": 2,
                   "lr": 0.1,
                   "hidden_layers": [32],
                   "adversary_layers": [32],
                   "mu": 100}
        self.input_dim = input_size
        self.num_classes = 2
        self.num_hidden_layers = 1
        self.num_adversaries_layers = 1
        self.num_neurons = [self.input_dim, hidden]
        self.softmax = nn.Linear(self.num_neurons[-1], 2)
        self.num_adversaries = [self.num_neurons[-1], hidden]

        """ net arch"""
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final softmax classification layer.
        # Parameter of the conditional adversary classification layer.
        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for
        # one class label.
        self.adversaries = nn.ModuleList([nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                                         for i in range(self.num_adversaries_layers)])
                                          for _ in range(self.num_classes)])
        self.sensitive_cls = nn.ModuleList([nn.Linear(self.num_adversaries[-1], 2) for _ in range(self.num_classes)])

        self.opt = torch.optim.Adam(self.parameters())

        self.mu = args.get('mu', 10.0)
        self.device = args.get('device')

    def forward(self, sample, ws):
        reweight_target_tensor, reweight_attr_tensors = ws
        print(type(reweight_attr_tensors))
        xx, yy, ss = sample
        xx = xx.to(self.device)
        yy = yy.to(self.device)
        ss = ss.to(self.device)
        reweight_target_tensor = reweight_target_tensor.to(self.device)
        reweight_attr_tensors = reweight_attr_tensors.to(self.device)
        h_relu = xx
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probabilities.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        # Adversary classification component.
        c_losses = []
        h_relu = grad_reverse(h_relu)
        for j in range(self.num_classes):
            idx = yy == j
            c_h_relu = h_relu[idx]
            for hidden in self.adversaries[j]:
                c_h_relu = F.relu(hidden(c_h_relu))
            c_cls = F.log_softmax(self.sensitive_cls[j](c_h_relu), dim=1)
            c_losses.append(c_cls)

        loss = F.nll_loss(logprobs, yy.long(), weight=reweight_target_tensor)
        print(ss.device)
        print(c_losses[0].device)
        print(reweight_attr_tensors[0].device)
        adv_loss = torch.mean(torch.stack([F.nll_loss(c_losses[j], ss[yy == j][:, 1].long(),
                                                      weight=reweight_attr_tensors[j])
                                           for j in range(self.num_classes)]))

        loss = loss + self.mu * adv_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss, adv_loss

    def fair_representation(self, inputs):
        inputs = inputs.to(self.device)
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # return F.log_softmax(self.softmax(h_relu), dim=1)

        return self.softmax(h_relu)

    def test(self, data):
        print()
        self.eval()
        with torch.no_grad():
            x2, y2, s2 = data['test']
            y_pred = self.fair_representation(x2)
            y_pred = torch.softmax(y_pred, dim=1).cpu().numpy()[:, 1]
            ms = mathtool.Statics(y_pred, y2.cpu().numpy(), s2.cpu().numpy())
            print(ms.acc)
            ms.diff()
        return ms

    def train_(self, loader, ws, es: tools.EpochLoss = None):
        for sample in loader:
            if sample[0].shape[0] < 50:
                continue
            ls = self.forward(sample, ws)
            es.amlt(*ls)
        es.message()

    def fit(self, data, epoch=40, test=True):
        """
        base fir function... just compute the loss and next
        :param dataloader:
        :param epoch:
        :return:
        """
        dataloader = data['loader']
        es = tools.EpochLoss()
        xl, yl, sl = [], [], []
        with torch.no_grad():
            for sample in dataloader:
                a, b, c = sample
                xl.append(a.cpu().numpy())
                yl.append(b.cpu().numpy())
                sl.append(c.cpu().numpy())

        ws = self.get_W((np.vstack(xl),
                         np.hstack(yl),
                         np.vstack(sl)))
        for i in range(epoch):
            self.train_(dataloader, ws, es)
        print('*' * 30, 'TEST RESULT', '*' * 30)
        return self.test(data)

    @staticmethod
    def get_W(samples):
        X1, y1, A1 = samples

        train_y_1 = np.mean(y1)

        train_idx = (A1[:, 0] == 0)
        train_base_0 = np.mean(y1[train_idx])
        train_base_1 = np.mean(y1[~train_idx])

        reweight_target_tensor = torch.Tensor([1.0 / (1.0 - train_y_1), 1.0 / train_y_1])

        reweight_attr_0_tensor = torch.Tensor([1.0 / (1.0 - train_base_0), 1.0 / train_base_0])
        reweight_attr_1_tensor = torch.Tensor([1.0 / (1.0 - train_base_1), 1.0 / train_base_1])

        reweight_attr_tensors = [reweight_attr_0_tensor, reweight_attr_1_tensor]

        return reweight_target_tensor, reweight_attr_tensors


def expm(seed=1, mu=10):
    tools.seed_everything(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ps = '../npz_data/heritage/train_dset.npz'
    cdata = np.load(ps)
    x, y, a = cdata['x_train'], cdata['y_train'], cdata['a_train']
    ls = x, y, a
    ls = list(map(lambda x: torch.from_numpy(x).float(), ls))

    train_load = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*ls), batch_size=4048, shuffle=True)

    ps = '../npz_data/heritage/validate_dset.npz'
    cdata = np.load(ps)
    x, y, a = cdata['x_train'], cdata['y_train'], cdata['a_train']
    ls = x, y, a
    ls = list(map(lambda x: torch.from_numpy(x).float(), ls))

    val_load = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*ls), batch_size=4048, shuffle=True)

    ps = '../npz_data/heritage/test_dset.npz'
    cdata = np.load(ps)
    x, y, a = cdata['x_train'], cdata['y_train'], cdata['a_train']
    ls = x, y, a
    ls = list(map(lambda x: torch.from_numpy(x).float(), ls))
    test_load = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*ls), batch_size=4048, shuffle=True)

    print(y.shape, a.shape)
    data_d = {'loader': train_load,
              'test': test_load.dataset.tensors,
              'train': train_load.dataset.tensors,
              'verify': val_load.dataset.tensors,
              'val': val_load}
    print(data_d.get('train')[-1].mean(0))

    model = CFairNet(data_d.get('train')[0].shape[1],
                     mu=mu, device=device).to(device)
    model.fit(data_d, 100)
    ms = model.test(data_d)
    exp = mathtool.Experiment(dataset='heritage',
                              method='Cfair',
                              seed=seed,
                              sensitive='auto',
                              parameters=[mu, 0],
                              abbr='')
    exp.append(ms)


if __name__ == '__main__':
    for p in [1, 10, 100, 1000, 10000, 100000]:
        for i in range(10):
            expm(i, p)
