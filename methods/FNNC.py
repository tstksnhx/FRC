import torch
import mathtool
from torch.utils.data import TensorDataset
import tools


class FNNC(torch.nn.Module):
    def __init__(self, input_size, **args):
        print(args)
        super(FNNC, self).__init__()
        p_dropout = 0.
        hidden = (32, 32)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(hidden[0], hidden[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(hidden[1], 2)
        )
        self.opt = torch.optim.Adam(self.parameters())
        self.gamma = args.get('fnnc_gamma', 1.0)
        self.device = args.get('device', 'cpu')
        self.acc_loss = args.get('fnnc_loss_method', 'ce')

    def forward(self, sample):
        """
        y: [N,]
        s: [N, 1]
        :param sample:
        :return:
        """
        x, y, s = sample
        x = x.to(self.device)
        y = y.to(self.device)
        s = s.to(self.device)
        s = s[:, [1]]
        y_predict = self.network(x)
        ce_loss = torch.nn.CrossEntropyLoss()(y_predict, y.view(-1).long())
        y_predict = torch.softmax(y_predict, dim=-1)

        c0 = torch.sum(torch.mul(y_predict[:, [1]], 1 - s)) / torch.sum(1 - s)
        c1 = torch.sum(torch.mul(y_predict[:, [1]], s)) / torch.sum(s)
        dp_loss = torch.abs(c0 - c1)

        q_mean_loss = torch.sqrt(
            ((1 - (torch.sum(torch.mul(y_predict[:, [1]], y.view(-1, 1))) / torch.sum(y))) ** 2 +
             (1 - (torch.sum(torch.mul(y_predict[:, [0]], 1 - y.view(-1, 1))) / torch.sum(
                 1 - y))) ** 2) / 2)
        if self.acc_loss == 'ce':
            ls = [ce_loss, self.gamma * dp_loss]
        elif self.acc_loss == 'q_mean':
            ls = [q_mean_loss, self.gamma * dp_loss]
        else:
            raise Exception('args fnnc_loss_method must in "ce" "q_mean"')
        self.opt.zero_grad()
        loss = ls[0] + ls[1]
        loss.backward()
        self.opt.step()
        return ls

    def fair_representation(self, x):
        x = x.to(self.device)
        return torch.softmax(self.network(x), dim=-1)

    def test(self, data):
        self.eval()
        with torch.no_grad():
            x2, y2, s2 = data['test']
            y_pred = self.fair_representation(x2)
            y_pred = torch.softmax(y_pred, dim=1).cpu().numpy()[:, 1]
            print(y_pred, y2, s2)
            ms = mathtool.Statics(y_pred, y2.cpu().numpy(), s2.cpu().numpy())
            print(ms.acc)
            ms.diff()
        return ms

    def fit(self, data_dic, epoch=40):
        es = tools.EpochLoss(flag=False)
        for i in range(epoch):
            dataloader = data_dic.get('loader')
            size = 0
            for sample in dataloader:
                size = sample[0].shape[0]
                break
            for sample in dataloader:
                if sample[0].shape[0] < size:
                    continue
                ls = self.forward(sample)
                es.amlt(*ls)
            es.message()
        es.line()
import numpy as np

def expm(seed=1, fnnc_gamma=2, d=None):
    tools.seed_everything(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_d = d
    print(data_d.get('train')[-1].mean(0))
    model = FNNC(data_d.get('train')[0].shape[1],
                 fnnc_gamma=fnnc_gamma, device=device).to(device)
    model.fit(data_d, 50)
    ms = model.test(data_d)
    exp = mathtool.Experiment(dataset='heritage',
                              method='FNNC',
                              seed=seed,
                              sensitive='auto',
                              parameters=[fnnc_gamma, 1],
                              abbr='')
    exp.append(ms)

def create_h_data():
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
    return data_d
if __name__ == '__main__':
    dt_d = create_h_data()
    for s in range(3, 10):
        for g in [0, 1, 2, 3, 4, 5]:
            expm(s, g, dt_d)
