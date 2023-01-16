import torch
import mathtool
from torch.utils.data import TensorDataset
import tools



class Corr(torch.nn.Module):
    def __init__(self, input_size, **args):
        print(args)
        super(Corr, self).__init__()
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
        self.gamma = args.get('corr_gamma', 1.0)

    def forward(self, sample):
        """
        y: [N,]
        s: [N, 1]
        :param sample:
        :return:
        """
        x, y, s = sample
        y_predict = self.network(x)
        ce_loss = torch.nn.CrossEntropyLoss()(y_predict, y.view(-1).long())
        y_predict = torch.softmax(y_predict, dim=-1)

        c_loss = tools.corr(y_predict[:, 1] * y, s[:, 1])
        self.opt.zero_grad()
        loss = ce_loss + self.gamma*c_loss
        loss.backward()
        self.opt.step()
        return ce_loss, c_loss



    def fair_representation(self, x):
        return torch.softmax(self.network(x), dim=-1)

    def test(self, data):
        self.eval()
        with torch.no_grad():
            x2, y2, s2 = data['test']
            y_pred = self.fair_representation(x2)
            y_pred = torch.softmax(y_pred, dim=1).cpu().numpy()[:, 1]
            ms = mathtool.Statics(y_pred, y2.cpu().numpy(), s2.cpu().numpy())
            print(ms.acc)
            ms.diff()
        return ms


    def fit(self, data_dic, epoch=40):
        es = tools.EpochLoss()
        for i in range(epoch):
            dataloader = data_dic.get('loader')
            size = 0
            for sample in dataloader:
                size = sample[0].shape[0]
                break
            for sample in dataloader:
                if sample[0].shape[0] < size:
                    # skip the batch which is not full batch size
                    continue
                ls = self.forward(sample)
                es.amlt(*ls)
            es.message()


import numpy as np
def expm(seed=1, corr_gamma=0.05):
    tools.seed_everything(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ps = '../npz_data/heritage/train_dset.npz'
    cdata = np.load(ps)
    x, y, a = cdata['x_train'], cdata['y_train'], cdata['a_train']
    ls = x, y, a
    ls = list(map(lambda x: torch.from_numpy(x).float().to(device), ls))

    train_load = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*ls), batch_size=4048, shuffle=True)

    ps = '../npz_data/heritage/validate_dset.npz'
    cdata = np.load(ps)
    x, y, a = cdata['x_train'], cdata['y_train'], cdata['a_train']
    ls = x, y, a
    ls = list(map(lambda x: torch.from_numpy(x).float().to(device), ls))

    val_load = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*ls), batch_size=4048, shuffle=True)

    ps = '../npz_data/heritage/test_dset.npz'
    cdata = np.load(ps)
    x, y, a = cdata['x_train'], cdata['y_train'], cdata['a_train']
    ls = x, y, a
    ls = list(map(lambda x: torch.from_numpy(x).float().to(device), ls))
    test_load = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*ls), batch_size=4048, shuffle=True)

    print(y.shape, a.shape)
    data_d = {'loader': train_load,
              'test': list(map(lambda x: x.to(device), test_load.dataset.tensors)),
              'train': list(map(lambda x: x.to(device), train_load.dataset.tensors)),
              'verify': list(map(lambda x: x.to(device), val_load.dataset.tensors)),
              'val': val_load}
    model = Corr(data_d.get('train')[0].shape[1],
                 corr_gamma=corr_gamma).to(device)
    model.fit(data_d, 50)
    ms = model.test(data_d)

    exp = mathtool.Experiment(dataset='heritage',
                              method='CORR',
                              seed=seed,
                              sensitive='auto',
                              parameters=[corr_gamma, 0],
                              abbr='')
    exp.append(ms)

if __name__ == '__main__':
    for p in [0.0002, 0.0003, 0.0004]:
        for i in range(10):
            expm(i, p)