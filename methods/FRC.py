from torch.nn import Linear
import torch
import tools
from sklearn.linear_model import LogisticRegression
import mathtool


class FRC(torch.nn.Module):
    """
    p1 = 0 means be beta-VAE
    beta-VAE:
        p0 = x (not 0), p1 = 0 result is bad both for acc and dp
        p0 = x (x > 0), p1 = 0 have dp but low acc
    """

    # def __init__(self, input_size, h_dim=64, z_dim=10, gs=None):
    def __init__(self, inputs_size, **kwargs):
        super(FRC, self).__init__()
        p_dropout = 0.2
        inputs_size = inputs_size
        h_dim = kwargs.get('h_dim', 68)
        self.z_dim = kwargs.get('z_dim', 10)  # !

        self.device = kwargs.get('device')

        self.loss_params = kwargs.get('parameters', [1, 1])
        self.fc1 = Linear(inputs_size, h_dim)
        self.fc2 = Linear(h_dim, self.z_dim)
        self.fc3 = Linear(h_dim, self.z_dim)
        self.fc4 = Linear(self.z_dim, h_dim)
        self.fc5 = Linear(h_dim, inputs_size)

        self.pre = torch.nn.Sequential(
            torch.nn.Linear(inputs_size, h_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(h_dim, 2)
        )

        self.opt = torch.optim.Adam(self.parameters())
        corr = kwargs.get('sens', [1, 0])
        ct = len(corr)
        sp = (self.z_dim - ct) // ct
        sk = (self.z_dim - ct) - sp * (ct - 1)
        ircorr = [[k] * sp for k in corr[:-1]]
        ircorr.append([corr[-1]] * sk)
        self.split = ct
        hu = []
        for item in ircorr:
            hu += item
        self.corr = kwargs.get('CORR', corr)
        self.irr = kwargs.get('irr', hu)
        print(self.corr)
        print(self.irr)

    def test(self, data):
        self.eval()
        x2, y2, s2 = data['test']
        x1, y1, s1 = data['train']
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        y2 = y2.to(self.device)
        y1 = y1.to(self.device)
        s2 = s2.to(self.device)
        s1 = s1.to(self.device)
        print()
        with torch.no_grad():
            lr = LogisticRegression(max_iter=2000)
            in_ = self.fair_representation(x1).cpu().numpy()
            print(in_.shape)
            la_ = y1.cpu().numpy()
            lr.fit(in_, la_)
            y_pred = lr.predict_proba(self.fair_representation(x2).cpu().numpy())[:, 1]

            ms = mathtool.Statics(y_pred, y2.cpu().numpy(), s2.cpu().numpy())
            ms.diff()
            print('acc:', ms.acc)
        return ms

    def verify(self, data):
        self.eval()
        x2, y2, s2 = data['verify']
        x1, y1, s1 = data['train']
        with torch.no_grad():
            lr = LogisticRegression(max_iter=1500)
            in_ = self.fair_representation(x1).cpu().numpy()
            la_ = y1.cpu().numpy()
            lr.fit(in_, la_)
            y_pred = lr.predict_proba(self.fair_representation(x2).cpu().numpy())[:, 1]
            ms = mathtool.Statics(y_pred, y2.cpu().numpy(), s2.cpu().numpy())
            print('acc:', ms.acc)
        return ms

    def fit(self, data_dic, epoch_num=50):
        es = tools.EpochLoss(flag=False)
        for i in range(epoch_num):
            for data in data_dic.get('loader'):
                if data[0].shape[0] < 50:
                    continue
                bs = data[-1].cpu().detach().numpy().mean(0).tolist()
                if 0 in bs or 1 in bs:
                    continue
                loss = self.forward(data)

                if es:
                    es.amlt(*loss)
            es.message()

    @staticmethod
    def corr(x, y):
        """相关系数 越低，越不相关"""
        xm, ym = torch.mean(x), torch.mean(x)
        xvar = torch.sum((x - xm) ** 2) / x.shape[0]
        yvar = torch.sum((y - ym) ** 2) / x.shape[0]
        return torch.abs(torch.sum((x - xm) * (y - ym)) / (xvar * yvar) ** 0.5)

    def encode(self, x):
        "get encode mu, logvar"
        h = torch.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        "reparameterize z from mu, log_var"
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        "decode x from z"
        h = torch.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def representation(self, x):
        mu, log_var, x_reconst, z = self.out(x)
        return z

    def fair_representation(self, x):
        z = self.representation(x)
        # z[:, :self.split] = torch.randn_like(z[:, :self.split])
        return z[:, self.split:]

    def unfair_representation(self, x):
        z = self.representation(x)
        return z[:, :self.split]

    def out(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return mu, log_var, x_reconst, z

    @staticmethod
    def corr_loss_fun(sensitive, y, save_sensitive_ids, remove_sensitive_ids):
        """
        :param sensitive: sensitive array
        :param y: representation array
        :param save_sensitive_ids: y index that need save sensitive message
        :param remove_sensitive_ids: y index that need remove sensitive message
        :return: total CORR
        """
        ans = 0
        ii = 0

        for i in save_sensitive_ids:
            ans -= tools.corr(sensitive[:, [i]], y[:, [ii]])
            ii += 1
        for i in remove_sensitive_ids:
            ans += tools.corr(sensitive[:, [i]], y[:, [ii]])
            ii += 1
        return ans

    def forward(self, data):
        x, y, s = data
        x = x.to(self.device)
        y = y.to(self.device)
        s = s.to(self.device)
        mu, log_var, x_reconst, z = self.out(x)
        y_ = self.pre(x_reconst)

        reconst_loss = torch.nn.MSELoss(reduction='sum')(x_reconst, x)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = reconst_loss + self.loss_params[0] * kl_div

        corr_loss = self.corr_loss_fun(s, z, self.corr, self.irr)
        loss += self.loss_params[1] * corr_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return reconst_loss, kl_div, corr_loss


import numpy as np


# def expm(seed=1, p=[0.01, 1], dim=40):
#     tools.seed_everything(seed)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     ps = '../npz_data/heritage/train_dset.npz'
#     cdata = np.load(ps)
#     x, y, a = cdata['x_train'], cdata['y_train'], cdata['a_train']
#     ls = x, y, a
#     ls = list(map(lambda x: torch.from_numpy(x).float(), ls))
#
#     train_load = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*ls), batch_size=4048, shuffle=True)
#
#     ps = '../npz_data/heritage/validate_dset.npz'
#     cdata = np.load(ps)
#     x, y, a = cdata['x_train'], cdata['y_train'], cdata['a_train']
#     ls = x, y, a
#     ls = list(map(lambda x: torch.from_numpy(x).float(), ls))
#
#     val_load = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*ls), batch_size=4048, shuffle=True)
#
#     ps = '../npz_data/heritage/test_dset.npz'
#     cdata = np.load(ps)
#     x, y, a = cdata['x_train'], cdata['y_train'], cdata['a_train']
#     ls = x, y, a
#     ls = list(map(lambda x: torch.from_numpy(x).float(), ls))
#     test_load = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*ls), batch_size=4048, shuffle=True)
#
#     print(y.shape, a.shape)
#     data_d = {'loader': train_load,
#               'test': test_load.dataset.tensors,
#               'train': train_load.dataset.tensors,
#               'verify': val_load.dataset.tensors,
#               'val': val_load}
#     print(x.shape)
#     model = FRC(inputs_size=data_d.get('train')[0].shape[1], parameters=p, z_dim=dim,
#                 # CORR=[3], irr=[3]*9,
#                 device=device,
#                 sens=[0, 1]
#                 ).to(device)
#     model.to(device)
#     model.fit(data_d, 50)
#     ms = model.test(data_d)
#     exp = mathtool.Experiment(dataset='heritage',
#                               method='FRC',
#                               seed=seed,
#                               sensitive='auto',
#                               parameters=p,
#                               abbr='')
#     exp.append(ms)



def expm(seed=1, p=[0.01, 1]):
    tools.seed_everything(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data_d = tools.load_single('../npz_data/ada.npz')
    pt = data_d.get('train')[-1].mean(0)
    print(pt)

    model = FRC(inputs_size=data_d.get('train')[0].shape[1], parameters=p, z_dim=20, device=device,
                # CORR=[3], irr=[3]*9,
                sens=[1, 0]
                ).to(device)
    model.fit(data_d, 200)
    ms = model.test(data_d)
    exp = mathtool.Experiment(dataset='compas',
                              method='frc',
                              seed=seed,
                              sensitive='auto',
                              parameters=p,
                              abbr='')
    exp.append(ms)


if __name__ == '__main__':

    expm(1, [0.00, 0.1])
