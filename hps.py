class Hyperparameters(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

HPS = Hyperparameters()
HPS['cuda'] = True

HPS['dataset'] = 'cifar10'
HPS['batch_size'] = 64

HPS['in_channels'] = 3
HPS['h_width'] = 256
HPS['m_width'] = 96
HPS['z_dim'] = 64
HPS['nb_blocks'] = 4
HPS['nb_res_blocks'] = 3
HPS['scale_rate'] = 2

HPS['nb_epochs'] = 200
HPS['lr'] = 3e-4
