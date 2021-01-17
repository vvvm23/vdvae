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
HPS['checkpoint'] = 1
HPS['tqdm'] = True

HPS['dataset'] = 'mnist' # cifar10 | mnist
HPS['batch_size'] = 32

HPS['in_channels'] = 1
HPS['h_width'] = 64
HPS['m_width'] = 32
HPS['z_dim'] = 16
HPS['nb_blocks'] = 3
HPS['nb_res_blocks'] = 2
HPS['scale_rate'] = 2

HPS['nb_iterations'] = 1_100_00
HPS['lr'] = 2e-4
HPS['decay'] = 1e-2
