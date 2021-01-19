import helper
class Hyperparameters(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

def get_parameters(task):
    HPS = Hyperparameters()
    HPS['cuda'] = True
    HPS['checkpoint'] = 5
    HPS['tqdm'] = True

    if task == 'cifar10':
        HPS['dataset'] = 'cifar10' # cifar10 | stl10 | mnist
        HPS['batch_size'] = 16

        HPS['in_channels'] = 3
        HPS['h_width'] = 64
        HPS['m_width'] = 32
        HPS['z_dim'] = 16
        HPS['nb_blocks'] = 4
        HPS['nb_res_blocks'] = 3
        HPS['scale_rate'] = 2

        HPS['nb_iterations'] = 1_100_00
        HPS['lr'] = 2e-4
        HPS['decay'] = 1e-2
    elif task == 'mnist':
        HPS['dataset'] = 'mnist' # cifar10 | stl10 | mnist
        HPS['batch_size'] = 32

        HPS['in_channels'] = 1
        HPS['h_width'] = 32
        HPS['m_width'] = 16
        HPS['z_dim'] = 8
        HPS['nb_blocks'] = 2
        HPS['nb_res_blocks'] = 3
        HPS['scale_rate'] = 2

        HPS['nb_iterations'] = 100_000
        HPS['lr'] = 2e-4
        HPS['decay'] = 1e-2
    else:
        error("Unrecognized HPS task! Exiting..")
        exit()

    return HPS
