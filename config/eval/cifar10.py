import ml_collections

def get_config():

    datasets_folder = 'path/to/datasets'
    model_location = 'path/to/ckpt_0001999999.pt'
    model_config_location = 'path/to/config/config_001.yaml'

    config = ml_collections.ConfigDict()
    config.eval_name = 'CIFAR10'
    config.train_config_overrides = [
        [['device'], 'cpu'],
        [['data', 'root'], datasets_folder],
        [['distributed'], False]
    ]
    config.train_config_path = model_config_location
    config.checkpoint_path = model_location

    config.device = 'cuda'

    config.data = data = ml_collections.ConfigDict()
    data.name = 'DiscreteCIFAR10'
    data.root = datasets_folder
    data.train = True
    data.download = True
    data.S = 256
    data.batch_size = 16
    data.shuffle = True
    data.shape = [3,32,32]
    data.random_flips = False

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = 'PCTauLeaping' # TauLeaping or PCTauLeaping
    sampler.num_steps = 500
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = 'gaussian'
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = 1.5
    sampler.corrector_entry_time = 0.1

    return config