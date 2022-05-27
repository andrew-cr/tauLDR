import ml_collections

def get_config():
    save_directory = 'path/to/output' 
    datasets_folder = 'path/to/datasets'



    config = ml_collections.ConfigDict()
    config.experiment_name = 'cifar10'
    config.save_location = save_directory

    config.init_model_path = None

    config.device = 'cuda'
    config.distributed = True
    config.num_gpus = 4

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = 'GenericAux'
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.01
    loss.one_forward_pass = True

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = 'Standard'
    training.n_iters = 2000000
    training.clip_grad = True
    training.warmup = 5000

    config.data = data = ml_collections.ConfigDict()
    data.name = 'DiscreteCIFAR10'
    data.root = datasets_folder
    data.train = True
    data.download = True
    data.S = 256
    data.batch_size = 128
    data.shuffle = True
    data.shape = [3,32,32]
    data.random_flips = True

    config.model = model = ml_collections.ConfigDict()
    model.name = 'GaussianTargetRateImageX0PredEMA'

    model.ema_decay = 0.9999 #0.9999

    model.ch = 128
    model.num_res_blocks = 2
    model.num_scales = 4
    model.ch_mult = [1, 2, 2, 2]
    model.input_channels = 3
    model.scale_count_to_put_attn = 1
    model.data_min_max = [0, 255]
    model.dropout = 0.1
    model.skip_rescale = True
    model.time_embed_dim = model.ch
    model.time_scale_factor = 1000
    model.fix_logistic = False

    model.rate_sigma = 6.0
    model.Q_sigma = 512.0
    model.time_exponential = 100.0
    model.time_base = 3.0


    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = 'Adam'
    optimizer.lr = 2e-4

    config.saving = saving = ml_collections.ConfigDict()

    saving.enable_preemption_recovery = False
    saving.preemption_start_day_YYYYhyphenMMhyphenDD = None
    saving.checkpoint_freq = 1000
    saving.num_checkpoints_to_keep = 2
    saving.checkpoint_archive_freq = 200000
    saving.log_low_freq = 10000
    saving.low_freq_loggers = ['denoisingImages']
    saving.prepare_to_resume_after_timeout = False

    return config
