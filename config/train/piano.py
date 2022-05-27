import ml_collections

def get_config():
    save_directory = 'path/to/output' 
    pianoroll_dataset_path = 'path/to/pianoroll_dataset'


    config = ml_collections.ConfigDict()
    config.experiment_name = 'piano'
    config.save_location = save_directory

    config.device = 'cpu'
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = 'ConditionalAux'
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.01
    loss.condition_dim = 32
    loss.one_forward_pass = True

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = 'Standard'
    training.n_iters = 1000000
    training.clip_grad = True
    training.warmup = 5000

    config.data = data = ml_collections.ConfigDict()
    data.name = 'LakhPianoroll'
    data.path = pianoroll_dataset_path + '/train.npy'
    data.S = 129
    data.batch_size = 64
    data.shuffle = True
    data.shape = [256]

    config.model = model = ml_collections.ConfigDict()
    model.name = 'UniformRateSequenceTransformerEMA'

    model.num_layers = 6
    model.d_model = 128
    model.num_heads = 8
    model.dim_feedforward = 2048
    model.dropout = 0.1
    model.temb_dim = 128
    model.num_output_FFresiduals = 2
    model.time_scale_factor = 1000
    model.use_one_hot_input = True

    model.ema_decay = 0.9999

    model.rate_const = 0.03

    model.rate_sigma = 3.0
    model.Q_sigma = 20.0
    model.time_exponential = 1000.0
    model.time_base = 0.5

    model.sigma_min = 1.0
    model.sigma_max = 100.0


    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = 'Adam'
    optimizer.lr = 2e-4 #1e-4

    config.saving = saving = ml_collections.ConfigDict()

    saving.enable_preemption_recovery = False
    saving.preemption_start_day_YYYYhyphenMMhyphenDD = None

    saving.checkpoint_freq = 1000
    saving.num_checkpoints_to_keep = 2
    saving.checkpoint_archive_freq = 20000000
    # saving.log_low_freq = config.training.n_iters / 1000
    saving.log_low_freq = 1000
    saving.low_freq_loggers = []
    saving.prepare_to_resume_after_timeout = False

    return config