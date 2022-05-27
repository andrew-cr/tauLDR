import ml_collections
import yaml
from pathlib import Path
from datetime import datetime
import subprocess
import torch
import torch.utils.tensorboard as tensorboard
import glob
import os
import numpy as np
import sys
import signal


def create_experiment_folder(save_location, inner_folder_name, include_time=True):
    today_date = datetime.today().strftime(r'%Y-%m-%d')
    now_time = datetime.now().strftime(r'%H-%M-%S')
    if include_time:
        total_inner_folder_name = now_time + '_' + inner_folder_name
    else:
        total_inner_folder_name = inner_folder_name
    path = Path(save_location).joinpath(today_date).joinpath(total_inner_folder_name)
    path.mkdir(parents=True, exist_ok=True)

    checkpoint_dir, config_dir = create_inner_experiment_folders(path)

    return path, checkpoint_dir, config_dir

def create_inner_experiment_folders(path):

    checkpoint_dir = path.joinpath('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_archive_dir = checkpoint_dir.joinpath('archive')
    checkpoint_archive_dir.mkdir(exist_ok=True)

    config_dir = path.joinpath('config')
    config_dir.mkdir(exist_ok=True)

    return checkpoint_dir, config_dir


def save_config_as_yaml(cfg, save_dir):
    existing_configs = sorted(glob.glob(Path(save_dir).joinpath('config_*.yaml').as_posix()))

    if len(existing_configs) == 0:
        save_name = Path(save_dir).joinpath('config_001.yaml')
    else:
        most_recent_config = existing_configs[-1]
        most_recent_num = int(most_recent_config[-8:-5])
        save_name = Path(save_dir).joinpath('config_{0:03d}.yaml'.format(most_recent_num+1))

    with open(save_name, 'w') as f:
        yaml.dump(cfg.to_dict(), f)

def save_git_hash(save_dir):
    git_hash = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"])
    git_hash = git_hash.decode("utf-8")
    save_name = Path(save_dir).joinpath('git_hash.txt')
    with open(save_name, 'w') as f:
        f.write(git_hash)

def setup_tensorboard(save_dir, rank):
    if rank == 0:
        logs_dir = Path(save_dir).joinpath('tensorboard')
        logs_dir.mkdir(exist_ok=True)

        writer = tensorboard.writer.SummaryWriter(logs_dir)

        return writer
    else:
        return DummyWriter('none')

def save_checkpoint(checkpoint_dir, state, num_checkpoints_to_keep):
    state_to_save = {
        'model': state['model'].state_dict(),
        'optimizer': state['optimizer'].state_dict(),
        'n_iter': state['n_iter']
    }
    torch.save(state_to_save,
        checkpoint_dir.joinpath('ckpt_{0:010d}.pt'.format(state['n_iter']))
    )
    all_ckpts = sorted(glob.glob(checkpoint_dir.joinpath('ckpt_*.pt').as_posix()))
    if len(all_ckpts) > num_checkpoints_to_keep:
        for i in range(0, len(all_ckpts) - num_checkpoints_to_keep):
            os.remove(all_ckpts[i])

def save_archive_checkpoint(checkpoint_dir, state):
    save_checkpoint(checkpoint_dir.joinpath('archive'), state, 9999999)

def resume_training(resume_path, state, mapping):


    loaded_state = get_most_recent_checkpoint(resume_path, mapping)

    state['model'].load_state_dict(loaded_state['model'])
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['n_iter'] = loaded_state['n_iter']

    return state

def get_most_recent_checkpoint(dir, mapping):
    # Get most recent checkpoint
    all_ckpts = sorted(glob.glob(dir.joinpath('checkpoints').joinpath('ckpt_*.pt').as_posix()))
    most_recent_checkpoint = all_ckpts[-1]

    loaded_state = torch.load(most_recent_checkpoint, map_location=mapping)

    return loaded_state

def load_ml_collections(path):
    with open(path, 'r') as f:
        raw_dict = yaml.safe_load(f)
    return ml_collections.ConfigDict(raw_dict)

def setup_eval_folders(experiment_path, eval_name, job_id=0, task_id=0):
    eval_folder = experiment_path.joinpath('eval')
    eval_folder.mkdir(exist_ok=True)

    today_date = datetime.today().strftime(r'%Y-%m-%d')
    now_time = datetime.now().strftime(r'%H-%M-%S')

    eval_folder_name_string = today_date + '_' + now_time + '_' + 'IvI' + eval_name + 'IvI' + f'_{job_id}_{task_id}'
    eval_named_folder = eval_folder.joinpath(eval_folder_name_string)
    eval_named_folder.mkdir(exist_ok=False)
    eval_named_folder_configs = eval_named_folder.joinpath('config')
    eval_named_folder_configs.mkdir(exist_ok=False)

    return eval_folder, eval_named_folder, eval_named_folder_configs

def get_most_recent_config(config_path):
    configs = sorted(glob.glob(config_path.joinpath('config_*.yaml').as_posix()))
    assert len(configs) > 0
    return configs[-1]

class NumpyWriter():
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.scalar_data = {}
        self.figure_data = []
        self.numpy_data = []

    def add_scalar(self, name, value, idx):
        if name in self.scalar_data:
            self.scalar_data[name][0].append(idx)
            self.scalar_data[name][1].append(value)
        else:
            self.scalar_data[name] = ([idx], [value])

    def add_figure(self, name, fig, idx):
        self.figure_data.append((name, fig, idx))

    def add_numpy_data(self, name, nparray, idx):
        self.numpy_data.append((name, nparray, idx))
        
    def save_to_disk(self):
        for name in self.scalar_data:
            np_arr = np.zeros((len(self.scalar_data[name][0]), 2))
            np_arr[:, 0] = np.array(self.scalar_data[name][0])
            np_arr[:, 1] = np.array(self.scalar_data[name][1])
            np.savetxt(Path(self.save_dir).joinpath(name), np_arr)
        for item in self.figure_data:
            name, fig, idx = item
            fig.savefig(Path(self.save_dir).joinpath(name + '_' + str(idx) + '.pdf'),
                format='pdf')
        for item in self.numpy_data:
            name, nparray, idx = item
            np.save(Path(self.save_dir).joinpath(name + '_' + str(idx)), nparray)

    def close(self):
        pass

class DummyWriter():
    def __init__(self, save_dir):
        pass

    def add_scalar(self, name, value, idx):
        pass

    def add_figure(self, name, fig, idx):
        pass

    def close():
        pass


def signal_handler(sig, frame, preemption_log_path, checkpoint_dir, state, num_checkpoints_to_keep):
    time_string = datetime.today().strftime(r'%Y-%m-%d:%H-%M-%S')
    if sig == signal.SIGCONT:
        write_string = time_string + "  SIGCONT signal\n"
    elif sig == signal.SIGINT:
        write_string = time_string + "  SIGINT signal\n"
    elif sig == signal.SIGTERM:
        write_string = time_string + "  SIGTERM signal\n"
    with open(preemption_log_path, 'a') as f:
        f.write(write_string)

    save_checkpoint(checkpoint_dir, state, num_checkpoints_to_keep)

    sys.exit(0)


def setup_preemption(save_dir, checkpoint_dir, state, num_checkpoints_to_keep,
    prepare_to_resume_after_timeout):

    preemption_log_path = save_dir.joinpath('preemption_log.txt')
    time_string = datetime.today().strftime(r'%Y-%m-%d:%H-%M-%S')

    if not os.path.isfile(preemption_log_path):
        with open(preemption_log_path, 'w') as f:
            f.write(time_string + '  start of log\n')
            if prepare_to_resume_after_timeout:
                f.write(time_string + ' Expecting Timeout\n')

    signal.signal(
        signal.SIGCONT,
        lambda sig, frame: signal_handler(sig, frame, preemption_log_path,
            checkpoint_dir, state, num_checkpoints_to_keep)
    )
    signal.signal(
        signal.SIGINT,
        lambda sig, frame: signal_handler(sig, frame, preemption_log_path,
            checkpoint_dir, state, num_checkpoints_to_keep)
    )
    signal.signal(
        signal.SIGTERM,
        lambda sig, frame: signal_handler(sig, frame, preemption_log_path,
            checkpoint_dir, state, num_checkpoints_to_keep)
    )

def check_for_preempted_run(save_location, start_date, cfg,
    prepare_to_resume_after_timeout):
    print("Bookkeeping: checking for preempted run")

    # checks the runs in the start_date folder for ones which have the same 
    # config and also were preemppted. Then picks the most recent one or none
    check_dir = Path(save_location).joinpath(start_date)
    inner_run_paths = sorted(glob.glob(check_dir.as_posix() + '/*'))
    for inner_run_path in reversed(inner_run_paths):
        inner_run_path = Path(inner_run_path)

        # if the log doesn't exist then it is likely we are just starting
        # a parallel run and the other tasks haven't created it yet. so 
        # no need to preempt them
        if not os.path.isfile(inner_run_path.joinpath('preemption_log.txt')):
            continue

        with open(inner_run_path.joinpath('preemption_log.txt'), 'r') as f:
            preemption_log = f.readlines()
        if not ('SIGCONT' in preemption_log[-1] or 'Expecting Timeout' in preemption_log[-1]):
            continue
        
        # its confusing if there's changes in configs going on as well
        assert not os.path.isfile(inner_run_path.joinpath('config').joinpath('config_002.yaml')) 

        run_config = load_ml_collections(inner_run_path.joinpath('config').joinpath('config_001.yaml'))
        if not run_config == cfg:
            continue

        time_string = datetime.today().strftime(r'%Y-%m-%d:%H-%M-%S')
        with open(inner_run_path.joinpath('preemption_log.txt'), 'a') as f:
            f.write(time_string + "  resuming after preemption\n")
            if prepare_to_resume_after_timeout:
                f.write(time_string + " Expecting Timeout\n")

        print("Bookkeeping: found preempted run: ", inner_run_path)
        return inner_run_path

    print("Bookkeeping: no preempted run found")

    return Path("null")

def no_more_preemption_recovery_needed(experiment_dir):
    print("Bookkeeping: adding to preemption log that run finished")
    preemption_log_path = experiment_dir.joinpath('preemption_log.txt')
    time_string = datetime.today().strftime(r'%Y-%m-%d:%H-%M-%S')
    with open(preemption_log_path, 'a') as f:
        f.write(time_string + '  Finished Run')

