"""General utility functions"""
from time import time
from datetime import datetime
import functools
import logging

import comet_ml
import torch
from omegaconf import OmegaConf, DictConfig


def get_conf(name: str):
    """Returns yaml config file in DictConfig format

    Args:
        name: (str) name of the yaml file
    """
    OmegaConf.register_new_resolver("from_yaml", lambda address, key, _root_: OmegaConf.load(address).get(key, _root_))
    OmegaConf.register_new_resolver("eval", eval)
    name = name if name.split(".")[-1] == "yaml" else name + ".yaml"
    cfg = OmegaConf.load(name)
    OmegaConf.resolve(cfg)
    return cfg


def fix_seed(seed: int) -> None:
    """Fix reproducibility arguments"""
    seed = seed if seed else 42
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')


def to_tensor(array: Any, dtype: torch.dtype = torch.float) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(dtype)
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array).float()
    elif isinstance(array, list):
        return torch.tensor(array, dtype=dtype)
    elif isinstance(array, int):
        # returns a zero dim tensor containing one scalar
        return torch.tensor(array, dtype=dtype)
    elif isinstance(array, float):
        return torch.tensor(array, dtype=dtype)


def init_device(cfg: DictConfig):
    """Initializes the device
    
    Args:
        cfg: (DictConfig) the configuration    
    """
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the device!")
    is_cuda_available = torch.cuda.is_available()
    device = cfg.train_params.device

    if "cpu" in device:
        print(f"Performing all the operations on CPU.")
        return torch.device(device)

    elif "cuda" in device:
        if is_cuda_available:
            device_idx = device.split(":")[1]
            if device_idx == "a":
                print(
                    f"Performing all the operations on CUDA; {torch.cuda.device_count()} devices."
                )
                cfg.dataloader.batch_size *= torch.cuda.device_count()
                return torch.device(device.split(":")[0])
            else:
                print(f"Performing all the operations on CUDA device {device_idx}.")
                return torch.device(device)
        else:
            print("CUDA device is not available, falling back to CPU!")
            return torch.device("cpu")
    else:
        raise ValueError(f"Unknown {device}!")


def init_logger(cfg: DictConfig):
    """Initializes the cometml logger

    Args:
        cfg: (DictConfig) the configuration
    """ 
    
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the logger!")
    logger = None
    cfg_full = cfg
    cfg = cfg.logger
    # Check to see if there is a key in environment:
    EXPERIMENT_KEY = cfg.experiment_key

    # First, let's see if we continue or start fresh:
    CONTINUE_RUN = cfg.resume
    if EXPERIMENT_KEY and CONTINUE_RUN:
        # There is one, but the experiment might not exist yet:
        api = comet_ml.API()  # Assumes API key is set in config/env
        try:
            api_experiment = api.get_experiment_by_id(EXPERIMENT_KEY)
        except Exception:
            api_experiment = None
        if api_experiment is not None:
            CONTINUE_RUN = True
            # We can get the last details logged here, if logged:
            # step = int(api_experiment.get_parameters_summary("batch")["valueCurrent"])
            # epoch = int(api_experiment.get_parameters_summary("epochs")["valueCurrent"])

    if CONTINUE_RUN:
        # 1. Recreate the state of ML system before creating experiment
        # otherwise it could try to log params, graph, etc. again
        # ...
        # 2. Setup the existing experiment to carry on:
        logger = comet_ml.ExistingExperiment(
            previous_experiment=EXPERIMENT_KEY,
            log_env_details=cfg.log_env_details,  # to continue env logging
            log_env_gpu=True,  # to continue GPU logging
            log_env_cpu=True,  # to continue CPU logging
            auto_histogram_weight_logging=True,
            auto_histogram_gradient_logging=True,
            auto_histogram_activation_logging=True,
        )
        # Retrieved from above APIExperiment
        # self.logger.set_epoch(epoch)

    else:
        # 1. Create the experiment first
        #    This will use the COMET_EXPERIMENT_KEY if defined in env.
        #    Otherwise, you could manually set it here. If you don't
        #    set COMET_EXPERIMENT_KEY, the experiment will get a
        #    random key!
        if cfg.online:
            logger = comet_ml.Experiment(
                disabled=cfg.disabled,
                project_name=cfg.project,
                log_env_details=cfg.log_env_details,
                log_env_gpu=True,  # to continue GPU logging
                log_env_cpu=True,  # to continue CPU logging
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=True,
                auto_histogram_activation_logging=True,
            )
            logger.set_name(cfg.experiment_name)
            logger.add_tags(cfg.tags.split())
            logger.log_parameters(cfg_full)
        else:
            logger = comet_ml.OfflineExperiment(
                disabled=cfg.disabled,
                project_name=cfg.project,
                offline_directory=cfg.offline_directory,
                auto_histogram_weight_logging=True,
                log_env_details=cfg.log_env_details,
                log_env_gpu=True,  # to continue GPU logging
                log_env_cpu=True,  # to continue CPU logging
            )
            logger.set_name(cfg.experiment_name)
            logger.add_tags(cfg.tags.split())
            logger.log_parameters(cfg_full)

    return logger


def timeit(fn):
    """Calculate time taken by fn().

    A function decorator to calculate the time a function needed for completion on GPU.
    returns: the function result and the time taken
    """
    # first, check if cuda is available
    cuda = torch.cuda.is_available()
    if cuda:

        @functools.wraps(fn)
        def wrapper_fn(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            # torch.cuda.synchronize()
            # t1 = time()
            start.record()
            result = fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            # t2 = time()
            # take = t2 - t1
            return result, start.elapsed_time(end) / 1000

    else:

        @functools.wraps(fn)
        def wrapper_fn(*args, **kwargs):
            t1 = time()
            result = fn(*args, **kwargs)
            t2 = time()
            take = t2 - t1
            return result, take

    return wrapper_fn


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
