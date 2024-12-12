#!/usr/bin/env python
import sys
import os
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle
# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config
from runner.share_jsbsim_runner import ShareJSBSimRunner
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import DummyVecEnv, ShareDummyVecEnv


def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.env_name == "MultipleCombat":
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return DummyVecEnv([get_env_fn(0)])


def parse_args(args, parser):
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--episode-length', type=int, default=1000,
                       help="the max length of an episode")
    group.add_argument('--scenario-name', type=str, default='singlecombat_vsbaseline',
                       help="number of fighters controlled by RL policy")
    group.add_argument('--num-agents', type=int, default=1,
                       help="number of fighters controlled by RL policy")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    assert all_args.model_dir is not None
    # seed
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results") \
        / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    curr_run = 'render'
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name)
                              + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # env init
    envs = make_render_env(all_args)
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "eval_envs": None,
        "envs": envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.env_name == "MultipleCombat":
        runner = ShareJSBSimRunner(config)
    else:
        if all_args.use_selfplay:
            from runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner as Runner
        else:
            from runner.jsbsim_runner import JSBSimRunner as Runner
        runner = Runner(config)
    runner.render()

    # post process
    envs.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    # Fixed parameters (equivalent to the system args in the shell script)
    env = "SingleControl"
    scenario = "1/heading"
    algo = "ppo"
    exp = "v1"
    seed = 5

    # Additional parameters
    n_training_threads = 1
    n_rollout_threads = 4
    cuda = True
    log_interval = 1
    save_interval = 1
    num_mini_batch = 5
    buffer_size = 3000
    num_env_steps = 1e8
    lr = 3e-4
    gamma = 0.99
    ppo_epoch = 4
    clip_params = 0.2
    max_grad_norm = 2
    entropy_coef = 1e-3
    hidden_size = "128 128"
    act_hidden_size = "128 128"
    recurrent_hidden_size = 128
    recurrent_hidden_layers = 1
    data_chunk_length = 8
    modeldir='D:\\HCH\\LAG\\scripts\\results\\SingleControl\\1\\heading\\ppo\\v1\\run5'
    # Set the environment variable for CUDA (this is the same as CUDA_VISIBLE_DEVICES=0 in the shell)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Prepare the argument list to simulate the command line args
    args = [
        "--env-name", env,
        "--algorithm-name", algo,
        "--scenario-name", scenario,
        "--experiment-name", exp,
        "--seed", str(seed),
        "--n-training-threads", str(n_training_threads),
        "--n-rollout-threads", str(n_rollout_threads),
        "--cuda" if cuda else "",  # Only add `--cuda` if `cuda` is True
        "--log-interval", str(log_interval),
        "--save-interval", str(save_interval),
        "--num-mini-batch", str(num_mini_batch),
        "--buffer-size", str(buffer_size),
        "--num-env-steps", str(num_env_steps),
        "--lr", str(lr),
        "--gamma", str(gamma),
        "--ppo-epoch", str(ppo_epoch),
        "--clip-params", str(clip_params),
        "--max-grad-norm", str(max_grad_norm),
        "--entropy-coef", str(entropy_coef),
        "--hidden-size", hidden_size,
        "--act-hidden-size", act_hidden_size,
        "--recurrent-hidden-size", str(recurrent_hidden_size),
        "--recurrent-hidden-layers", str(recurrent_hidden_layers),
        "--data-chunk-length", str(data_chunk_length),
        "--model-dir",str(modeldir)
    ]

    args = [arg for arg in args if arg]
    # Simulate passing these as command line arguments to the main function
    argv = ["render_jsbsim.py"] + args  # Simulate the args being passed from the command line
    main(argv[1:])
