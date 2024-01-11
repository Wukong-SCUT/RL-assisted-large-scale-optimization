import os
import json

import torch
import pprint
import numpy as np
from tqdm import tqdm
import warnings
from tensorboard_logger import Logger as TbLogger

import env
from env.basic_env import cmaes
from options import get_options


from Ppo.utils.make_dataset import Make_dataset
from Ppo.utils.logger import log_to_tb_val_per_step
from ppo import PPO


def load_agent(name):
    agent = {
        'ppo': PPO,
    }.get(name, None)
    assert agent is not None, "Currently unsupported agent: {}!".format(name)
    return agent


def run(opts): 

    # Pretty print the run args
    pprint.pprint(vars(opts)) #更有结构的方式（pprint.pprint)呈现opts的参数设置

    # Optionally configure tensorboard  #注册tb_logger
    tb_logger = None
    if not opts.no_tb and not opts.distributed: #如果不是分布式的，就用tensorboard
        tb_logger = TbLogger(os.path.join(opts.log_dir,
                                          "{}".format(opts.divide_method),
                                          'debug',
                                          opts.run_name))

    if not opts.no_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
        
    # Save arguments so exact configuration can always be found
    if not opts.no_saving:
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu") #注意此处，默认的是gpu

    # Figure out the RL algorithm
    agent = load_agent(opts.RL_agent)(opts) #此处的agent是ppo

    # Load data from load_path
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        agent.load(load_path)

    # Do validation only
    if opts.eval_only:
        # Load the validation datasets
        agent.start_inference(tb_logger) #这段没什么用，agent.start_inference(tb_logger)不存在
        
    else:
        if opts.resume:
            epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
            print("Resuming after {}".format(epoch_resume))
            agent.opts.epoch_start = epoch_resume + 1

        # Start the actual training loop
        agent.start_training(tb_logger)


def test(opts):
    # Set the random seed
    # torch.manual_seed(opts.seed)
    # np.random.seed(opts.seed)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")

    test_data = Make_dataset.test_problem_set(problems=opts.problem,
                                 dim=opts.dim,
                                 num_samples=opts.val_size,
                                 batch_size=opts.val_size,
                                 filename=opts.dataset_path,
                                 shifted=True,
                                 rotated=True,
                                 biased=True,
                                 training_seed=opts.seed)

    log_dir = os.path.join(opts.log_dir,
                           "{}_{}".format(opts.problem, opts.dim),
                           "test_{}".format(opts.run_name))

    print(f'\nLogs saved in {log_dir}\n')

    # run DE_PPO
    tb_logger = TbLogger(os.path.join(log_dir, "{}_DE_PPO".format(opts.run_name)))
    agent = load_agent(opts.RL_agent)(opts)
    # Load data from load_path_for_test
    assert opts.load_path_for_test is not None, "load path for test must be given"
    agent.load(opts.load_path_for_test)
    agent.start_inference(tb_logger, test_data=test_data)

    # run normal DE
    tb_logger = TbLogger(os.path.join(log_dir, "{}_DE_normal".format(opts.run_name)))
    for batch in test_data:
        # create envs
        DE_envs = [lambda e=p: DE(problem=p,
                                  dim=opts.dim,
                                  lower_bound=opts.lower_bound,
                                  upper_bound=opts.upper_bound,
                                  population_size=opts.population_size,
                                  max_generation=opts.T_train,
                                  rank=None,
                                  distributed=False,
                                  device=opts.device,
                                  reward_definition=opts.reward_definition,
                                  mutate_strategy=opts.mutate_strategy,
                                  feature_dim=opts.feature_dim) for i, p in enumerate(batch)]

        envs = env.SubprocVectorEnv(DE_envs)
        obs = envs.reset()

        for t in tqdm(range(opts.Max_Eval // opts.population_size), disable=opts.no_progress_bar,
                      desc='rollout for normal DE', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):

            action = torch.stack([torch.rand(opts.val_size, opts.population_size) * 1.5,  # F
                                  torch.rand(opts.val_size, opts.population_size)])  # Cr

            obs, _, _, _ = envs.step(torch.permute(action, [1, 0, 2]).numpy())

            log_to_tb_val_per_step(tb_logger,
                                   action,
                                   torch.stack([torch.from_numpy(obs[i]['c_bsf']) for i in range(len(batch))]),
                                   t,
                                   t)

    print(f'Test completed. Logs saved in {log_dir}\n')


if __name__ == "__main__":
    # This is for 3090

    torch.set_num_threads(2) # 设置线程数

    warnings.filterwarnings("ignore")
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    run(get_options()) # 从options.py中获取参数 进入run函数

    # test(get_options())
