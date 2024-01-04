import os
import time
import argparse
import torch


def get_options(args=None):

    parser = argparse.ArgumentParser(description="CMAES_PPO")

    # CMAES(basic_env) settings
    parser.add_argument('--backbone', default='cmaes', choices=['cmaes'], help='backbone algorithm')
    parser.add_argument('--m', type=int, default=20, help='number of subgroups')
    parser.add_argument('--sub_popsize', type=int, default=50, help='population size of each subgroup')
    parser.add_argument('--max_fes', type=int, default=3e6, help='maximum number of function evaluations')

    #problem settings
    parser.add_argument('--divide_method', default="train_sep", choices=["random_divide", "train_sep", "train_sep_parsep", "train_sep_parsep_2"], help='method to divide the problem set')
    
    #rollout settings
    parser.add_argument('--fes_one_cmaes', type=int, default=10000, help='number of function evaluations for each cmaes')
    parser.add_argument('--one_problem_batch_size', type=int, default=3, help='number of instances for each problem')
    parser.add_argument('--per_eval_time',  type=int, default=1, help='number of evaluations for each instance')

    #PPO settings
    parser.add_argument('--state', default=[0.0 for _ in range(15)], help='initial state of actor') 
    #此处单纯为了创建一个actor实例，数值没有实际意义，此处不用np的目的是run中需要转换为json文件
    parser.add_argument('--test', type=int, default=0, help='swith to test mode')
    parser.add_argument('--device', default='cpu', help='device to use for training / testing')
    parser.add_argument('--batch_size', type=int, default=2, help='number of instances per batch during training')

    # DE settings
    parser.add_argument('--problem', default='Schwefel', choices=['Sphere', 'Schwefel', 'Ackley', 'Bent_cigar'])
    parser.add_argument('--reward_definition', type=float, default=0., choices=[0., 0.1, 0.2, 3.1, 3.])
    parser.add_argument('--mutate_strategy', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--upper_bound', type=int, default=100)
    parser.add_argument('--lower_bound', type=int, default=-100)
    parser.add_argument('--feature_dim', type=int, default=9, help='dim of population features')
    parser.add_argument('--F_range', type=float, default=[0.1, 1.], action='append', help='range of F')
    parser.add_argument('--Cr_range', type=float, default=[0., 1.], action='append', help='range of Cr')
    parser.add_argument('--sigma_range', type=float, default=[0.0001, 0.02], action='append', help='range of sigma')

    # parameters in framework
    parser.add_argument('--no_cuda', action='store_true', help='disable GPUs')
    parser.add_argument('--no_tb', action='store_true', help='disable Tensorboard logging')
    parser.add_argument('--show_figs', action='store_true', help='enable figure logging')
    parser.add_argument('--no_saving', action='store_true', help='disable saving checkpoints')
    parser.add_argument('--use_assert', action='store_true', help='enable assertion')
    parser.add_argument('--no_DDP', action='store_true', help='disable distributed parallel')
    parser.add_argument('--seed', type=int, default=1234, help='random seed to use')

    # Net(Attention Aggregation) parameters
    parser.add_argument('--v_range', type=float, default=6., help='to control the entropy')
    parser.add_argument('--encoder_head_num', type=int, default=4, help='head number of encoder')
    parser.add_argument('--decoder_head_num', type=int, default=4, help='head number of decoder')
    parser.add_argument('--critic_head_num', type=int, default=6, help='head number of critic encoder')
    parser.add_argument('--embedding_dim', type=int, default=16, help='dimension of input embeddings')
    parser.add_argument('--hidden_dim', type=int, default=16, help='dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=1, help='number of stacked layers in the encoder')
    parser.add_argument('--normalization', default='layer', help="normalization type, 'layer' (default) or 'batch'")

    # Training parameters
    parser.add_argument('--RL_agent', default='ppo', choices=['ppo'], help='RL Training algorithm')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor for future rewards')
    parser.add_argument('--decision_interval', type=int, default=1, help='make action decision per decision_interval generations')
    parser.add_argument('--K_epochs', type=int, default=3, help='mini PPO epoch')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO clip ratio')
    parser.add_argument('--T_train', type=int, default=2000, help='number of iterations for training')
    parser.add_argument('--n_step', type=int, default=10, help='n_step for return estimation')
    
    parser.add_argument('--epoch_end', type=int, default=200, help='maximum training epoch')
    parser.add_argument('--epoch_size', type=int, default=1024, help='number of instances per epoch during training')
    parser.add_argument('--lr_model', type=float, default=1e-5, help="learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-5, help="learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1., help='learning rate decay per epoch')
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='maximum L2 norm for gradient clipping')

    # Inference and validation parameters
    parser.add_argument('--Max_Eval', type=int, default=200000, help='number of obj evaluation for inference')
    parser.add_argument('--eval_only', action='store_true', default=False, help='switch to inference mode')
    parser.add_argument('--val_size', type=int, default=1024, help='number of instances for validation/inference')
    parser.add_argument('--greedy_rollout', action='store_true')
    parser.add_argument('--dataset_path', default=None,)
    parser.add_argument('--inference_interval', type=int, default=3)
    parser.add_argument('--load_path_for_test',
                        default="D:/Users/Desktop/SRP/全参数控制/DE_PPO/outputs/Schwefel_10/run_name_20221017T220824/epoch-89.pt",
                        help='path to load model parameters for test')

    # "D:\Users\Desktop\SRP\全参数控制\DE_PPO\outputs\Schwefel_10\run_name_20221017T220824\epoch-89.pt"

    # resume and load models
    parser.add_argument('--load_path',
                        default=None,
                        help='path to load model parameters and optimizer state from')

    parser.add_argument('--resume',
                        default=None,
                        help='resume from previous checkpoint file')
    parser.add_argument('--epoch_start', type=int, default=0, help='start at epoch # (relevant for learning rate decay)')

    # logs/output settings
    parser.add_argument('--no_progress_bar', action='store_true', help='disable progress bar')
    parser.add_argument('--log_dir', default='logs', help='directory to write TensorBoard information to')
    parser.add_argument('--log_step', type=int, default=1, help='log info every log_step gradient steps')
    parser.add_argument('--output_dir', default='outputs', help='directory to write output models to')
    parser.add_argument('--run_name', default='run_name', help='name to identify the run')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='save checkpoint every n epochs (default 1), 0 to save no checkpoints')

    opts = parser.parse_args(args)

    # figure out whether to use distributed training if needed
    opts.world_size = 1
    opts.distributed = False
    # opts.world_size = torch.cuda.device_count()
    # opts.distributed = (torch.cuda.device_count() > 1) and (not opts.no_DDP)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4869'
    # processing settings
    opts.use_cuda =  opts.no_cuda #torch.cuda.is_available() and not
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S")) \
        if not opts.resume else opts.resume.split('/')[-2]
    opts.save_dir = os.path.join(
        opts.output_dir,
        opts.run_name
    ) if not opts.no_saving else None

    return opts