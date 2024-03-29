import os
import warnings
import torch
import numpy as np
from tqdm import tqdm
# from tensorboardX import SummaryWriter
# import torch.multiprocessing as mp
# import torch.distributed as dist
from Ppo.utils.utils import set_random_seed
from Ppo.utils.utils import clip_grad_norms
from Ppo.nets_transformer.actor_network import Actor
from Ppo.nets_transformer.critic_network import Critic
from Ppo.utils.utils import torch_load_cpu, get_inner_model
from Ppo.utils.logger import log_to_tb_train, log_to_tb_val
from env.venvs import SubprocVectorEnv
# import copy
#from problems.gpso_np import GPSO_numpy
#from problems.dmspso import DMS_PSO_np
#from problems.de_np import DE_np
#from problems.madde import MadDE
from Ppo.utils.make_dataset import Make_dataset 
from rollout import rollout
from env.basic_env import cmaes
import pandas as pd
from openpyxl import load_workbook

# 指定输出文件夹路径
output_folder = 'RL_assist/output_files/output_lr_1e-5'

# 创建输出文件夹，如果不存在的话
os.makedirs(output_folder, exist_ok=True)

# memory for recording transition during training process
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

# control the learning rate decay
def lr_sd(epoch, opts):
    return opts.lr_decay ** epoch


class PPO:
    def __init__(self, opts):

        # figure out the options
        self.opts = opts
        # the parallel environment
        self.vector_env=SubprocVectorEnv
        # figure out the actor network
        self.actor = Actor()
        
        if not opts.test: #test的时候是不会执行的
            # figure out the critic network
            # 这里有点问题 这个维度的选择似乎是固定的
            self.critic = Critic(
                embedding_dim=  opts.feature_num, #和state的大小一样，为15
                hidden_dim = opts.feature_num ,
            ) 
            # figure out the optimizer
            self.optimizer = torch.optim.Adam(
                [{'params': self.actor.parameters(), 'lr': opts.lr_model}] +
                [{'params': self.critic.parameters(), 'lr': opts.lr_model}])
            # figure out the lr schedule
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay, last_epoch=-1,)

        if opts.use_cuda: #将除了basic_env之外的所有东西都放入cuda:0
            # move to cuda
            self.actor.to(opts.device) #此处将actor放入cuda:0
            if not opts.test:
                self.critic.to(opts.device) 


    # load model from load_path
    def load(self, load_path):

        assert load_path is not None
        load_data = torch_load_cpu(load_path)

        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})

        if not self.opts.test:
            # load data for critic
            model_critic = get_inner_model(self.critic)
            model_critic.load_state_dict({**model_critic.state_dict(), **load_data.get('critic', {})})
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))

    # save trained model
    def save(self, epoch):
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    # change working mode to evaling
    def eval(self):
        torch.set_grad_enabled(False)  ##
        self.actor.eval()
        if not self.opts.test: self.critic.eval()

    # change working mode to training
    def train(self):
        torch.set_grad_enabled(True)  ##
        self.actor.train()
        if not self.opts.test: self.critic.train()


    def start_training(self, tb_logger):
        train(0, self, tb_logger)

# inference for training
def train(rank, agent, tb_logger):  #这里agent就是ppo
    print("begin training")
    opts = agent.opts
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    # move optimizer's data onto chosen device
    for state in agent.optimizer.state.values(): #Adam
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(opts.device) 


    # generatate the train_dataset and test_dataset
    set_random_seed(42)
    make_dataset = Make_dataset(opts.divide_method)

    training_dataloader = make_dataset.problem_set("train")   
    if opts.epoch_size==1: 
        test_dataloader=training_dataloader
    else:
        set_random_seed(42)
        test_dataloader = make_dataset.problem_set("test")

    # training_dataloader=agent.training_dataloader
    # test_dataloader=agent.test_dataloader


    best_epoch=None
    best_avg_best_cost=None
    best_epoch_list=[]
    mean_per_list=[]
    sigma_per_list=[]
    outperform_ratio_list=[]
    pre_step=0

    # modify the learning ray after resume(if needed)
    if opts.resume:
        for e in range(opts.epoch_start):
            agent.lr_scheduler.step(e)

    stop_training=False

    # 此处为了节约时间先设置关闭rollout
    # rollout in the 0 epoch model to get the baseline data
    baseline_all = {'1':18004806.63, '2':2473.74789187304, '3':20.1812018587071, '4':60798610661.5867, '5':9136356.05, '6':1057675.98232308, '7':561349654.297463,
                '8':2989332805662390, '9':526718730.623028, '10':93949538.660442, '11':91704674695.1519, '12':1232.46609753493, '13':10019658256.2841,
                '14':350069430500.856, '15':13373771.8541858}
    
    # init_avg_best,baseline=rollout(test_dataloader,opts,agent,tb_logger,-1)
    # baseline = np.array([2.92242934e+12, 3.01869040e+16, 2.19217652e+13, 5.19558640e+13,
    #    3.21623644e+12, 6.26348550e+16, 1.83685347e+13, 5.05585378e+13,
    #    2.92320559e+12, 6.34554712e+16, 6.20060268e+14, 7.58340799e+13])
    # Start the actual training loop
    for epoch in range(opts.epoch_start , opts.epoch_end): #epoch_start=0,epoch_end=200
        # Training mode
        set_random_seed()
        agent.train() #此处只是将actor和critic的状态都设置为train

        # agent.lr_scheduler_critic.step(epoch)
        # agent.lr_scheduler_actor.step(epoch)
        agent.lr_scheduler.step(epoch)

        # logging
        if rank == 0:
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            print("Training with actor lr={:.3e} critic lr={:.3e} for run {}".format(agent.optimizer.param_groups[0]['lr'],
                                                                                     agent.optimizer.param_groups[1]['lr'], opts.run_name) , flush=True)

        # start training
        # step = epoch * (opts.epoch_size // opts.batch_size)
        # episode_step = opts.max_fes // opts.fes_one_cmaes #max_fes=3e6,fes_one_cmaes=10000 
        # episode_step=8500
            
        pbar = tqdm(total = 30*len(training_dataloader)  , #这里有点问题
                    disable = opts.no_progress_bar or rank!=0, desc = 'training',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') #这里是不是有点太大了？？ training:   0%|                    | 0/4608000.0 [00:00<?, ?it/s]  4608000？？
                    #(opts.K_epochs) * opts.epoch_size  * (episode_step)
        
        for question in training_dataloader:

            each_question_batch_num = opts.each_question_batch_num  #5

            baseline =[ baseline_all[str(question)] for _ in range(each_question_batch_num) ]

            backbone = cmaes
            # backbone={
            #     'PSO':GPSO_numpy,
            #     'DMSPSO':DMS_PSO_np,
            #     'DE':DE_np,
            #     'madde':MadDE
            # }.get(opts.backbone,None)
            assert backbone is not None,'Backbone algorithm is currently not supported'
            env_list = [lambda e = question: backbone(e) for _ in range(each_question_batch_num)]
            
            #env_list=[lambda e=p: backbone(question=question) for p in range(batch)] 
            envs=agent.vector_env(env_list) #创建并行环境
            # train procedule for a batch

            batch_step=train_batch(rank,
                                envs, #这里是vector_env
                                agent,  #这里是ppo
                                pre_step,   #这里是pre_step 开始传入的是0
                                tb_logger,  #这里是tb_logger
                                opts,
                                pbar,   #进程表
                                baseline
                                )
            
            pre_step += batch_step
            envs.close()
            # see if the learning step reach the max_learning_step, if so, stop training
            # if pre_step>=opts.max_learning_step:   #最大学习步骤 乱设了一个10000 因为一开始没有这个参数
            #     stop_training=True
            #     break
        
        pbar.close()

        # save new model after one epoch
        if rank == 0 and not opts.distributed:
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                                       epoch == opts.epoch_end - 1): agent.save(epoch)
                                       
        if (epoch-opts.epoch_start) % opts.update_best_model_epochs==0 or epoch == opts.epoch_end-1: #update_best_model_epochs 乱设了一个99999 因为一开始没有这个参数
            # validate the new model
            avg_best,_=rollout(test_dataloader,opts,agent,tb_logger,-1)
            baseline = [ baseline_all[str(question)] for _ in range(len(test_dataloader)) ]
            #avg_best,sigma,per_problem_performance=rollout(test_dataloader,opts,agent,tb_logger,epoch)
            outperform_ratio=np.sum(avg_best<baseline)/len(baseline)
            #这里要检查一下
            #tb_logger.add_scalar('performance/outperform_ratio',outperform_ratio,epoch)
            mean_per_list.append(avg_best)
            #sigma_per_list.append(sigma)
            if epoch==opts.epoch_start:
                best_avg_best_cost=avg_best
                best_epoch=epoch
            elif avg_best.all()>best_avg_best_cost.all():
                best_avg_best_cost=avg_best
                best_epoch=epoch
            best_epoch_list.append(best_epoch)
            outperform_ratio_list.append(outperform_ratio)


        # logging
        print('current_epoch:{}, best_epoch:{}'.format(epoch,best_epoch))
        print('best_epoch_list:{}'.format(best_epoch_list))
        print(f'outperform_ratio_list:{outperform_ratio_list}')
        #print(f'init_mean_performance:{init_avg_best}')
        #print(f'init_sigma:{init_sigma}')
        print(f'cur_mean_performance:{mean_per_list}')
        #print(f'cur_sigma_performance:{sigma_per_list}')
        print(f'cur_outperform_ratio:{outperform_ratio}')
        #print(f'best_rl_pso_mean:{mean_per_list[(best_epoch-opts.epoch_start)  // opts.update_best_model_epochs]}')
        #print(f'best_rl_pso_std:{sigma_per_list[(best_epoch-opts.epoch_start)  // opts.update_best_model_epochs]}')
        
        if stop_training:
            print('Have reached the maximum learning steps')
            break
    print(best_epoch_list)


def train_batch(
        rank, #这里的rank不知道是什么
        vector_env, #这里是vector_env
        agent, #这里是ppo
        pre_step, #这里是pre_step 开始传入的是0
        tb_logger,
        opts,
        pbar, #这是一个进度条
        baseline,
        ):

    
    # setup
    agent.train() #切换为训练模型
    memory = Memory()

    # initial instances and solutions
    state=vector_env.reset() 
    state=torch.FloatTensor(state).to(opts.device) #将state放入cuda:0
    state=torch.where(torch.isnan(state),torch.zeros_like(state),state)
    # 对于输入张量 state 中的每个元素，
    # 如果该元素是 NaN（Not a Number），则将其替换为 0；否则，保持不变

    # params for training
    gamma = opts.gamma
    n_step = opts.n_step #使用多少个时间步来计算回报，这里设置为10
    
    K_epochs = opts.K_epochs
    eps_clip = opts.eps_clip
    
    t = 0
    # initial_cost = obj
    done=False
    
    state_next = None

    # sample trajectory # 轨迹抽样
    while not done:
        t_s = t
        total_cost = 0
        entropy = []
        bl_val_detached = []
        bl_val = []

        # accumulate transition
        while t - t_s < n_step :  
            
            memory.states.append(state.clone())


            action,log_lh,entro_p = agent.actor(state)
            action_cpu = action.cpu()
            

            memory.actions.append(action) 
            memory.logprobs.append(log_lh)
            #action=action.cpu().numpy()

            entropy.append(entro_p) #entro_p.detach().cpu()


            baseline_val_detached, baseline_val = agent.critic(state) #直接把state放入即可
            bl_val_detached.append(baseline_val_detached)
            bl_val.append(baseline_val)


            # state transient
            next_state, rewards, is_end, info = vector_env.step(action_cpu)
            
            memory.rewards.append(torch.FloatTensor(rewards).to(opts.device))

            state_next = next_state

            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))

            # store info
            if info[0]['Fes'] >= 1e6:
                    
                    question = info[0]['question']
                    # info_without_env_id = np.array([{k: v for k, v in d.items() if k != 'env_id'} for d in info], dtype=object)
                    # 转换为 pandas 的 DataFrame
                    df = pd.DataFrame([info[0]])
                    # 使用 pd.concat 连接第二个元素（字典）的 DataFrame
                    for i in range(1, len(info)):
                        df = pd.concat([df, pd.DataFrame([info[i]])], ignore_index=True)
                    
                    # 根据 'question' 和 'Fes' 进行分组，并计算最大值、最小值、平均值
                    result_table = df.groupby(['question', 'Fes']).agg({'gbest_val': ['max', 'min', 'mean']}).reset_index()

                    # 指定文件夹路径和文件名，将结果保存为 Excel 文件
                    
                    output_file = f'{output_folder}_{question}_output.xlsx'
                    result_table.to_excel(output_file, index=True)

                    # # 设置输出文件路径
                    # output_file = f'{output_folder}output.xlsx'

                    # # 检查文件是否存在
                    # try:
                    #     # 尝试加载现有 Excel 文件
                    #     with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
                    #         # 如果文件存在，将 DataFrame 追加到现有的 sheet 上
                    #         result_table.to_excel(writer, sheet_name='Sheet1', index=False, header=False)
                    # except FileNotFoundError:
                    #     # 如果文件不存在，直接创建新的 Excel 文件
                    #     result_table.to_excel(output_file, index=True)

            # total_cost = total_cost + gbest_val

            # next
            t = t + 1
            state=torch.FloatTensor(next_state).to(opts.device)
            state=torch.where(torch.isnan(state),torch.zeros_like(state),state)
            if is_end.all():
                done=True
                break


        
        # store info
        t_time = t - t_s
        total_cost = total_cost / t_time

        # begin update
        # 如果是madde这里的action就不能直接stack
        old_actions = torch.stack(memory.actions)
        old_states = torch.stack(memory.states).detach() #.view(t_time, bs, ps, dim_f)
        # old_actions = all_actions.view(t_time, bs, ps, -1)
        # print('old_actions.shape:{}'.format(old_actions.shape))

        old_logprobs = torch.stack(memory.logprobs).detach().view(-1)

        # Optimize PPO policy for K mini-epochs:
        old_value = None
        for _k in range(K_epochs):
            if _k == 0:
                logprobs = memory.logprobs

            else:
                # Evaluating old actions and values :
                logprobs = []
                entropy = []
                bl_val_detached = []
                bl_val = []

                for tt in range(t_time):

                    # for i in range(each_question_batch_num):
                    # get new action_prob
                    _, log_p,  entro_p = agent.actor(old_states[tt],fixed_action = old_actions[tt])

                    logprobs.append(log_p)
                    entropy.append(entro_p.detach().cpu())

                    baseline_val_detached, baseline_val = agent.critic(state)

                    bl_val_detached.append(baseline_val_detached)
                    bl_val.append(baseline_val)
            
            logprobs = torch.stack(logprobs).view(-1)
            entropy = torch.stack(entropy).view(-1)
            bl_val_detached = torch.stack(bl_val_detached).view(-1)
            bl_val = torch.stack(bl_val).view(-1)


            # get traget value for critic
            Reward = []
            reward_reversed = memory.rewards[::-1]
            # get next value
            R , _ = agent.critic(state)  #这里only_critic=True，所以actor的输出是critic的输入 #记录一下R

            # R = torch.tensor(np.array([r.detach().cpu().numpy() for r in R])) #这里是元组，没法处理
            # R = agent.critic(x_in)[0]
            # critic_output=R.clone()
            for r in range(len(reward_reversed)):
                R = R * gamma + reward_reversed[r]
                Reward.append(R)
            # clip the target:
            Reward = torch.stack(Reward[::-1], 0)
            Reward = Reward.view(-1)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            ratios = ratios.to(opts.device)
            #ratios = torch.cat(( ratios , torch.tensor([1.0,1.0])),dim=0)
            # Finding Surrogate Loss:
            advantages = Reward - bl_val_detached

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
            reinforce_loss = -torch.min(surr1, surr2).mean()

            # define baseline loss
            if old_value is None:
                baseline_loss = ((bl_val - Reward) ** 2).mean()
                old_value = bl_val.detach()
            else:
                vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                baseline_loss = v_max.mean()

            # check K-L divergence (for logging only)
            approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
            approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
            # calculate loss
            loss = baseline_loss + reinforce_loss

            # update gradient step
            agent.optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm and get (clipped) gradient norms for logging
            current_step = int(pre_step + t//n_step * K_epochs  + _k)
            grad_norms = clip_grad_norms(agent.optimizer.param_groups, opts.max_grad_norm)

            # perform gradient descent
            agent.optimizer.step()
            
            
            #Logging to tensorboard
            if(not opts.no_tb) and rank == 0:
                if current_step % int(opts.log_step) == 0:
                    log_to_tb_train(tb_logger, agent, Reward, ratios, bl_val_detached, total_cost, grad_norms, memory.rewards, entropy, approx_kl_divergence,
                                    reinforce_loss, baseline_loss, logprobs, baseline, opts.show_figs, current_step, R , state, state_next)

            if rank == 0: pbar.update(1)
            # end update
        

        memory.clear_memory()

    # return learning steps
    return ( t // n_step + 1) * K_epochs #此处返还的是3

