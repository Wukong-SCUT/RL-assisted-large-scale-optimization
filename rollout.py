from Ppo.utils.utils import set_random_seed
import numpy as np
import torch
from tqdm import tqdm
from Ppo.utils.logger import log_to_tb_val
import os
from env.venvs import DummyVectorEnv,SubprocVectorEnv
from env.basic_env import cmaes
from Ppo.nets_transformer.actor_network import Actor
import sys
import warnings

# interface for rollout
def rollout(dataloader,opts,agent=None,tb_logger=None, epoch_id=0):

    rollout_name=f'{opts.backbone}_{opts.divide_method}'
    print(f'Rollout: {rollout_name}')

    if agent:
        rollout_name='PPO-'+rollout_name
        agent.eval() 
    
    T =(opts.max_fes // opts.fes_one_cmaes) / 300 #3e6/10000=300 cc-cmaes 要迭代的次数

    # to store the whole rollout process
    batch_size = len(dataloader)*opts.one_problem_batch_size

    cost_rollout=np.zeros((int(T),batch_size))
    
    time_eval=0
    collect_mean=[]
    collect_std=[]
    
    # set the same random seed before rollout for the sake of fairness
    set_random_seed(42)

    # figure out the backbone algorithm
    # backbone={
    #     'PSO':GPSO_numpy,
    #     'DMSPSO':DMS_PSO_np,
    # }.get(opts.backbone,None)
    
    #action, _, _ = Actor(input_dim=opts.dim, state=opts.dim+1)  #注意这里只是一个示例，并非真正的actor

    backbone=cmaes #此处为基础环境
    assert backbone is not None,'Backbone algorithm is currently not supported'
    env_list = [lambda e=p,q=question: backbone(q) for p in range(opts.one_problem_batch_size) for question in dataloader]  #注意此处只是一个示例

    # Parallel environmen SubprocVectorEnv can only be used in Linux
    vector_env=SubprocVectorEnv #if opts.is_linux else DummyVectorEnv
    problem=vector_env(env_list)  #此处problem就是为基础环境

    # list to store the final optimization result
    collect_gbest=np.zeros(batch_size)

    # reset the backbone algorithm
    is_end=False
    state=problem.reset()  
    
    if agent:
        state=torch.FloatTensor(state).to(opts.device)

    time_eval+=1
    
    # visualize the rollout process
    for t in tqdm(range(int(T))):
        
        action_test = []
        for _ in range(batch_size):
            state_i = state[_]
            #state_i = state_i.astype(np.float32)
            actor = Actor()
            action,_,_ = actor.forward(state_i)
            action_test.append(action)

        # if agent:
        #     # if RL_agent is provided, the action is from the agent.actor
        #     action,_,_ = agent.actor(state)
        #     action=action.cpu().numpy()
        # else:
        #     # if RL_agent is not provided, the action is set to zeros because of the need of the parallel environment,
        #     # but inside the backbone algorithm, the parameter controlled by action will use the default choices
        #     action=np.zeros(opts.batch_size)
        
        # put action into environment(backbone algorithm to be specific)
        next_state,rewards,is_end,info = problem.step(action_test)

        state=next_state
        if agent:
            state=torch.FloatTensor(state).to(opts.device)
            
        # store the rollout cost history
        for tt in range(batch_size):
            cost_rollout[t,tt]+=info[tt]['gbest_val']
            if tt == batch_size-1:
                collect_gbest = cost_rollout[-1,:]
        # if is_end.all():
        #     if t+1<T:
        #         for tt in range(batch_size):
        #             cost_rollout[tt,t+1:]+=info[tt]['gbest_val']
        #     # store the final cost in the end of optimization process
        #     for tt in range(batch_size):
        #         collect_gbest[tt,i]=info[tt]['gbest_val']
            #break
    
    # collect the mean and std of final cost
    # collect_std.append(np.mean(np.std(collect_gbest,axis=-1)).item())
    # collect_mean.append(np.mean(collect_gbest).item())
    # close the 
    problem.close()

    cost_rollout/=time_eval
    cost_rollout=np.mean(cost_rollout,axis=0)
    

    # # save rollout data to file
    # saving_path=os.path.join(opts.log_dir,opts.RL_agent,"{}_{}".format(opts.problem,
    #                                                     opts.dim),"rollout_{}_{}".format(opts.run_name,epoch_id))
    # # only save part of the optimization process
    # save_list=[cost_rollout[int((opts.dim**(k/5-3) * opts.max_fes )// opts.population_size -1 )].item() for k in range(15)]
    # save_dict={'mean':np.mean(collect_mean).item(),'std':np.mean(collect_std).item(),'process':save_list}
    # np.save(saving_path,save_dict)

    # log to tensorboard if needed

    #注意注意，此处注释掉了
    # if tb_logger:
    #     log_to_tb_val(tb_logger,cost_rollout,epoch_id)
    
    collect_gbest
    # 将数组按照每3个一组分割
    groups = np.array_split(collect_gbest, len(collect_gbest) // opts.one_problem_batch_size)

    # 计算每组的平均值
    collect_gbest_mean = np.array([group.mean() for group in groups])

    # 计算每组的标准差
    collect_gbest_std = np.array([group.std() for group in groups])

    # calculate and return the mean and std of final cost
    return collect_gbest_mean , collect_gbest_std

#测试
# dataloader = [1,2,3]
# from options import get_options
# opts = get_options()
# input1,input2 =rollout(dataloader,opts)
# print(input1,input2)