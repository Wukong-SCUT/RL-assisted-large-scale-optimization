from Ppo.utils.utils import set_random_seed
import numpy as np
import torch
from tqdm import tqdm
from Ppo.utils.logger import log_to_tb_val
import os
from env.venvs import DummyVectorEnv,SubprocVectorEnv
from env.basic_env import cmaes
from Ppo.nets_transformer.actor_network import Actor


# interface for rollout
def rollout(dataloader,opts,agent=None,tb_logger=None, epoch_id=0):

    rollout_name=f'func_{opts.problem}_{opts.backbone}'
    if agent:
        rollout_name='GLEET-'+rollout_name
        agent.eval() 
    T = opts.max_fes // opts.population_size+1

    # to store the whole rollout process
    cost_rollout=np.zeros((opts.batch_size,int(T-1)))
    
    time_eval=0
    collect_mean=[]
    collect_std=[]
    
    
    # set the same random seed before rollout for the sake of fairness
    set_random_seed(42)
    for question in dataloader:
        batch = range(10)
        # figure out the backbone algorithm
        # backbone={
        #     'PSO':GPSO_numpy,
        #     'DMSPSO':DMS_PSO_np,
        # }.get(opts.backbone,None)
        
        #action, _, _ = Actor(input_dim=opts.dim, state=opts.dim+1)  #注意这里只是一个示例，并非真正的actor

        backbone=cmaes #此处为基础环境
        assert backbone is not None,'Backbone algorithm is currently not supported'
        # see if there is agent to aid the backbone
        origin=True
        if agent:
            origin=False
        env_list=env_list = [lambda e=p: backbone(question) for p in batch]  #注意此处只是一个示例
        # Parallel environmen SubprocVectorEnv can only be used in Linux
        vector_env=SubprocVectorEnv #if opts.is_linux else DummyVectorEnv
        problem=vector_env(env_list)  #此处problem就是为基础环境

        # list to store the final optimization result
        collect_gbest=np.zeros((opts.batch_size,opts.per_eval_time))

        for i in range(opts.per_eval_time):
            # reset the backbone algorithm
            is_end=False
            state=problem.reset()  
            
            if agent:
                state=torch.FloatTensor(state).to(opts.device)

            time_eval+=1
            
            # visualize the rollout process
            for t in tqdm(range(int(T)), disable = opts.no_progress_bar,
                            desc = rollout_name, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
                
                action_test = []
                for i in range(10):
                    state_i = state[i]
                    state_i = state_i.astype(np.float32)
                    actor = Actor(state_i)
                    action,_,_ = actor.forward()
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
                for tt in range(opts.batch_size):
                    cost_rollout[tt,t]+=info[tt]['gbest_val']
                if is_end.all():
                    if t+1<T:
                        for tt in range(opts.batch_size):
                            cost_rollout[tt,t+1:]+=info[tt]['gbest_val']
                    # store the final cost in the end of optimization process
                    for tt in range(opts.batch_size):
                        collect_gbest[tt,i]=info[tt]['gbest_val']
                    break
        
        # collect the mean and std of final cost
        collect_std.append(np.mean(np.std(collect_gbest,axis=-1)).item())
        collect_mean.append(np.mean(collect_gbest).item())
        # close the 
        problem.close()


    cost_rollout/=time_eval
    cost_rollout=np.mean(cost_rollout,axis=0)
    

    # save rollout data to file
    saving_path=os.path.join(opts.log_dir,opts.RL_agent,"{}_{}".format(opts.problem,
                                                        opts.dim),"rollout_{}_{}".format(opts.run_name,epoch_id))
    # only save part of the optimization process
    save_list=[cost_rollout[int((opts.dim**(k/5-3) * opts.max_fes )// opts.population_size -1 )].item() for k in range(15)]
    save_dict={'mean':np.mean(collect_mean).item(),'std':np.mean(collect_std).item(),'process':save_list}
    np.save(saving_path,save_dict)

    # log to tensorboard if needed
    if tb_logger:
        log_to_tb_val(tb_logger,cost_rollout,epoch_id)
    
    # calculate and return the mean and std of final cost
    return np.mean(collect_gbest).item(),np.mean(collect_std).item()

#测试
dataloader = [1,2,3]
from options import get_options
opts = get_options()
rollout(dataloader,opts)