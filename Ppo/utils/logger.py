import torch
import math
from Ppo.utils.plots import plot_grad_flow, plot_improve_pg
import numpy as np
    
def log_to_screen(init_value, best_value, reward, search_history,
                  batch_size, dataset_size, T, records):
    # reward
    print('\n', '-'*60)
    print('Avg total reward:'.center(35), '{:<10f} +- {:<10f}'.format(
            reward.sum(1).mean(), torch.std(reward.sum(1)) / math.sqrt(batch_size)))
    print('Avg step reward:'.center(35), '{:<10f} +- {:<10f}'.format(
            reward.mean(), torch.std(reward) / math.sqrt(batch_size)))
            
    # cost
    print('-'*60)
    print('Avg init cost:'.center(35), '{:<10f} +- {:<10f}'.format(
            init_value.mean(), torch.std(init_value) / math.sqrt(batch_size)))
    # best cost
    print('-'*60)
    
    # for per in range(20,100,20):
    #     cost_ = search_history[:,round(T*per/100)]
    #     print(f'Avg best cost after {per}% steps:'.center(35), '{:<10f} +- {:<10f}'.format(
    #             cost_.mean(), 
    #             torch.std(cost_) / math.sqrt(batch_size)))
    print('Avg current best cost:'.center(35), '{:<10f} +- {:<10f}'.format(
                best_value.mean(), torch.std(best_value) / math.sqrt(dataset_size)))
    print('Avg best cost so far:'.center(35), '{:<10f} +- {:<10f}'.format(
                records['best cost'].mean(), torch.std(records['best cost']) / math.sqrt(dataset_size)))
    print('current ratio:'.center(35), '{:<10f}'.format(
                torch.sum(best_value < records['baseline cost']) / dataset_size))
    print('best ratio so far:'.center(35), '{:<10f}'.format(
                records['success ratio']))
    print('Avg current best descent:'.center(35), '{:<10f} +- {:<10f}'.format(
                1 - (best_value / init_value).mean(), torch.std(best_value / init_value)))
    print('Avg best descent so far:'.center(35), '{:<10f} +- {:<10f}'.format(
                1 - (records['best cost'] / init_value).mean(), torch.std(records['best cost'] / init_value) / math.sqrt(dataset_size)))
    print('Avg baseline cost:'.center(35), '{:<10f} +- {:<10f}'.format(
                records['baseline cost'].mean(), torch.std(records['baseline cost']) / math.sqrt(dataset_size)))
    print('-'*60, '\n')


def log_to_tb_val(tb_logger, time_used, init_value, best_value, reward, search_history, records,
                  batch_size, dataset_size, T, show_figs, epoch):
    if show_figs:
        tb_logger.log_images('validation/search_pg',[plot_improve_pg(search_history)], epoch)
        
    # tb_logger.log_value('validation/avg_time',  time_used.mean() / dataset_size, epoch)
    tb_logger.log_value('validation/avg_total_reward', reward.mean(), epoch)
    # tb_logger.log_value('validation/avg_step_reward', reward.mean(), epoch)

    tb_logger.log_value('validation/avg_init_cost', init_value.mean(), epoch)
    tb_logger.log_value('validation/avg_best_cost', best_value.mean(), epoch)
    tb_logger.log_value('validation/sccess_ratio', torch.sum(best_value < records['baseline cost']) / dataset_size, epoch)
    tb_logger.log_value('validation/descent', 1 - (best_value / init_value).mean(), epoch)


def log_to_tb_train(tb_logger, agent, Reward, ratios, bl_val_detached, total_cost, grad_norms, reward, entropy, approx_kl_divergence,
               reinforce_loss, baseline_loss, log_likelihood, baseline, show_figs, mini_step, R , state, state_next):
    
    tb_logger.log_value('learnrate_pg/actor_lr', agent.optimizer.param_groups[0]['lr'], mini_step)
    tb_logger.log_value('learnrate_pg/critic_lr', agent.optimizer.param_groups[1]['lr'], mini_step)
    avg_cost = total_cost #.mean().item()
    tb_logger.log_value('train/avg_cost', avg_cost, mini_step)
    tb_logger.log_value('train/Target_Return', Reward.mean().item(), mini_step)
    tb_logger.log_value('train/ratios', ratios.mean().item(), mini_step)
    avg_reward = torch.cat(reward).mean()#torch.stack(reward, 0).sum(0).mean().item()
    max_reward = torch.cat(reward).max()#torch.stack(reward, 0).max(0)[0].mean().item() #reward检查一下
    tb_logger.log_value('train/avg_reward', avg_reward, mini_step)
    # tb_logger.log_value('train/init_cost', np.array(initial_cost).mean(), mini_step)
    tb_logger.log_value('train/max_reward', max_reward, mini_step)
    tb_logger.log_value('train/baseline', np.array(baseline).mean(), mini_step)

    grad_norms, grad_norms_clipped = grad_norms
    tb_logger.log_value('loss/actor_loss', reinforce_loss.item(), mini_step)
    tb_logger.log_value('loss/nll', -log_likelihood.mean().item(), mini_step)
    tb_logger.log_value('train/entropy', entropy.mean().item(), mini_step)
    tb_logger.log_value('train/approx_kl_divergence', approx_kl_divergence.item(), mini_step)
    tb_logger.log_value('train/bl_val',bl_val_detached.mean().cpu(),mini_step)

    tb_logger.log_value('train/R', R.mean().cpu(), mini_step)

    tb_logger.log_value('train/mean_state', state.mean().cpu(), mini_step)
    tb_logger.log_value('train/max_state', state.max().cpu(), mini_step)
    tb_logger.log_value('train/min_state', state.min().cpu(), mini_step)

    tb_logger.log_value('train/mean_state_next', state_next.mean(), mini_step)
    tb_logger.log_value('train/max_state_next', state_next.max(), mini_step)
    tb_logger.log_value('train/min_state_next', state_next.min(), mini_step)
    #记录R max state 
    
    tb_logger.log_value('grad/actor', grad_norms[0], mini_step)
    tb_logger.log_value('grad_clipped/actor', grad_norms_clipped[0], mini_step)
    tb_logger.log_value('loss/critic_loss', baseline_loss.item(), mini_step)
            
    tb_logger.log_value('loss/total_loss', (reinforce_loss+baseline_loss).item(), mini_step)
    
    tb_logger.log_value('grad/critic', grad_norms[1], mini_step)
    tb_logger.log_value('grad_clipped/critic', grad_norms_clipped[1], mini_step)
    
    if show_figs and mini_step % 1000 == 0:
        tb_logger.log_images('grad/actor', [plot_grad_flow(agent.actor)], mini_step)
        tb_logger.log_images('grad/critic', [plot_grad_flow(agent.critic)], mini_step)


def log_to_tb_train_per_step(tb_logger, action, cost, step):
    tb_logger.log_value('sample/F', action[0].mean(), step)
    tb_logger.log_value('sample/Cr', action[1].mean(), step)
    tb_logger.log_value('sample/cost', cost.mean(), step)


def log_to_tb_val_per_step(tb_logger, action, cost, total_steps, epoch_steps, epoch=0):
    tb_logger.log_value('validation/step/F', action[0].mean(), total_steps)
    tb_logger.log_value('validation/step/Cr', action[1].mean(), total_steps)
    for i in range(4):
        tb_logger.log_value(f'validation/step/action[{i}]', torch.sum(action[2] == i) / torch.sum(action[2] > -1000), total_steps)
    tb_logger.log_value('validation/step/Cr', action[1].mean(), total_steps)
    # tb_logger.log_value('validation/step/cost_bsf', cost.mean(), total_steps)
    tb_logger.log_value(f'validation_per_epoch_F/epoch {epoch}', action[0].mean(), epoch_steps)
    tb_logger.log_value(f'validation_per_epoch_Cr/epoch {epoch}', action[1].mean(), epoch_steps)
    for i in range(4):
        tb_logger.log_value(f'validation_per_epoch_Operator/epoch {epoch}/action[{i}]', torch.sum(action[2] == i) / torch.sum(action[2] > -1000), total_steps)
    tb_logger.log_value(f'validation_per_epoch_Cost_bsf/epoch {epoch}', cost.mean(), epoch_steps)
