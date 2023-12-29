from torch import nn
import torch
from Ppo.nets_transformer.graph_layers import MLP_for_actor
import torch.nn.functional as F

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MLP(nn.Module): 
    def __init__(self, config):
        super(MLP, self).__init__()
        self.net = nn.Sequential()
        #self.net_config = config.net_config
        self.net_config = config
        for layer_id, layer_config in enumerate(self.net_config):
            linear = nn.Linear(layer_config['in'], layer_config['out'])
            self.net.add_module(f'layer{layer_id}-linear', linear)
            drop_out = nn.Dropout(layer_config['drop_out'])
            self.net.add_module(f'layer{layer_id}-drop_out', drop_out)
            if layer_config['activation'] != 'None':
                activation = eval('nn.'+layer_config['activation'])()
                self.net.add_module(f'layer{layer_id}-activation', activation)

    def forward(self,x):
        return self.net(x)


class Actor(nn.Module):

    def __init__(self,
                 input_dim,
                 state
                 ):
        super(Actor, self).__init__()

        self.input_dim = input_dim #和state+1的维度一致
        self.state = state

        # 网络创建
        self.CC_method_net = MLP_for_actor(self.input_dim, 16, 1)

        # input  : [CC_method_num, state_1, state_2,..., state_n]
        # output : pobability in the state

        # self.MiVD_net = MLP_for_actor(self.input_dim, 16, 1)
        # self.MaVD_net = MLP_for_actor(self.input_dim, 16, 1)
        # self.RD_net = MLP_for_actor(self.input_dim, 16, 1)

        print(self.get_parameter_number()) #打印模型参数

    def get_parameter_number(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Actor: Total': total_num, 'Trainable': trainable_num}

    def forward(self, fixed_action = None):
        """
        x_in: 放入state
        """
        x_in = self.state
        input_tensor = torch.tensor([[0,x_in],[1,x_in],[2,x_in]])

        score = self.CC_method_net(input_tensor) 
        action_prob = F.softmax(score, dim=-1) 
        action_dist = torch.distributions.Categorical(action_prob)

        if fixed_action is not None:
            action = torch.tensor(fixed_action)
        else:
            action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action)

        ll = log_prob

        entropy = action_dist.entropy()  # for logging only
        out = (action,ll,entropy)

        return out
    
    


# 创建 Actor 模型
# input_dim = 2  # 两个状态特征 + 1
# state = 1.0

# actor_model = Actor(input_dim, state)


# # 使用 Actor 模型生成动作
# action, log_prob,entrop = actor_model.forward()

# # 输出结果
# print("Generated Action:", action.item())
# print("Log Probability of Action:", log_prob.item())