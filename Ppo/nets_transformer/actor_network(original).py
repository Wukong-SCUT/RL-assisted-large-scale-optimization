from torch import nn
import torch
from nets_transformer.graph_layers import MultiHeadEncoder, MLP_for_actor, EmbeddingNet
from torch.distributions import Normal, Gamma, Categorical


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
                 dim_p,
                 embedding_dim,
                 hidden_dim,
                 n_heads_actor,
                 n_heads_decoder,
                 n_layers,
                 normalization,
                 v_range,
                 node_dim,
                 F_range,
                 Cr_range,
                 sigma_range,
                 beta_F=7.,
                 beta_Cr=7.,
                 ):
        super(Actor, self).__init__()

        self.dim_p = dim_p
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads_actor = n_heads_actor
        self.n_heads_decoder = n_heads_decoder        
        self.n_layers = n_layers
        self.normalization = normalization
        self.range = v_range
        self.node_dim = node_dim
        self.F_range = F_range
        self.Cr_range = Cr_range
        self.sigma_range = sigma_range
        self.beta_F = beta_F
        self.beta_Cr = beta_Cr
        config = [{'in': self.embedding_dim, 'out': self.hidden_dim, 'drop_out': 0.001, 'activation': 'ReLU'},
                  {'in': self.hidden_dim, 'out': self.hidden_dim, 'drop_out': 0.001, 'activation': 'ReLU'},
                  {'in': self.hidden_dim, 'out': self.embedding_dim, 'drop_out': 0.001, 'activation': 'ReLU'},
                 ]
        # self.log_sigma_min = -20.
        # self.log_sigma_max = -1.5

        self.embedder = EmbeddingNet(self.node_dim,
                                     self.embedding_dim)
        
        # self.encoder = mySequential(*(
        #         MultiHeadEncoder(self.n_heads_actor,
        #                          self.embedding_dim,
        #                          self.hidden_dim,
        #                          self.normalization)
        #         for _ in range(self.n_layers)))  # stack L layers
        self.encoder = MLP(config)

        # F和Cr的mu、sigma分别共享网络
        # self.mu_net = MLP_for_actor(self.embedding_dim, 16, 2)    # 64 -> 16 -> 2
        # self.sigma_net = MLP_for_actor(self.embedding_dim, 16, 2)

        # F、Cr不共享网络 #参考 
        self.mu_net_F = MLP_for_actor(self.embedding_dim, 16, 1)
        # self.sigma_net_F = MLP_for_actor(self.embedding_dim, 16, 1)
        self.mu_net_Cr = MLP_for_actor(self.embedding_dim, 16, 1)
        # self.sigma_net_Cr = MLP_for_actor(self.embedding_dim, 16, 1)
        self.mu_net_op = MLP_for_actor(self.embedding_dim, 16, 4)
        self.co_net_op = MLP_for_actor(self.embedding_dim, 16, 2)

        print(self.get_parameter_number())

    def get_parameter_number(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Actor: Total': total_num, 'Trainable': trainable_num}

    def forward(self, x_in, fixed_action=None, require_entropy=False, to_critic=False, only_critic=False, sample=True):
        """
        x_in: shape=[bs, ps, feature_dim]
        """
        # pass through embedder
        h_em = self.embedder(x_in)  # [bs, ps, dim_em]
        # pass through encoder
        logits = self.encoder(h_em)  # [bs, ps, dim_em]

        # share embeddings to critic net
        if only_critic:
            return logits
        
        # pass through decoder

        # F和Cr的mu、sigma分别共享网络
        # mu = self.mu_net(logits).permute(2, 0, 1)  # [2, bs, ps]
        # mu_F = torch.sigmoid(mu[0]) * (self.F_range[1] - self.F_range[0]) + self.F_range[0]
        # mu_Cr = torch.sigmoid(mu[1]) * (self.Cr_range[1] - self.Cr_range[0]) + self.Cr_range[0]
        #
        # sigma = self.sigma_net(logits).permute(2, 0, 1)  # [2, bs, ps]
        # sigma_F = torch.sigmoid(sigma[0]) * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
        # sigma_Cr = torch.sigmoid(sigma[1]) * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]

        # F、Cr不共享网络
        # mu_F = torch.sigmoid(self.mu_net_F(logits)) * (self.F_range[1] - self.F_range[0]) + self.F_range[0]  # 映射至[F_lb, F_ub], shape=[bs, ps]
        # sigma_F = torch.sigmoid(self.sigma_net_F(logits)) * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]  # 映射至[sigma_lb, sigma_ub]
        # mu_Cr = torch.sigmoid(self.mu_net_Cr(logits)) * (self.Cr_range[1] - self.Cr_range[0]) + self.Cr_range[0]  # 映射至[Cr_lb, Cr_ub], shape=[bs, ps]
        # sigma_Cr = torch.sigmoid(self.sigma_net_Cr(logits)) * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]  # 映射至[sigma_lb, sigma_ub]

        # 分别控制Gamma分布的alpha，不共享网络
        alpha_F = torch.sigmoid(self.mu_net_F(logits)) * (self.F_range[1] - self.F_range[0]) + self.F_range[0]  # 映射至[F_lb, F_ub], shape=[bs, ps]
        alpha_Cr = torch.sigmoid(self.mu_net_Cr(logits)) * (self.Cr_range[1] - self.Cr_range[0]) + self.Cr_range[0]  # 映射至[Cr_lb, Cr_ub], shape=[bs, ps]
        mu_op = torch.softmax(self.mu_net_op(logits), -1)
        co_op = torch.softmax(self.co_net_op(logits), -1)

        # policy_F = Normal(mu_F, sigma_F)
        # policy_Cr = Normal(mu_Cr, sigma_Cr)
        policy_F = Gamma(alpha_F, self.beta_F)
        policy_Cr = Gamma(alpha_Cr, self.beta_Cr)
        policy_mu = Categorical(mu_op)
        policy_co = Categorical(co_op)

        # sample actions (number of controlled params, bs, ps)
        if fixed_action is not None:
            action = torch.tensor(fixed_action)
        else:
            if sample:
                action = torch.stack([policy_F.sample(), policy_Cr.sample(), policy_mu.sample(), policy_co.sample()])
            else:
                action = torch.stack([alpha_F.detach(), alpha_Cr.detach(), policy_mu.sample().detach(), policy_co.sample()])  # greedy

        # torch.clamp_(action[0], self.F_range[0], self.F_range[1])  # F
        # torch.clamp_(action[1], self.Cr_range[0], self.Cr_range[1])  # Cr

        # softmax (bs, ps * number of controlled params)
        ll = torch.cat((policy_F.log_prob(action[0]), policy_Cr.log_prob(action[1]), policy_mu.log_prob(action[2]), policy_co.log_prob(action[3])), -1) #log
        # ll[ll < -1e5] = -1e5

        if require_entropy:
            entropy = torch.cat((policy_F.entropy(), policy_Cr.entropy(), policy_mu.entropy(), policy_co.entropy()), -1)  # for logging only
            out = (action,
                   ll.sum(1),  # bs
                   logits if to_critic else None,
                   entropy)
        else:
            out = (action,
                   ll.sum(1),  # bs
                   logits if to_critic else None,)

        return out
