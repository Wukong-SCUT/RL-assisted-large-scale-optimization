from torch import nn
from Ppo.nets_transformer.graph_layers import ValueDecoder


class Critic(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 ):

        super(Critic, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # self.n_heads = n_heads
        # self.n_layers = n_layers
        # self.normalization = normalization

        # todo: remove the ValueDecoder and use the MLP directly
        self.value_head = ValueDecoder(input_dim=self.embedding_dim,
                                       embed_dim=self.embedding_dim)

        print(self.get_parameter_number())

    def forward(self, h_features):

        # get input
        # h_features = h_features.detach()

        # pass through value_head, get estimated value
        baseline_value = self.value_head(h_features)

        return baseline_value.detach().squeeze(), baseline_value.squeeze()  #记录一下

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Critic: Total': total_num, 'Trainable': trainable_num}

