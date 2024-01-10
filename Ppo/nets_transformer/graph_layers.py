import torch
import torch.nn.functional as F
from torch import nn
import math

# implements skip-connection module / short-cut module
class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


# implements MLP module for critic
class MLP_for_critic(torch.nn.Module):
    def __init__(self,
                 input_dim=128,
                 feed_forward_dim=64,
                 embedding_dim=64,
                 output_dim=1,
                 p=0.001
    ):
        # todo: change the hyperparam config
        super(MLP_for_critic, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, feed_forward_dim)       # 19->19
        self.fc2 = torch.nn.Linear(feed_forward_dim, embedding_dim)   # 19->64
        self.fc3 = torch.nn.Linear(embedding_dim, output_dim)         # 64->1
        self.dropout = torch.nn.Dropout(p=p)
        self.ReLU = nn.ReLU(inplace=True)

        # self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, in_):
        result = self.fc1(in_)
        result = self.fc2(result)
        result = self.fc3(result).squeeze(-1)
        return result


# implements MLP module for actor
class MLP_for_actor(torch.nn.Module):
    def __init__(self,
                 input_dim=64,
                 embedding_dim=16,
                 output_dim=1,
                 p=0.001
    ):
        super(MLP_for_actor, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, embedding_dim)
        self.fc2 = torch.nn.Linear(embedding_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=p)
        self.ReLU = nn.ReLU(inplace=True)

        # self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, in_):
        result = self.fc1(in_)
        result = self.fc2(result).squeeze(-1)
        return result


# implements Normalization module
class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalization = normalization

        if not self.normalization == 'layer':
            self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.normalization == 'layer':
            return (input - input.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(input.var((1, 2)).view(-1, 1, 1) + 1e-05)

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


# implements the encoder for Critic net
class MultiHeadAttentionLayerforCritic(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionLayerforCritic, self).__init__(
            SkipConnection(  # Attn & Residual
                    MultiHeadAttention(
                        n_heads,
                        input_dim=embed_dim,
                        embed_dim=embed_dim
                    )                
            ),
            Normalization(embed_dim, normalization),  # Norm
            SkipConnection(  # FFN & Residual
                    nn.Sequential(
                        nn.Linear(embed_dim, feed_forward_hidden),
                        nn.ReLU(inplace = True),
                        nn.Linear(feed_forward_hidden, embed_dim)
                    ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)   # Norm
        ) 


# implements the decoder for Critic net
class ValueDecoder(nn.Module):
    def __init__(
            self,
            embed_dim,
            input_dim,
    ):
        super(ValueDecoder, self).__init__()
        self.hidden_dim = embed_dim
        self.embedding_dim = embed_dim
        
        # for Pooling
        # self.project_graph = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        # self.project_node = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
        # for output
        self.MLP = MLP_for_critic(input_dim, embed_dim)  # 三个全连接层，output_dim=1

    def forward(self, h_em):  # h_em: [bs, ps, feature_dim]
        # mean Pooling
        mean_pooling = h_em #.mean(1)  # [bs, feature_dim]
        # graph_feature = self.project_graph(mean_pooling)[:, None, :]  # [bs, 1, dim_em]
        # node_feature = self.project_node(h_em)  # [bs, ps, dim_em]
        # fusion = node_feature + graph_feature.expand_as(node_feature)  # [bs, ps, dim_em]
        
        # pass through value_head, get estimated values
        value = self.MLP(mean_pooling)  # bs
      
        return value


# implements the original Multi-head Self-Attention module
class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        # todo randn?rand
        self.W_query = nn.Parameter(torch.randn(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.randn(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.randn(n_heads, input_dim, val_dim))

        # self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.randn(n_heads, val_dim, embed_dim))
            # self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        # self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q):
        
        h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, batch_size, n_query, key_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        
        # Calculate keys and values (n_heads, batch_size, graph_size, key_size or val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)   
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        attn = F.softmax(compatibility, dim=-1)   # (n_heads, batch_size, n_query, graph_size)
       
        heads = torch.matmul(attn, V)  # (n_heads, batch_size, n_query, val_size)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),  # (batch_size * n_query, n_heads * val_size)
            self.W_out.view(-1, self.embed_dim)  # (n_heads * val_size, embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out

  
# implements the multi-head compatibility layer
class MultiHeadCompat(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadCompat, self).__init__()
    
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(1 * key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        # self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h = None, mask=None):
        if h is None:
            h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)  
        K = torch.matmul(hflat, self.W_key).view(shp)   

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = torch.matmul(Q, K.transpose(2, 3))
        
        return self.norm_factor * compatibility


# implements the encoder
class MultiHeadEncoder(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer'
    ):
        super(MultiHeadEncoder, self).__init__()
        self.MHA_sublayer = MultiHeadAttentionsubLayer(
                        n_heads,
                        embed_dim,
                        feed_forward_hidden,
                        normalization=normalization,
                )
        
        self.FFandNorm_sublayer = FFandNormsubLayer(
                        n_heads,
                        embed_dim,
                        feed_forward_hidden,
                        normalization=normalization,
                )
        
    def forward(self, input):
        out = self.MHA_sublayer(input)
        return self.FFandNorm_sublayer(out)

# implements the encoder (DAC-Att sublayer)   
class MultiHeadAttentionsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionsubLayer, self).__init__()
        
        self.MHA = MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
        
        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input):
        # Attention
        out = self.MHA(input)
        
        # Residual connection and Normalization
        return self.Norm(out + input)


# implements the encoder (FFN sublayer)   
class FFandNormsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(FFandNormsubLayer, self).__init__()
        
        self.FF = nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
        
        self.Norm = Normalization(embed_dim, normalization)
    
    def forward(self, input):
    
        # FF
        out = self.FF(input)
        
        # Residual connection and Normalization
        return self.Norm(out + input)


class EmbeddingNet(nn.Module):
    
    def __init__(self,
                 node_dim,
                 embedding_dim):

        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.embedder = nn.Linear(node_dim, embedding_dim, bias=False)

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        h_em = self.embedder(x)
        return h_em
    