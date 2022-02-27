import torch
import torch.nn as nn
import math, copy

class Embeddings(nn.Module):
    '''
    对图像进行编码，把图片当做一个句子，把图片分割成块，每一块表示一个单词
    '''
    def __init__(self, patch_size, img_size, in_channels, hidden_size, drop_rate):
        super(Embeddings,self).__init__()
        img_size=img_size#224
        patch_size=patch_size#16
        ##将图片分割成多少块（224/16）*（224/16）=196
        n_patches=(img_size[0]//patch_size[0])*(img_size[1]//patch_size[1])
        #对图片进行卷积获取图片的块，并且将每一块映射成config.hidden_size维（768）
        self.patch_embeddings=nn.Conv2d(in_channels=in_channels,
                                     out_channels=hidden_size,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        
        #设置可学习的位置编码信息，（1,196+1,786）
        self.position_embeddings=nn.Parameter(torch.zeros(1,
                                                          n_patches+1,
                                                          hidden_size))
        #设置可学习的分类信息的维度
        self.classifer_token=nn.Parameter(torch.zeros(1,1,hidden_size))
        self.dropout=nn.Dropout(drop_rate)
        self.hidden_size = hidden_size

    def forward(self,x):
        bs=x.shape[0]
        cls_tokens=self.classifer_token.expand(bs,1,self.hidden_size)
        x=self.patch_embeddings(x)#（bs,768,14,14）
        x=x.flatten(2)#(bs,768,196)
        x=x.transpose(-1,-2)#(bs,196,768)
        x=torch.cat((cls_tokens,x),dim=1)#将分类信息与图片块进行拼接（bs,197,768）
        embeddings=x+self.position_embeddings#将图片块信息和对其位置信息进行相加(bs,197,768)
        embeddings=self.dropout(embeddings)
        return  embeddings

#2.构建self-Attention模块
class Attention(nn.Module):
    def __init__(self, head_num, hidden_size, drop_rate, vis=True):
        super(Attention,self).__init__()
        self.vis=vis
        self.num_attention_heads=head_num#12
        self.attention_head_size = int(hidden_size / self.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64=768

        self.query = nn.Linear(hidden_size, self.all_head_size)#wm,768->768，Wq矩阵为（768,768）
        self.key = nn.Linear(hidden_size, self.all_head_size)#wm,768->768,Wk矩阵为（768,768）
        self.value = nn.Linear(hidden_size, self.all_head_size)#wm,768->768,Wv矩阵为（768,768）
        self.out = nn.Linear(hidden_size, hidden_size)  # wm,768->768
        self.attn_dropout = nn.Dropout(drop_rate)
        self.proj_dropout = nn.Dropout(drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
        self.num_attention_heads, self.attention_head_size)  # wm,(bs,197)+(12,64)=(bs,197,12,64)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # wm,(bs,12,197,64)

    def forward(self, hidden_states):
        # hidden_states为：(bs,197,768)
        mixed_query_layer = self.query(hidden_states)#wm,768->768
        mixed_key_layer = self.key(hidden_states)#wm,768->768
        mixed_value_layer = self.value(hidden_states)#wm,768->768

        query_layer = self.transpose_for_scores(mixed_query_layer)#wm，(bs,12,197,64)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))#将q向量和k向量进行相乘（bs,12,197,197)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)#将结果除以向量维数的开方
        attention_probs = self.softmax(attention_scores)#将得到的分数进行softmax,得到概率
        weights = attention_probs if self.vis else None#wm,实际上就是权重
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)#将概率与内容向量相乘
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)#wm,(bs,197)+(768,)=(bs,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights#wm,(bs,197,768),(bs,197,197)

#3.构建前向传播神经网络
#两个全连接神经网络，中间加了激活函数
class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, drop_rate):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)#wm,786->3072
        self.fc2 = nn.Linear(mlp_dim, hidden_size)#wm,3072->786
        self.act_fn = torch.nn.functional.gelu#wm,激活函数
        self.dropout = nn.Dropout(drop_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)#wm,786->3072
        x = self.act_fn(x)#激活函数
        x = self.dropout(x)#wm,丢弃
        x = self.fc2(x)#wm3072->786
        x = self.dropout(x)
        return x

#4.构建编码器的可重复利用的Block()模块：每一个block包含了self-attention模块和MLP模块
class Block(nn.Module):
    def __init__(self, head_num, hidden_size, mlp_dim, drop_rate, vis=True):
        super(Block, self).__init__()
        self.hidden_size = hidden_size#wm,768
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)#wm，层归一化
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        self.ffn = Mlp(hidden_size=hidden_size, mlp_dim=mlp_dim, drop_rate=drop_rate)
        self.attn = Attention(head_num=head_num, hidden_size=hidden_size, drop_rate=drop_rate, vis=vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h#残差结构

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h#残差结构
        return x, weights

#5.构建Encoder模块，该模块实际上就是堆叠N个Block模块
class Encoder(nn.Module):
    def __init__(self, hidden_size, layer_num, head_num, mlp_dim, drop_rate, vis=True):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(layer_num):
            layer = Block(head_num, hidden_size, mlp_dim, drop_rate, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

#6构建transformers完整结构，首先图片被embedding模块编码成序列数据，然后送入Encoder中进行编码
class Transformer(nn.Module):
    def __init__(self, patch_size, img_size, in_channels, hidden_size, layer_num, head_num, mlp_dim, drop_rate, vis=True):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(patch_size, img_size, in_channels, hidden_size, drop_rate)#wm,对一幅图片进行切块编码，得到的是（bs,n_patch+1（196）,每一块的维度（768））
        self.encoder = Encoder(hidden_size, layer_num, head_num, mlp_dim, drop_rate, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)#wm,输出的是（bs,196,768)
        encoded, attn_weights = self.encoder(embedding_output)#wm,输入的是（bs,196,768)
        return encoded, attn_weights#输出的是（bs,197,768）

#7构建VisionTransformer，用于图像分类
class VisionTransformer(nn.Module):
    def __init__(self, patch_size, img_size, in_channels, hidden_size, layer_num, head_num, mlp_dim, num_classes, drop_rate, zero_head=False, vis=True):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = 'token'

        self.transformer = Transformer(patch_size, img_size, in_channels, hidden_size, layer_num, head_num, mlp_dim, drop_rate, vis)
        self.head = nn.Linear(hidden_size, num_classes)#wm,768-->10

    def forward(self, x):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])
        return logits
        
