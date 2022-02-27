import torch
import torch.nn as nn
class MlpBlock(nn.Module):
    def __init__(self,in_dim,hidden_dim,drop_rate=0):
        super(MlpBlock,self).__init__()
        self.mlp=nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(drop_rate))
    def forward(self,x):
        return self.mlp(x)

class MixerLayer(nn.Module):
    '''
    ns:序列数（patch个数）；nc：通道数（嵌入维度）；ds：token mixing的隐层神经元数；dc：channel mixing的隐层神经元数；
    '''
    def __init__(self,ns,nc,ds=256,dc=2048,drop_rate=0.):
        super(MixerLayer,self).__init__()
        self.norm1=nn.LayerNorm(nc)
        self.norm2=nn.LayerNorm(nc)
        self.tokenMix=MlpBlock(in_dim=ns,hidden_dim=ds,drop_rate=drop_rate)
        self.channelMix=MlpBlock(in_dim=nc,hidden_dim=dc,drop_rate=drop_rate)
    #(7352, 128*9, 1)  nc=1, ns=128*9
    def forward(self,x):
        x=self.norm1(x) #对每一行归一化 (7352, 128*9, 1)
        x2=self.tokenMix(x.transpose(1,2)).transpose(1,2) # (7352, 128*9, 1)-(7352, 1, 128*9)(MIX—FLC)-(7352, 128*9, 1)
        x=x+x2 # (7352, 128*9, 1)
        x2=self.norm2(x) #对结果每一行再次归一化(7352, 128*9, 1)
        x2=self.channelMix(x2) #(7352, 128*9, 1)（MIX-FLC)
        return x+x2 #叠加 (7352, 128*9, 1)

class Mixer(nn.Module):
    def __init__(self,num_classes,image_size,patch_size=(1, 1),num_layers=8,embed_dim=512,ds=256,dc=2048,drop_rate=0):
        '''
        :param image_size: 输入图像分辨率(n*n)
        :param num_classes: 分类类别数
        :param num_layers: mixer层数
        :param patch_size: patch的宽高
        :param embed_dim: 通道数C
        :param ds: token-mixing的隐层神经元数
        :param dc: channel-mixing的隐层神经元数
        '''
        super(Mixer,self).__init__()
        # assert image_size%patch_size==0
        self.embed = nn.Conv2d(1,embed_dim,kernel_size=patch_size,stride=patch_size) #(7352, 512, 128//ps, 9//ps)
        ns=(image_size[0]//patch_size[0])*(image_size[1]//patch_size[1]) # 序列数 (128//ps) * (9//ps)
        MixBlock=MixerLayer(ns=ns,nc=embed_dim,ds=ds,dc=dc,drop_rate=drop_rate)
        self.mixlayers=nn.Sequential(*[MixBlock for _ in range(num_layers)]) #8层长得一样的
        self.norm=nn.LayerNorm(embed_dim) #对每一行512进行归一化
        self.cls=nn.Linear(embed_dim,num_classes)  #分类（7352, 512)*(512, 6)

    def forward(self,x):
        x=self.embed(x).flatten(2).transpose(1,2) # n c2 hw->n hw c2 (7352, 1, 128, 9)--(7352, 512, ns)--(7352, ns, 512)
        x=self.mixlayers(x)  #(7352, ns, 512)
        x=self.norm(x) #对一行512个数进行归一化(7352, ns, 512)
        x=torch.mean(x,dim=1) # 逐通道求均值 N C (7352, 512)
        x=self.cls(x)
        return x