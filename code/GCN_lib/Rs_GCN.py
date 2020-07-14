import torch
from torch import nn
from torch.nn import functional as F
from disout import Disout,LinearScheduler



class Rs_GCN(nn.Module):

    def __init__(self, in_channels, inter_channels, num_head=16, bn_layer=True, kernel_size=1, padding=0):
        super(Rs_GCN, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.num_head = num_head

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd = nn.Conv1d
        max_pool = nn.MaxPool1d
        #bn = BatchNorm1d
        #bn = LayerNorm1d
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=kernel_size, stride=1, padding=padding)

        if bn_layer:
            #self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
            #            kernel_size=kernel_size, stride=1, padding=padding)
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=kernel_size, stride=1, padding=padding),
                bn(self.in_channels)
            )
            #self.disout1=LinearScheduler(Disout(dist_prob=0.09,block_size=6,alpha=5),
            #                            start_value=0.,stop_value=0.09,nr_steps=5e3)
            #self.bn = bn(self.in_channels)
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
            #nn.init.constant(self.bn.weight, 0)
            #nn.init.constant(self.bn.bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=kernel_size, stride=1, padding=padding)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None


        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=kernel_size, stride=1, padding=padding)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=kernel_size, stride=1, padding=padding)

        #self.bn = bn(self.in_channels)



    def forward(self, v, mask):
        '''
        :param v: (B, D, N)
        :return:
        '''
        batch_size = v.size(0)

        x = v
        #x = self.bn(v)
        g_v = self.g(x).view(batch_size, self.inter_channels, -1)
        #g_v = self.g(v).view(batch_size, self.inter_channels, -1)
        g_v = g_v.permute(0, 2, 1)
        g_v = g_v * mask.unsqueeze(2).repeat(1,1,g_v.size(2))

        theta_v = self.theta(x).view(batch_size, self.inter_channels, -1)
        #theta_v = self.theta(v).view(batch_size, self.inter_channels, -1)
        theta_v = theta_v.permute(0, 2, 1)
        phi_v = self.phi(x).view(batch_size, self.inter_channels, -1)
        #phi_v = self.phi(v).view(batch_size, self.inter_channels, -1)
        size = theta_v.shape
        theta_v = theta_v.view(size[0],size[1],self.num_head, int(size[2] / self.num_head))
        theta_v = torch.transpose(theta_v,1,2)
        phi_v = phi_v.permute(0, 2, 1)
        phi_v = phi_v.view(size[0],size[1],self.num_head, int(size[2] / self.num_head))
        phi_v = torch.transpose(phi_v,1,2)
        phi_v = torch.transpose(phi_v,2,3)
        
        R = torch.matmul(theta_v, phi_v)
        #N = R.size(-1)
        #R_div_C = R / N
        R_div_C = R / torch.sum(mask,1,keepdim=True).unsqueeze(1).unsqueeze(3).repeat(1,self.num_head,1,R.size(2))
        #R_div_C = R / torch.sqrt(torch.sum(mask,1,keepdim=True).unsqueeze(1).unsqueeze(3).repeat(1,self.num_head,1,R.size(2)))
        R_div_C = R_div_C * mask.unsqueeze(1).unsqueeze(3).repeat(1,self.num_head,1,R.size(2))
        R_div_C = R_div_C.permute(0,1,3,2)
        R_div_C = R_div_C * mask.unsqueeze(1).unsqueeze(3).repeat(1,self.num_head,1,R.size(2))
        R_div_C = R_div_C.permute(0,1,3,2)

        #R_div_C = softmax(R_div_C,3,mask)
        #R_div_C = R_div_C * mask.unsqueeze(1).unsqueeze(3).repeat(1,self.num_head,1,R.size(2))
        

        g_v = g_v.view(size[0],size[1],self.num_head, int(size[2] / self.num_head))
        g_v = torch.transpose(g_v,1,2)
        y = torch.matmul(R_div_C, g_v)
        
        y = torch.transpose(y,1,2).contiguous()
        y = y.view(size[0],size[1],size[2])
 
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *v.size()[2:])
        W_y = self.W(y)
        #W_y = self.disout1(y)
        #W_y = self.bn(W_y)
        W_y = W_y * mask.unsqueeze(1).repeat(1,W_y.size(1),1)
        v_star = W_y + x

        return v_star

def softmax(x, dim, mask):
     size = x.shape
     m = torch.max(x, dim, keepdim=True)[0]
     y = torch.exp(x - m)
     ma = mask.unsqueeze(1).unsqueeze(3).repeat(1,size[1],1,size[3])
     ma = 1 - ma
     y = y + ma
     #y = torch.exp(x)
     s = torch.sum(y, dim, keepdim=True)
     y = y * mask.unsqueeze(1).unsqueeze(3).repeat(1,size[1],1,size[3])
     y = torch.div(y, s)
     return y

def Mean(x, mask):
    N, C, H = x.size()
    mask =  torch.sum(mask, dim=1, keepdim=True).unsqueeze(2)
    mask = torch.sum(mask)
    x = torch.sum(x,(0,2), keepdim=True)
    mean = x / mask
    return mean

def Var(x, mean, mask):
    x = x - mean
    x = x * mask.unsqueeze(1)
    x = torch.pow(x, 2)
    var = Mean(x, mask)
    return var

def batch_norm(x,  mask, gamma, beta, running_mean, running_var,training, eps=1e-5, momentum=0.9):
         # x: input features with shape [N,C,H,W]
         # gamma, beta: scale and offset, with shape [1,C,1,1]
         N, C, H = x.size()
         if training:
             #mean = torch.mean(x,(0,2),keepdim=True)
             #var = torch.var(x,(0,2),keepdim=True)
             mean = Mean(x,mask)
             var = Var(x, mean, mask)
             x = x - mean
             #x = (x - mean) / torch.sqrt(var + eps)
             mean = momentum*running_mean + (1-momentum)*mean
             var = momentum*running_var + (1-momentum)*var
             running_mean.copy_(mean.detach())
             running_var.copy_(var.detach())
         else:
             mean = running_mean
             var = running_var
             x = (x - mean) / torch.sqrt(var + eps)
         if not gamma is None:
             return x * gamma + beta
         else:
             return x

class BatchNorm1d(nn.Module):

        __constants__ = ['num_channels', 'eps', 'affine', 'weight',
                         'bias']

        def __init__(self, num_channels, eps=1e-5, affine=True):
            super(BatchNorm1d, self).__init__()
            self.num_channels = num_channels
            self.eps = eps
            self.affine = affine
            if self.affine:
                self.weight = nn.Parameter(torch.Tensor(1,num_channels,1))
                self.bias = nn.Parameter(torch.Tensor(1,num_channels,1))
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
            self.register_buffer('running_mean', torch.zeros(1,num_channels,1))
            self.register_buffer('running_var', torch.ones(1,num_channels,1))
            self.reset_parameters()

        def reset_parameters(self):
            self.running_mean.zero_()
            self.running_var.fill_(1)
            if self.affine:
                #nn.init.ones_(self.weight)
                nn.init.constant(self.weight, 0)
                nn.init.constant(self.bias, 0)
        def forward(self, input, mask):
            return batch_norm(
                input, mask, self.weight, self.bias, self.running_mean, self.running_var, self.training, self.eps)

def Mean_ln(x, mask):
    N, C, H = x.size()
    mask =  torch.sum(mask, dim=1, keepdim=True).unsqueeze(2)
    x = torch.sum(x,(1,2), keepdim=True)
    mean = x / (mask * C)
    return mean

def Var_ln(x, mean, mask):
    x = x - mean
    x = x * mask.unsqueeze(1)
    x = torch.pow(x, 2)
    var = Mean_ln(x, mask)
    return var


def layer_norm(x,  mask, gamma, beta, training, eps=1e-5, momentum=0.9):
         # x: input features with shape [N,C,H,W]
         # gamma, beta: scale and offset, with shape [1,C,1,1]
         N, C, H = x.size()
         mean = Mean_ln(x,mask)
         var = Var_ln(x, mean, mask)
         x = (x - mean) / torch.sqrt(var + eps)
         if not gamma is None:
             return x * gamma + beta
         else:
             return x

class LayerNorm1d(nn.Module):

        __constants__ = ['num_channels', 'eps', 'affine', 'weight',
                         'bias']
        def __init__(self, num_channels, eps=1e-5, affine=True):
            super(LayerNorm1d, self).__init__()
            self.num_channels = num_channels
            self.eps = eps
            self.affine = affine
            if self.affine:
                self.weight = nn.Parameter(torch.Tensor(1,num_channels,1))
                self.bias = nn.Parameter(torch.Tensor(1,num_channels,1))
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
            self.reset_parameters()

        def reset_parameters(self):
            if self.affine:
                nn.init.zeros_(self.weight)
                nn.init.zeros_(self.bias)
        def forward(self, input, mask):
            return layer_norm(
                input, mask, self.weight, self.bias, self.training, self.eps)

        def extra_repr(self):
            return '{num_groups}, {num_channels}, eps={eps}, ' \
                'affine={affine}'.format(**self.__dict__)
