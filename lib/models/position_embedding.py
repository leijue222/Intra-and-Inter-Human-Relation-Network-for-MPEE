import math
import torch
from torch import nn
from torchvision.models import resnet18

class PositionEmbeddingImage(nn.Module):
    def __init__(self, trans_size, d_model=96, mode='conv', vec_dim=None):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.trans_size = trans_size
        self.d_model = d_model
        self.mode = mode
        self.patch_len=self.trans_size[0]*self.trans_size[1]
        if self.mode == 'res':
            self.conv_pre = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False)
            res = resnet18(pretrained=False)
            self.res = nn.Sequential(*list(res.children())[:5])
            self.conv_end = nn.Conv2d(64, d_model, kernel_size=3, stride=1, padding=1, bias=False)
        elif self.mode == 'cat_vec':
            # 256,192 -> 16,12
            # bs,16,12 -> bs,96
            # bs,1,96 -> bs,16*12,96
            self.fc = nn.Linear(self.patch_len, vec_dim)
        elif self.mode == 'conv':
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
            self.conv2 = nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(d_model, momentum=0.1)
            # self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
            # self.bn3 = nn.BatchNorm2d(512, momentum=0.1)
            # self.reduce = nn.Conv2d(512, 256, 1, bias=False)
            self.relu = nn.ReLU(inplace=True)

    def make_sine_position_embedding(self, b, n, temperature=10000, scale=2 * math.pi):
        d_model = self.d_model
        h, w = self.trans_size
        w = n * w
        area = torch.ones(b, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        # dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats) # torch 1.7版本适用
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(2, 0, 1)

        # return pos  # [h*w, 1, d_model]
        return nn.Parameter(pos,requires_grad=False)  # [h*w, 1, d_model]


    def forward(self, x):
        bs, n, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        
        if self.mode == 'cat_vec':
            down_rate = x.shape[-1] // self.trans_size[-1]
            down_rate = int(math.log(down_rate, 2))
            
            # 256,192 -> 16,12
            for _ in range(down_rate):
                x = self.maxpool(x)
            _, c, h, w = x.shape
            # bs,16,12 -> bs,96
            x = self.fc(x.view(bs*n,c*h*w))
            # bs,96 -> bs,n,96
            x = x.view(bs,n,-1)
            # bs,n,96 -> bs,n*16*12,96
            tmp_list=[]
            for i in range(n):
                tmp_list.append(x[:,i:i+1].repeat(1,self.patch_len,1))
            x = torch.cat(tmp_list,dim=1)
            # bs,n*16*12,96  -> bs,n,96,16,12
            x = x.view(bs,n,self.trans_size[0],self.trans_size[1],-1)
            x = x.permute(0,1,4,2,3)
        elif self.mode == 'sine':
            device = x.device
            x = self.make_sine_position_embedding(bs, n)
            x = x.to(device)
        else:
            if self.mode == 'res':
                x = self.conv_pre(x)
                x = self.res(x)
                x = self.conv_end(x)
            else: 
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
        
            down_rate = x.shape[-1] // self.trans_size[-1]
            down_rate = int(math.log(down_rate, 2))
            for _ in range(down_rate):
                x = self.maxpool(x)
            # x = self.conv3(x)
            # x = self.bn3(x)
            # x = self.relu(x)
            # x = self.reduce(x)
            # x = self.maxpool(x)
            # x = x.reshape(bs, n, x.shape[-3], x.shape[-2], x.shape[-1])
            x = x.view(bs, n, x.shape[-3], x.shape[-2], x.shape[-1])
        return x


def build_position_encoding(d_model, trans_size, mode, vec_dim):
    position_embedding = PositionEmbeddingImage(d_model, trans_size, mode, vec_dim)
    return position_embedding