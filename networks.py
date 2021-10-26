import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.xavier_uniform_(m.weight)

class network(nn.Module):
    def __init__(self, actions_count):
        super(network, self).__init__()
        self.actions_count = actions_count

        self.conv1s = nn.Conv2d(4, 32, 3, stride=2, padding=1)  # B, CH, H, W
        self.attention_layer = MultiHeadAttention(32)
        self.conv2s = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3s = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv4s = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv5s = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv6s = nn.Conv2d(64, 32, 3, stride=1, padding=1)

        self.fca1 = nn.Linear(5 * 5 * 32, 512)
        self.fcc1 = nn.Linear(5 * 5 * 32, 512)

        self.fca2 = nn.Linear(512, actions_count)
        self.fcc2 = nn.Linear(512, 1)

        self.apply(weights_init_xavier)

    def forward(self, x):
        x = F.relu(self.conv1s(x))

        x = self.attention_layer(x, x, x)
        
        x = F.relu(self.conv2s(x))
        x = F.relu(self.conv3s(x))
        x = F.relu(self.conv4s(x))
        x = F.relu(self.conv5s(x))
        x = F.relu(self.conv6s(x))

        x = x.flatten(start_dim=1)

        x_a = F.relu(self.fca1(x))
        x_c = F.relu(self.fcc1(x))

        outActor = self.fca2(x_a)
        outCritic = self.fcc2(x_c)

        action = F.softmax(outActor, dim=-1).detach()

        action = action.multinomial(num_samples=1)

        return outActor, outCritic, action

class MultiHeadAttention(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.w_qs = nn.Conv2d(size, size, 1)
        self.w_ks = nn.Conv2d(size, size, 1)
        self.w_vs = nn.Conv2d(size, size, 1)


    def forward(self, q, k, v):
        residual = q

        q = self.w_qs(q).permute(0, 2, 3, 1)
        k = self.w_ks(k).permute(0, 2, 3, 1)
        v = self.w_vs(v).permute(0, 2, 3, 1)

        attn = torch.matmul(q, k.transpose(2, 3))

        attention = torch.matmul(attn, v).permute(0, 3, 1, 2)

        out = attention + residual
        return out
