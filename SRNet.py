import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
# 30 SRM filtes
from srm_filter_kernel import all_normalized_hpf_list
# Global covariance pooling
from MPNCOV import *  # MPNCOV


# Truncation operation
class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()
        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)
        return output


# Pre-processing Module
class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()
        # Load 30 SRM Filters
        all_hpf_list_5x5 = []
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            all_hpf_list_5x5.append(hpf_item)
        hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)
        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight
        # Truncation, threshold = 3
        self.tlu = TLU(3.0)

    def forward(self, input):

        output = self.hpf(input)
        output = self.tlu(output)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.group1 = HPF()

        self.group2 = nn.Sequential(
            nn.Conv2d(30, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

        )

        self.fc1 = nn.Linear(int(256 * (256 + 1) / 2), 2)

    def forward(self, input):
        output = input

        output = self.group1(output)
        output = self.group2(output)
        output = self.group3(output)
        output = self.group4(output)
        output = self.group5(output)

        # Global covariance pooling
        output = CovpoolLayer(output)
        output = SqrtmLayer(output, 5)
        output = TriuvecLayer(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)

        return output


class SRNet(nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()
        # Layer 1
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=64,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Layer 2
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=16,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        # Layer 3
        self.layer31 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(16)
        self.layer32 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(16)
        # Layer 4
        self.layer41 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(16)
        self.layer42 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(16)
        # Layer 5
        self.layer51 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(16)
        self.layer52 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn52 = nn.BatchNorm2d(16)
        # Layer 6
        self.layer61 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn61 = nn.BatchNorm2d(16)
        self.layer62 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn62 = nn.BatchNorm2d(16)
        # Layer 7
        self.layer71 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(16)
        self.layer72 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn72 = nn.BatchNorm2d(16)
        # Layer 8
        self.layer81 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=1, stride=2, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(16)
        self.layer82 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn82 = nn.BatchNorm2d(16)
        self.layer83 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn83 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 9
        self.layer91 = nn.Conv2d(in_channels=16, out_channels=64,
                                 kernel_size=1, stride=2, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(64)
        self.layer92 = nn.Conv2d(in_channels=16, out_channels=64,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(64)
        self.layer93 = nn.Conv2d(in_channels=64, out_channels=64,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn93 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 10
        self.layer101 = nn.Conv2d(in_channels=64, out_channels=128,
                                  kernel_size=1, stride=2, padding=0, bias=False)
        self.bn101 = nn.BatchNorm2d(128)
        self.layer102 = nn.Conv2d(in_channels=64, out_channels=128,
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.bn102 = nn.BatchNorm2d(128)
        self.layer103 = nn.Conv2d(in_channels=128, out_channels=128,
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.bn103 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 11
        self.layer111 = nn.Conv2d(in_channels=128, out_channels=256,
                                  kernel_size=1, stride=2, padding=0, bias=False)
        self.bn111 = nn.BatchNorm2d(256)
        self.layer112 = nn.Conv2d(in_channels=128, out_channels=256,
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.bn112 = nn.BatchNorm2d(256)
        self.layer113 = nn.Conv2d(in_channels=256, out_channels=256,
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        self.pool4 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 12
        self.layer121 = nn.Conv2d(in_channels=256, out_channels=512,
                                  kernel_size=3, stride=2, padding=0, bias=False)
        self.bn121 = nn.BatchNorm2d(512)
        self.layer122 = nn.Conv2d(in_channels=512, out_channels=512,
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.bn122 = nn.BatchNorm2d(512)
        # avgp = torch.mean() in forward before fc
        # Fully Connected layer
        self.fc = nn.Linear(512 * 1 * 1, 2)
        # self.fc1 = nn.Linear(512 * 1 * 1, 2)
        # self.fc1 = nn.Linear(int(512 * (512 + 1) / 2), 2)

    def forward(self, inputs):
        # Layer 1
        conv1 = self.layer1(inputs)
        actv1 = F.relu(self.bn1(conv1))
        # Layer 2
        conv2 = self.layer2(actv1)
        actv2 = F.relu(self.bn2(conv2))
        # Layer 3
        conv31 = self.layer31(actv2)
        actv3 = F.relu(self.bn31(conv31))
        conv32 = self.layer32(actv3)
        bn3 = self.bn32(conv32)
        res3 = torch.add(actv3, bn3)
        # Layer 4
        conv41 = self.layer41(res3)
        actv4 = F.relu(self.bn41(conv41))
        conv42 = self.layer42(actv4)
        bn4 = self.bn42(conv42)
        res4 = torch.add(res3, bn4)
        # Layer 5
        conv51 = self.layer51(res4)
        actv5 = F.relu(self.bn51(conv51))
        conv52 = self.layer52(actv5)
        bn5 = self.bn52(conv52)
        res5 = torch.add(res4, bn5)
        # Layer 6
        conv61 = self.layer61(res5)
        actv6 = F.relu(self.bn61(conv61))
        conv62 = self.layer62(actv6)
        bn6 = self.bn62(conv62)
        res6 = torch.add(res5, bn6)
        # Layer 7
        conv71 = self.layer71(res6)
        actv7 = F.relu(self.bn71(conv71))
        conv72 = self.layer72(actv7)
        bn7 = self.bn72(conv72)
        res7 = torch.add(res6, bn7)
        # Layer 8
        convs81 = self.layer81(res7)
        bn81 = self.bn81(convs81)
        conv82 = self.layer82(res7)
        actv8 = F.relu(self.bn82(conv82))
        conv83 = self.layer83(actv8)
        bn83 = self.bn83(conv83)
        pool1 = self.pool1(bn83)
        res8 = torch.add(bn81, pool1)
        # Layer 9
        convs91 = self.layer91(res8)
        bn91 = self.bn91(convs91)
        conv92 = self.layer92(res8)
        actv9 = F.relu(self.bn92(conv92))
        conv93 = self.layer93(actv9)
        bn93 = self.bn93(conv93)
        pool2 = self.pool2(bn93)
        res9 = torch.add(bn91, pool2)
        # Layer 10
        convs101 = self.layer101(res9)
        bn101 = self.bn101(convs101)
        convs102 = self.layer102(res9)
        actv101 = F.relu(self.bn102(convs102))
        convs103 = self.layer103(actv101)
        bn103 = self.bn103(convs103)
        pool3 = self.pool3(bn103)
        res10 = torch.add(bn101, pool3)
        # Layer 11
        convs111 = self.layer111(res10)
        bn111 = self.bn111(convs111)
        conv112 = self.layer112(res10)
        actv112 = F.relu(self.bn112(conv112))
        conv113 = self.layer113(actv112)
        bn113 = self.bn113(conv113)
        pool4 = self.pool4(bn113)
        res11 = torch.add(bn111, pool4)
        # Layer 12
        conv121 = self.layer121(res11)
        actv12 = F.relu(self.bn121(conv121))
        conv122 = self.layer122(actv12)
        bn122 = self.bn122(conv122)
        # print(bn.shape)
        avgp = torch.mean(bn122, dim=(2, 3), keepdim=True)
        # avgp = torch.mean(bn, dim=2, keepdim=True)
        # avgp = torch.mean(bn, dim=3, keepdim=True)
        # avgp = torch.nn.functional.adaptive_avg_pool2d(bn,(1,1))
        # print(avgp.shape)
        flatten = avgp.view(avgp.size(0), -1)
        # print(flatten.shape)
        # output = CovpoolLayer(bn)
        # output = SqrtmLayer(output, 5)
        # output = TriuvecLayer(output)
        # flatten = output.view(output.size(0), -1)
        fc = self.fc(flatten)
        # fc = self.fc1(flatten)
        # out = F.log_softmax(fc, dim=1)
        return fc
