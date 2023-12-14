from mmcls.models import RedNet
import torch.nn as nn

class Red50(nn.Module):
    def __init__(self, num_classes=1000):
        super(Red50, self).__init__()
        self.model = RedNet(50)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.classifier(self.avgpool(self.model(x)).flatten(1))


class Red26(nn.Module):
    def __init__(self, num_classes=1000):
        super(Red26, self).__init__()
        self.model = RedNet(26)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.classifier(self.avgpool(self.model(x)).flatten(1))


class Red101(nn.Module):
    def __init__(self, num_classes=1000):
        super(Red101, self).__init__()
        self.model = RedNet(101)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.classifier(self.avgpool(self.model(x)).flatten(1))