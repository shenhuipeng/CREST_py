import torch
import torchvision
net1 = torchvision.models.vgg13(pretrained=True, )
print(net1.state_dict())