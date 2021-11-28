# -*- File Info -*-
# @File      : model.py
# @Date&Time : 2021-11-06, 21:20:33
# @Project   : aiBenchmark
# @Author    : yoc
# @Email     : iyoc@foxmail.com
# @Software  : PyCharm - Razer Blade


import torch
import torchvision


weight_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

if torch.cuda.is_available():

    resnet_18p = torchvision.models.resnet18(pretrained=True).cuda()
    resnet_18u = torchvision.models.resnet18(pretrained=False).cuda()

    resnet_34p = torchvision.models.resnet34(pretrained=True).cuda()
    resnet_34u = torchvision.models.resnet34(pretrained=False).cuda()

    resnet_50p = torchvision.models.resnet50(pretrained=True).cuda()
    resnet_50u = torchvision.models.resnet50(pretrained=False).cuda()

    resnet_101p = torchvision.models.resnet101(pretrained=True).cuda()
    resnet_101u = torchvision.models.resnet101(pretrained=False).cuda()

    resnet_152p = torchvision.models.resnet152(pretrained=True).cuda()
    resnet_152u = torchvision.models.resnet152(pretrained=False).cuda()

else:

    resnet_18p = torchvision.models.resnet18(pretrained=True)
    resnet_18u = torchvision.models.resnet18(pretrained=False)

    resnet_34p = torchvision.models.resnet34(pretrained=True)
    resnet_34u = torchvision.models.resnet34(pretrained=False)

    resnet_50p = torchvision.models.resnet50(pretrained=True)
    resnet_50u = torchvision.models.resnet50(pretrained=False)

    resnet_101p = torchvision.models.resnet101(pretrained=True)
    resnet_101u = torchvision.models.resnet101(pretrained=False)

    resnet_152p = torchvision.models.resnet152(pretrained=True)
    resnet_152u = torchvision.models.resnet152(pretrained=False)
