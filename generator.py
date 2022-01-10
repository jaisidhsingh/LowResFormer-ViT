import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

class pixelNorm(nn.Module):
    def __init__(self) :
        super(pixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self,x):
        return x/torch.sqrt(torch.mean(x**2,dim=1,keepdim=True)+self.epsilon)

class mappingNetwork(nn.Module):
    def __init__(self, zDim,wDim):
        super().__init__()
        self.mapping = nn.Sequential(
            pixelNorm(),
            WSLinear(zDim,wDim),
            nn.ReLU(),
            WSLinear(wDim,wDim),
            nn.ReLU(),
            WSLinear(zDim,wDim),
            nn.ReLU(),
            WSLinear(zDim,wDim),
            nn.ReLU(),
            WSLinear(zDim,wDim),
            nn.ReLU(),
            WSLinear(zDim,wDim),
            nn.ReLU(),
            WSLinear(zDim,wDim),
            nn.ReLU(),
            WSLinear(zDim,wDim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mapping(x)


class noiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1,channels,1,1))

    def forward(self,x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x+self.weight*noise
    
class AdaIN(nn.Module):
    def __init__(self,channels,wDim):
        super().__init__()
        self.instanceNorm=nn.InstanceNorm2d(channels)
        self.styleScale=WSLinear(wDim,channels)
        self.styleBias=WSLinear(wDim,channels)

    def forward(self,x,w):
        x = self.instanceNorm(x)
        styleScale = self.styleScale(w).unsqueeze(2).unsqueeze(3)
        stleBias = self.styleBias(w).unsqueeze(2).unsqueeze(3)
        return styleScale*x + styleBias


class WSConv2d(nn.Module):
    def __init__(self , inChannels, outChannels,kernelSize=3,stride=1,padding=1,gain=2,):  #parameter after gain is unknown as of now

        super(WSConv2d,self).__init__()
        self.conv = nn.Conv2d(inChannels,outChannels,kernelSize,stride,padding)
        self.scale = (gain/(inChannels*(kernelSize**2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        #conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self,x):
        return self.conv(x*self.scale)+self.bias.view(1,self.bias.shape[0],1,1)

class WSLinear(nn.Module):
    def __init__(self, inFeatures,outFeatures,gain=2,):#again last parameter not known
        super(WSLinear,self).__init__()
        self.linear = nn.Linear(inFeatures,outFeatures)
        self.scale = (gain/inFeatures)**0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        #linear layer
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self,x):
        return self.linear(x*self.scale)+self.bias

class GenBlock(nn.Module):
    def __init__(self, inChannels,outChannels,wDim):
        super(GenBlock,self).__init__()
        self.conv1 = WSConv2d(inChannels,outChannels)
        self.conv2 = WSConv2d(outChannels,outChannels)
        self.leaky = nn.LeakyReLU(0.2,inplace = True)
        self.injectNoise1= noiseInjection(outChannels)
        self.injectNoise2 = noiseInjection(outChannels)
        self.adain1 = AdaIN(outChannels,wDim)
        self.adain2 = AdaIN(outChannels,wDim)

    def forward (self,x,w):
        x = self.adain1(self.leaky(self.injectNoise1(self.conv1(x))),w)
        x = self.adain2(self.leaky(self.injectNoise2(self.conv2(x))),w)
        return x

class ConvBlock(nn.Module):
    def __init__(self,inChannels, outChannels):
        super(ConvBlock,self).__init__()
        self.conv1 = WSConv2d(inChannels,outChannels)
        self.conv2 = WSConv2d(outChannels,outChannels)
        self.leaky = nn.LeakyReLU(0.2)
    
    def forward(self,x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x

class Generator(nn.Module):
    def __init__(self, zDim, wDim, inChannels, imgChannels=3):
        super(Generator, self).__init__()
        self.startingConstant = nn.Parameter(torch.ones((1, inChannels,4,4)))
        self.map = mappingNetwork(zDim,wDim)
        self.initialAdain1 = AdaIN(inChannels,wDim)
        self.initialAdain2 = AdaIN(inChannels, wDim)
        self.initialNoise1 = noiseInjection(inChannels)
        self.initialNoise2 = noiseInjection(inChannels)
        self.initialConv = nn.Conv2d(inChannels,inChannels,kernelSize = 3,stride = 1,padding =1)
        self.leaky = nn.LeakyReLU(0.2, inplace = True)

        self.initialRgb = WSConv2d(inChannels,imgChannels,kernelSize=1,stride=1,padding=0)
        self.progBlocks,self.rgbLayers = (nn.ModuleList([]),nn.ModuleList([self.initialRgb]),)

        for i in range(len(factors)-1): 
            convInC = int(inChannels*factors[i])
            convOutC = int(inChannels*factors[i+1])
            self.progBlocks.append(GenBlock(convInC,convOutC,wDim))
            self.rgbLayers.append(WSConv2d(convOutC,imgChannels,kernelSize=1,stride=1,padding=0))


    def fadeIn(self,alpha,upscaled,generated):
        #alpha is scalar belonging to [0,1] and upscale.shape == generated.shape
        return torch.tanh(alpha*generated+(1-alpha)*upscaled)

    def forward(self, noise,alpha,steps):
        w = self.map(noise)
        x = self.initialAdain1(self.initialNoise1(self.startingConstant),w)
        x = self.initialConv(x)
        out = self.initialAdain2(self.leaky(self.initialNoise2(x)),w)

        if steps == 0:
            return self.initialRgb(x)
        
        for step in range(steps):
            upscaled = F.interpolate(out, scaleFactor = 2, mode = "bilinear")
            out = self.progBlocks[steps](upscaled,w)

        finalUpscaled = self.rgbLayers[steps-1](upscaled)
        finalOut = self.rgbLayers[steps](out)
        return self.fadeIn(alpha, finalUpscaled,finalOut)
