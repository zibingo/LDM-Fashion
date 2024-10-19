
import torch.nn as nn 
from torchvision import transforms, models
import torch
layers_map = {
    '3':'relu1_2',
    '8':'relu2_2',
    '13':'relu3_2',
    '22':'relu4_2',
    '31':'relu5_2',
    '35':'relu5_4',
    }
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class VGG_FeatureExtractor(nn.Module):
    # Extract features from intermediate layers of a network

    def __init__(self,extracted_layers,rgb_range=1):
        super(VGG_FeatureExtractor, self).__init__()
        self.vgg_features = models.vgg19(pretrained=True).features
        self.extracted_layers = extracted_layers
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean =  MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        features = {}
        x = (x + 1.0) / 2
        x = self.sub_mean(x)
        for id, module in list(self.vgg_features._modules.items()):
            x = module(x)
            if id in layers_map.keys():
                if layers_map[id] in self.extracted_layers:
                    features[layers_map[id]] = x
        
        return features
    
class VGG_Loss(nn.Module):
    def __init__(self,perceptual_layers,style_layers):
        super(VGG_Loss, self).__init__()
        self.perceptual_layers = perceptual_layers
        self.style_layers = style_layers
        self.layers =list(set(perceptual_layers+style_layers))
        self.VGG_FeatureExtractor = VGG_FeatureExtractor(self.layers)

    def GramMatrix(self,features):
        b,c,h,w = features.shape
        features = features.view(b,c,h*w)
        gram = torch.bmm(features,features.transpose(1,2))
        # return gram
        return gram.div(c * h * w)

    def Perceptual_Loss(self,image1,image2,layer):
        image1_feature = self.VGG_FeatureExtractor(image1)[layer]
        image2_feature = self.VGG_FeatureExtractor(image2)[layer]
        return mean_flat((image1_feature - image2_feature) ** 2)
    def Batch_Percept_Loss(self,image1,image2):
        batch_size,_,_,_ = image1.shape
        perceptual_loss = torch.zeros(batch_size,).to("cuda")
        
        for layer in self.perceptual_layers:
             perceptual_loss += self.Perceptual_Loss(image1,image2,layer)
       
        return perceptual_loss
    def Style_Loss(self,image1,image2,layer):
        image1_feature = self.VGG_FeatureExtractor(image1)[layer]
        image2_feature = self.VGG_FeatureExtractor(image2)[layer]
        image1_gram = self.GramMatrix(image1_feature)
        image2_gram = self.GramMatrix(image2_feature)
        return mean_flat((image1_gram - image2_gram) ** 2)
    
    def Batch_Style_Loss(self,image1,image2):
        batch_size,_,_,_ = image1.shape
        style_loss = torch.zeros(batch_size,).to("cuda")
        
        for layer in self.style_layers:
            style_loss += self.Style_Loss(image1,image2,layer)

        return style_loss