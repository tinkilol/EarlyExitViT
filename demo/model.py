import torchvision . transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch

import torch
import torchvision . transforms as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Big_model(nn.Module):
    """ Conditional Variational Auto-Encoder class. """

    def __init__(self, model, num_pred):
        super().__init__()

        if len(model.blocks) % num_pred != 0:
            num_pred = 6
        self.mini_trans = len(model.blocks) / num_pred
        """
        print(f"Number of blocks in the model: {len(model.blocks)}")
        print(f"Number of predictions: {num_pred}")
        print(f"Number of blocks per prediction: {self.mini_trans}")
        print("\n")
        """
        self.num_classes = model.num_classes
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre
        self.blocks = model.blocks
        self.norm = nn.ModuleList([])
        self.prediction_layers = nn.ModuleList([])

        for i in range(num_pred):
            #self.blocks.append(model.blocks)
            self.norm.append(model.norm)
            
            self.prediction_layers.append(nn.Sequential(model.fc_norm,
                                                        model.head_drop,
                                                        model.head))


        # make a prediction layer
    def _pos_embed(self, x):
        # original timm, JAX, and deit vit impl
        # pos_embed has entry for class token, concat then add
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim = 1
                         )
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward(self, x):

        # embedding layer
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        j = 0
        for i in range(len(self.blocks)):

            x = self.blocks[i](x)
            
            if (i + 1) % self.mini_trans == 0: 

                out = self.norm[j](x)

                out = self.prediction_layers[i](out[:,0])

                j += 1

            else:
                continue
    
        return out

class Big_model_confidence(nn.Module):
    """ Conditional Variational Auto-Encoder class. """

    def __init__(self, model):
        super().__init__()

        self.num_classes = model.module.num_classes
        self.mini_trans = model.module.mini_trans

        self.cls_token = model.module.cls_token
        self.pos_embed = model.module.pos_embed
        self.patch_embed = model.module.patch_embed
        self.pos_drop = model.module.pos_drop
        self.patch_drop = model.module.patch_drop
        self.norm_pre = model.module.norm_pre


        self.blocks = model.module.blocks
        self.norm = model.module.norm
        self.prediction_layers = model.module.prediction_layers

        self.confidence = nn.Linear(self.num_classes, self.num_classes)


        # make a prediction layer
    def _pos_embed(self, x):
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim = 1
                         )
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward(self, x, threshold = 1):

        # embedding layer
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
     
        j = 0

        # for loop through layers
        for i in range(len(self.blocks)):
            # through blocks
            x = self.blocks[i](x)
            
            if (i + 1) % self.mini_trans == 0: 

                out = self.norm[j](x)

                out = self.prediction_layers[i](out[:,0])
                confidence = self.confidence(out)

                # Softmax:
                softmax = torch.nn.Softmax(dim=1)
                confidence = softmax(confidence)

                # check if batch size = 1 (inference) and exit early if threshold is satisfied: 
                #if confidence.shape[0] == 1:
                if i > 2 and torch.max(confidence, dim=1).values.item() >= threshold:
                    layer_nr = i+1

                    return out, confidence, layer_nr
                
                j += 1

            else:
                continue

        layer_nr = i +1
        #return output, all_confidences
        return out, confidence, layer_nr
