import torchvision . transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch

import torch
import torchvision . transforms as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# stacked vit - line 19
# stacked with last blocks - line 82
# stacked confidence - line 153
# big model - line 258
# big model confidence - line 
# horizontal - line 343
# monte carlo - line 413

class Stacked_ViT(nn.Module):
    """ Conditional Variational Auto-Encoder class. """

    def __init__(self, model, num_pred):
        super().__init__()

        self.num_stacks = num_pred
        self.num_classes = model.num_classes

        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre

        print(f"Number of stacked ViT's in the model: {num_pred}")
        print("\n")

        self.blocks = nn.ModuleList([])
        self.norm = nn.ModuleList([])
        self.prediction_layers = nn.ModuleList([])
        
        for i in range(self.num_stacks):
            self.blocks.append(model.blocks)
            
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
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        output = []
        for i in range(self.num_stacks):
            if i % 3 == 0:
                if i != 0:
                    x_res_start = x_res_start + self.blocks[i](x)
                    x = x_res_start
                else:
                    x_res_start = self.blocks[i](x)
                    x = x_res_start

            else:
            # through blocks
                x = self.blocks[i](x)

            out = self.norm[i](x)
            
            out = self.prediction_layers[i](out[:,0]) 

            output.append(out)

        output = torch.stack(output)
    
        return output
    
class Stacked_ViT_confidence(nn.Module):
    """ Conditional Variational Auto-Encoder class. """

    def __init__(self, model):
        super().__init__()

        self.num_classes = model.num_classes
        self.num_stacks = model.num_stacks

        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre

        print(f"Number of stacked ViT's in the model: {self.num_stacks}")
        print("\n")

        self.blocks = model.blocks
        self.norm = model.norm
        self.prediction_layers = model.prediction_layers

        self.confidence = nn.Linear(self.num_classes, self.num_classes)

        # make a prediction layer
    def _pos_embed(self, x):
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim = 1
                         )
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward(self, x, threshold = 1):
        # default threshold to 1 during training = never exit early

        # embedding layer
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # for loop through layere
        output = []
        all_confidences = []
        for i in range(self.num_stacks):
            x = self.blocks[i](x)

            out = self.norm[i](x)

            out = self.prediction_layers[i](out[:, 0])

            confidence = self.confidence(out)

            # Softmax:
            softmax = torch.nn.Softmax(dim=1)
            confidence = softmax(confidence)

            # check if batch size = 1 (inference) and exit early if threshold is satisfied: 
            if confidence.shape[0] == 1:
                if torch.max(confidence) >= threshold:
                    return output, confidence

            output.append(out)

            all_confidences.append(confidence)

        
        output = torch.stack(output)
        all_confidences = torch.stack(all_confidences)
        all_confidences = all_confidences[:-1]
        
        return output, all_confidences 



    
class Stacked_last_layer_ViT(nn.Module):
    """ Conditional Variational Auto-Encoder class. """

    def __init__(self, model, num_pred):
        super().__init__()

        self.num_stacks = num_pred

        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre
        self.original_vit_block = model.blocks

        print(f"Number of stacked ViT's in the model: {num_pred}")
        print("\n")

        self.blocks = nn.ModuleList([])
        self.norm = nn.ModuleList([])
        self.prediction_layers = nn.ModuleList([])
        
        for i in range(self.num_stacks - 1):
            self.blocks.append(model.blocks[11:])
            
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
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        output = []

        x = self.original_vit_block(x)
        out_vit = self.norm[0](x)
        out_vit = self.prediction_layers[0](out_vit[:,0]) 

        output.append(out_vit)


        for i in range(self.num_stacks - 1):
            x = self.blocks[i](x)

            out = self.norm[i](x)
            
            out = self.prediction_layers[i](out[:,0]) 

            output.append(out)

        output = torch.stack(output)
    
        return output
    
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
        #self.blocks = nn.ModuleList([])
        self.norm = nn.ModuleList([])
        self.prediction_layers = nn.ModuleList([])

        for i in range(num_pred):
            #self.blocks.append(model.blocks)
            self.norm.append(model.norm)
            
            self.prediction_layers.append(nn.Sequential(model.fc_norm,
                                                        model.head_drop,
                                                        model.head))
            
        
        # edit vit_layer so that it is only a block

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

        # for loop through layere
        output = []
        j = 0
        for i in range(len(self.blocks)):

            x = self.blocks[i](x)
            
            if (i + 1) % self.mini_trans == 0: 

                #print(f"Predicting at layer {i + 1}")
                # predict for each layer
                out = self.norm[j](x)

                out = self.prediction_layers[i](out[:,0])

                #print(out.shape)
                
                output.append(out)
                j += 1
                
            else:
                continue
                
        output = torch.stack(output)
    
        return output
    
    

class Big_model_residual(nn.Module):
    """ Conditional Variational Auto-Encoder class. """

    def __init__(self, model, num_pred):
        super().__init__()
    
        if len(model.blocks) % num_pred != 0:
            num_pred = 6
        self.mini_trans = len(model.blocks) / num_pred


        print(f"Number of blocks in the model: {len(model.blocks)}")
        print(f"Number of predictions: {num_pred}")
        print(f"Number of blocks per prediction: {self.mini_trans}")
        print("\n")

        self.num_classes = model.num_classes

        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre

        self.blocks = model.blocks
        #self.blocks = nn.ModuleList([])
        self.norm = nn.ModuleList([])
        self.prediction_layers = nn.ModuleList([])

        for i in range(num_pred):
            #self.blocks.append(model.blocks)
            self.norm.append(model.norm)
            
            self.prediction_layers.append(nn.Sequential(model.fc_norm,
                                                        model.head_drop,
                                                        model.head))
            
        
        # edit vit_layer so that it is only a block

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

        # for loop through layere
        output = []
        j = 0
        for i in range(len(self.blocks)):
            if i % 11 == 0:
                if i != 0:
                    x_res_start = x_res_start + self.blocks[i](x)
                    x = x_res_start
                else:
                    x_res_start = self.blocks[i](x)
                    x = x_res_start

            else:
            # through blocks
                x = self.blocks[i](x)
            
            if (i + 1) % self.mini_trans == 0: 

                #print(f"Predicting at layer {i + 1}")
                # predict for each layer
                out = self.norm[j](x)

                out = self.prediction_layers[i](out[:,0])

                #print(out.shape)
                
                output.append(out)
                j += 1
                
            else:
                continue
                
        output = torch.stack(output)
    
        return output
    
class Big_model_confidence(nn.Module):
    """ Conditional Variational Auto-Encoder class. """

    def __init__(self, model):
        super().__init__()

        # need to add ".module." infront of "DataParallel" references due to the model object being
        self.num_classes = model.module.num_classes
        self.mini_trans = model.module.mini_trans
        #self.num_stacks = model.num_stacks

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
        # original timm, JAX, and deit vit impl
        # pos_embed has entry for class token, concat then add
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim = 1
                         )
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward(self, x, threshold = 1):
        # default threshold to 1 during training = never exit early

        # embedding layer
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # for loop through layere
        output = []
        all_confidences = []
        j = 0
        for i in range(len(self.blocks)):
            # through blocks
            x = self.blocks[i](x)
            
            if (i + 1) % self.mini_trans == 0: 
                out = self.norm[j](x)

                out = self.prediction_layers[i](out[:,0])
                confidence = self.confidence(out)

                softmax = torch.nn.Softmax(dim=1)
                confidence = softmax(confidence)

                # check if batch size = 1 (inference) and exit early if threshold is satisfied: 
                if confidence.shape[0] == 1:
                    if torch.max(confidence) >= threshold:
                        return output, confidence


                output.append(out)
                all_confidences.append(confidence)
                j += 1

            else:
                continue

        output = torch.stack(output)
        all_confidences = torch.stack(all_confidences)

        return output, all_confidences
    

class Horizontal_ViT(nn.Module):
    """ Conditional Variational Auto-Encoder class. """

    def __init__(self, model, num_pred):
        super().__init__()
        self.num_stacks = num_pred

        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre
        self.num_blocks = len(model.blocks)

        print(f"Horizontal Vit going through {self.num_stacks} layer 1's then pedictions, then {self.num_stacks} layer 2's then pedictions ...")
        print(f"Number of stacked ViT's in the model: {self.num_stacks}")
        print(f"Number of predictions = {self.num_blocks} = 1 for each layer in the ViT")
        print("\n")

        self.blocks = nn.ModuleList([])
        self.norm = nn.ModuleList([])
        self.prediction_layers = nn.ModuleList([])
        
        for i in range(self.num_stacks):
            self.blocks.append(model.blocks)
        
        for j in range(self.num_blocks):
            
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
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        output = []
        for layer in range(self.num_blocks):
            for stack in range(len(self.blocks)):
                
                x = self.blocks[stack][layer](x)

            out = self.norm[layer](x)
            
            out = self.prediction_layers[layer](out[:,0]) 

            output.append(out)

        output = torch.stack(output)
    
        return output
    


class Monte_Carlo_ViT(nn.Module):
    """ Conditional Variational Auto-Encoder class. """

    def __init__(self, model, num_pred):
        super().__init__()

        self.preds = num_pred

        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre

        self.layer_lengths = np.random.randint(low=4, high=len(model.blocks), size=(self.preds))

        print(f"Number of predictions: {num_pred}")
        print(f"Number of blocks in each new block: {self.layer_lengths}")


        self.blocks = nn.ModuleList([])
        self.norm = nn.ModuleList([])
        self.prediction_layers = nn.ModuleList([])

        for i, layer_lengths in enumerate(self.layer_lengths):    
            self.norm.append(model.norm)
            
            self.prediction_layers.append(nn.Sequential(model.fc_norm,
                                                        model.head_drop,
                                                        model.head))

            random_blocks = np.append(np.sort(np.random.randint(low=0, high=11, size=(layer_lengths - 1))), 11)

            print(f"Block {i + 1} will have blocks {random_blocks}")

            new_block = nn.ModuleList([])
            for i in random_blocks:
                new_block.append(model.blocks[i])

            self.blocks.append(new_block)

        #print(self.blocks)
        print("\n")

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
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        output = []
        for i, block in enumerate(self.blocks):
            for layer in block:
                x = layer(x)

            out = self.norm[i](x)
            
            out = self.prediction_layers[i](out[:,0]) 

            output.append(out)

        output = torch.stack(output)
    
        return output