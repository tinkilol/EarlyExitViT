import torch
import argparse
from torch.utils.data import DataLoader
from litdata import litdata
import torchvision . transforms as T
from torch import nn
import timm
from torchvision.transforms import v2
import numpy as np
import pandas as pd

from useful_functions import seed_everything, train, evaluate, train_epochs
from model import Stacked_ViT, Big_model, Horizontal_ViT, Monte_Carlo_ViT, Stacked_last_layer_ViT, Stacked_ViT_confidence, Big_model_confidence, Big_model_residual

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train model")
    parser.add_argument("--batch_size", default = 128)
    parser.add_argument("--lr", default = 0.001)
    parser.add_argument("--weight_decay", default = 0.1)
    parser.add_argument("--clip", default = 100)
    parser.add_argument("--epochs", default = 10)
    parser.add_argument("--seed", default = 5310)
    parser.add_argument("--num_pred", default = 12)

    parser.add_argument("--save", default = False)
    parser.add_argument("--fox", default = True)
    parser.add_argument("--train", default = False)
    parser.add_argument("--test_model", default = False)
    parser.add_argument("--imagenet", default = False)

    parser.add_argument("--stacked", default = False)
    parser.add_argument("--big_model", default = False) 
    parser.add_argument("--big_model_residual", default = False) 
    parser.add_argument("--horizontal", default = False)
    parser.add_argument("--last_layer", default = False)
    parser.add_argument("--monte_carlo", default = False)

    parser.add_argument("--train_confidence", default = True)
    parser.add_argument("--model_weights_confidence", default = "big_model_data_aug_90acc.pt")
    parser.add_argument("--hinge_loss", default = False)

    parser.add_argument("--pretrained", default = False)
    parser.add_argument("--data_augmentation", default = False)

    args = parser.parse_args()

    seed_everything(args.seed)

    print("Loading data ...") 
    print(f"batch size = {args.batch_size}") 
    if args.fox:
        data_path = "/fp/projects01/ec232/data/"

    else:
        #katinka
        #data_path = "C:/Users/laila/Documents/Studium/5.Semester/AdvancedDeepLearning/g05-p3"
        #coco
        data_path = "../../../../../../Desktop/"
        #amir
        #data_path = "/Users/amir/Documents/UiO/IN5310 – Advanced Deep Learning for Image Analysis/project3/"

    in_mean = [0.485, 0.456, 0.406]
    in_std = [0.229, 0.224, 0.225]

    postprocess = (
        T.Compose([
            T.ToTensor(),
            T.Resize((224, 224), antialias=True),
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  
            T.Normalize(in_mean, in_std),
        ]),
        nn.Identity(), 
    )

    total_traindata = litdata.LITDataset('Caltech256', data_path, override_extensions = ["jpg", "cls"] ).map_tuple(*postprocess)
    
    if args.data_augmentation:
        augmentations = [
                    v2.RandomResizedCrop(size=(224, 224), antialias=True),
                    v2.RandomPerspective(distortion_scale=0.6, p=1.0),
                    v2.RandomRotation(degrees=(0, 180)),
                    v2.ElasticTransform(alpha=250.0),
                    v2.Grayscale(),
                    v2.ColorJitter(brightness=.5, hue=.3)
                    ]

        for aug in augmentations:
            postprocess = (
                T.Compose([
                    T.ToTensor(),
                    T.Resize((224, 224), antialias=True),
                    aug,  
                    T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
                ]),
                nn.Identity(), 
            )
        
            traindata = litdata.LITDataset('Caltech256', data_path, override_extensions = ["jpg", "cls"] ).map_tuple(*postprocess)
            total_traindata += traindata

    if args.imagenet:
        dataset = "IN1k"
        num_classes = 1000
    else:
        dataset = "Caltech256"
        num_classes = 257
        
    print(f"\ndataset = {dataset}\nnumber of training data = {len(total_traindata)}\nnumber of classes = {num_classes}\nnumber of predictions = {args.num_pred}\n")
    valdata = litdata.LITDataset(dataset, data_path, train=False, override_extensions = ["jpg", "cls"]).map_tuple(*postprocess)


    if args.test_model:
        num_samples = 10_000
        print(f"using {num_samples} samples")
        val_data = []
        for i in range(num_samples):
            val_data.append(valdata[i])
        valdata = val_data
    
    train_loader = DataLoader(total_traindata, shuffle=True, batch_size = args.batch_size)
    val_loader = DataLoader(valdata, shuffle=False, batch_size = args.batch_size)
        
    print("Loading data done") 

    print("Loading model ...")
    args.num_classes = num_classes

    tiny = 'vit_tiny_patch16_224'
    base = "vit_base_patch16_224"

    if args.train and not args.train_confidence:

        pretrained_model = base
        model = timm.create_model(pretrained_model, pretrained=True, num_classes = num_classes).to(device)

        model.blocks = model.blocks[0]

        print(f"\npretrained model = {pretrained_model}\n")

        if args.pretrained:
            model_path = "fine_tuned_tiny_model_64%valacc.pt"
            model_save = torch.load(model_path, map_location='cpu')
            model.load_state_dict(model_save["model"])
            print("Loaded pretrained weights")


        if args.stacked:
            pretrained_model = tiny
            print("Model = stacked")
            model = Stacked_ViT(model, args.num_pred)

        elif args.big_model:
            pretrained_model = base
            model = timm.create_model(pretrained_model, pretrained=True, num_classes = num_classes).to(device)
            print("Model = big model")
            model = Big_model(model, args.num_pred)
        
        elif args.big_model_residual:
            pretrained_model = base
            model = timm.create_model(pretrained_model, pretrained=True, num_classes = num_classes).to(device)
            print("Model = big model with residual connection")
            model = Big_model_residual(model, args.num_pred)

        elif args.horizontal:
            pretrained_model = base
            print("Model = Horizontal")
            model = Horizontal_ViT(model, args.num_pred)

        elif args.monte_carlo:
            seed = np.random.randint(100000)
            seed_everything(seed)
            pretrained_model = base
            print("Model = Monte Carlo")
            model = Monte_Carlo_ViT(model, args.num_pred)

        elif args.last_layer:
            pretrained_model = base
            print("Model = stacked ViT using only last 2 blocks")
            model = Stacked_last_layer_ViT(model, args.num_pred)

            
    if args.train and args.train_confidence:

        tiny = 'vit_tiny_patch16_224'
        base = "vit_base_patch16_224"

        if args.big_model:
            pretrained_model = base
            model = timm.create_model(pretrained_model, pretrained=True, num_classes = num_classes).to(device)
            print("Model = big model")
            model = Big_model(model, args.num_pred)
            model = torch.nn.DataParallel(model)
         
            model_save = torch.load(args.model_weights_confidence, map_location='cpu')
            model.load_state_dict(model_save["model"])
            
            print("Loaded pretrained weights for big model confidence")
            model = Big_model_confidence(model)

        elif args.stacked:
            pretrained_model = tiny
            print("Model = stacked")
            model = Stacked_ViT(model, args.num_pred)
            model = torch.nn.DataParallel(model)

            model_save = torch.load(args.model_weights_confidence, map_location='cpu')
            model.load_state_dict(model_save["model"])

            print("Loaded pretrained weights for stacked confidence")
            model = Stacked_ViT_confidence(model)
        else:
            print("Something went wrong with confidence training")
            exit()

    
    model = torch.nn.DataParallel(model)
    print("Loading model done")

    print("Training model ...")
    if args.imagenet:
        evaluate(args, model, val_loader, args.num_pred)
    else: 
        if args.hinge_loss:
            loss = nn.HingeEmbeddingLoss()
        else:
            loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))
        if args.train_confidence:
            all_predictions, all_confidences = train_epochs(args, model, loss, optimizer, scheduler, train_loader, val_loader, args.num_pred)
        else:
            train_epochs(args, model, loss, optimizer, scheduler, train_loader, val_loader, args.num_pred)
    print("Training model done")

    saved_predictions, saved_confidences = [], []
    for batch in all_predictions:
        for i in range(batch.shape[0]):
            for j in range(batch.shape[1]):
                saved_predictions.append(batch[i][j].item())

    
    for batch in all_confidences:
        for i in range(batch.shape[0]):
            for j in range(batch.shape[1]):
                saved_confidences.append(batch[i][j].item())


    preds = all_predictions[0].shape[1] # should be the number of prediction heads
    picture_numbers = []
    prediction_number = []

    for i in range(1,len(valdata)+1):
        for j in range(preds):
            picture_numbers.append(i)
            prediction_number.append(j+1)

    print("len valdata: ", len(valdata))
    print("len picture numbers: ", len(picture_numbers))
    print("len prediciton numbers: ", len(prediction_number))
    print("len saved predictions: ", len(saved_predictions))
    print("len saved confidences: ", len(saved_confidences))

    df = pd.DataFrame({'Image nr': picture_numbers, 'Prediction nr': prediction_number, 'Predictions': saved_predictions, 'Confidences': saved_confidences})
    file_name = 'all_predictions_with_confidences_bs128.csv'
    df.to_csv(file_name, index=False)  # Set index to False to exclude the index column
    
    
    print("file name: ", file_name)
    print("batch size: ", args.batch_size)
    print("lr: ", args.lr)
    print("hinge loss: ", args.hinge_loss)
    print("epochs: ", args.epochs)

    
