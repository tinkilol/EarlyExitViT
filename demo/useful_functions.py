import torch
import tqdm
import torch.nn.functional as F
import os
import numpy as np
import random
import torchvision . transforms as T
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed_value=5310):
    "Set same seed to all random operations for reproduceability purposes"
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    

def confidence_loss(current_layer_pred, last_layer_pred):
    loss = nn.CrossEntropyLoss()
    return loss(current_layer_pred,last_layer_pred)
    

def train(args, model: nn.Module, train_iter: DataLoader, optimizer: Optimizer, scheduler: _LRScheduler, loss_fun, num_classes):
    """ Training process of Neural-Network.
        Parameters
        ----------
        model: nn.Module - the neural network to train
        train_iter: torch.utils.data.DataLoader
        optimizer: torch.optim.Optimizer - Optimzer method
        scheduler: torch.optim.lr_scheduler._LRScheduler - Scheduler method
        Returns
        -------
        blank
    """

    model.train()
    #layer_scaler = np.flip(np.arange(1, args.num_pred + 1))
    # annen scale, exponentiel
    for train_data, label_true in tqdm.tqdm(train_iter):
        train_data, label_true = train_data.to(device), label_true.to(device)
        optimizer.zero_grad()
        label_pred = model(train_data)
        label_true = F.one_hot(label_true, num_classes).to(torch.float32)

        if args.stacked or args.big_model or args.monte_carlo or args.last_layer or args.horizontal:
            total_loss = 0
            for i, layer in enumerate(label_pred):
                #loss_layer = layer_scaler[i] * loss_fun(layer, label_true)
                loss_layer = loss_fun(layer, label_true)
                total_loss += loss_layer
            total_loss.backward()

        else:
            loss = loss_fun(label_pred, label_true)
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()

def train_confidence(args, model: nn.Module, train_iter: DataLoader, optimizer: Optimizer, scheduler: _LRScheduler, loss_fun, num_classes):
    model.train()
    eval = False

    # freeze layers here
    #for param in model.parameters():
    #        param.requires_grad = False

    for name, param in model.named_parameters():
            #print("name: ", name)
            #print("name: ", param)
        if name == "module.confidence.weight" or name == "module.confidence.bias":
            print("name: ", name)
            print("name: ", param)
            param.requires_grad = True
        else:
            param.requires_grad = False

    for train_data, label_true in tqdm.tqdm(train_iter):
        train_data, label_true = train_data.to(device), label_true.to(device)
        optimizer.zero_grad()

        # all_confidence is confidence for all layers except the last layer
        output, all_confidence = model(train_data, eval)
        # confidence "ground truth" is the last layer prediction
        all_confidence = all_confidence[:-1]

        if args.hinge_loss:
            last_layer_pred = output[-1]
        else:
            last_layer_pred = torch.argmax(output[-1], dim=1)


        total_loss = 0
        for conf_layer in all_confidence:
            #print("conf layer shape:", conf_layer.shape)
            # smooth L1 loss
            # log cosh los
            # hvis last_layer predicter 1, så er confidence på layer x uttifra softmax på prediction 1
            # vekte ut ifra hvor mange lag unna hverandre de er
            loss_layer = loss_fun(conf_layer, last_layer_pred)
            total_loss += loss_layer

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()

    return

# created by Cornelius and Amir in previous class
@torch.no_grad()
def evaluate(args, model: nn.Module, data_iter: DataLoader, num_blocks):
    """ Retrives the accuracy of the neural network 
        Parameters
        ----------
        model: nn.Module - the neural network used to predict
        data_iter: torch.utils.data.DataLoader
        Returns
        -------
        accuracy: float value
    """
    
    model.eval()
    if args.train_confidence:

        true_labels, last_layer_predictions, all_predictions, all_confidences = [], [], [], []

        for val_data, label_true in tqdm.tqdm(data_iter):
            val_data, label_true = val_data.to(device), label_true.to(device)
            output, confidences = model(val_data)

            true_labels += label_true.tolist()
            last_layer_predictions += output[-1].argmax(dim=1).tolist()
            output = output.argmax(dim=2)
            output = torch.transpose(output, 0, 1)
            all_predictions.append(output)
            confidences = torch.max(confidences, dim=2).values
            confidences = torch.transpose(confidences, 0, 1)
            all_confidences.append(confidences)


        accuracy = (torch.tensor(last_layer_predictions) == torch.tensor(true_labels)).float().mean() * 100.0
        print(f"Accuracy: {accuracy:.1f}%")



        return all_predictions, all_confidences

    elif args.stacked or args.big_model or args.horizontal or args.last_layer or args.monte_carlo:
        if args.horizontal:
            num_blocks = 12
        true_labels, predictions = [], []
        for i in range(num_blocks):
            empty = []
            predictions.append(empty)

        for val_data, label_true in tqdm.tqdm(data_iter):
            val_data, label_true = val_data.to(device), label_true.to(device)
            output = model(val_data)
            true_labels += label_true.tolist()
            for i in range(len(output)):
                predictions[i] += output[i].argmax(dim=1).tolist()

        if args.horizontal:
            for i in range(num_blocks):
                results = (torch.tensor(predictions[i]) == torch.tensor(true_labels)).float().mean() * 100.0
                similarity = (torch.tensor(predictions[i]) == torch.tensor(predictions[-1])).float().mean() * 100.0
                print(f"passing through {num_blocks} Layer {i + 1}'s give accuracy: {results:.1f}%      Similarity to last prediction layer: {similarity:.1f}%")
        
        else:
            for i in range(len(predictions)):
                results = (torch.tensor(predictions[i]) == torch.tensor(true_labels)).float().mean() * 100.0
                similarity = (torch.tensor(predictions[i]) == torch.tensor(predictions[-1])).float().mean() * 100.0
                if args.last_layer:
                    if i == 0:
                        print(f"Predictions after whole ViT accuracy: {results:.1f}%      Similarity to last prediction layer: {similarity:.1f}%")
                    else:
                        print(f"last 2block layer {i + 1} accuracy: {results:.1f}%      Similarity to last prediction layer: {similarity:.1f}%")
                else:
                    print(f"Layer {i + 1} accuracy: {results:.1f}%      Similarity to last prediction layer: {similarity:.1f}%")

    else:
        true_labels, predictions = [], []
        for val_data, label_true in tqdm.tqdm(data_iter):
            val_data, label_true = val_data.to(device), label_true.to(device)
            output = model(val_data)
            true_labels += label_true.tolist()
            predictions += output.argmax(dim=1).tolist()

        results = (torch.tensor(predictions) == torch.tensor(true_labels)).float().mean() * 100.0
        print(f"Accuracy: {results:.1f}%")

    return results

# created by Cornelius and Amir in previous class
def train_epochs(args, model, loss_fun, optimizer, scheduler, train_iter, val_iter, num_preds):
    """ Calculating training and validation accuracy per epoch.
        Parameters
        ----------
        epochs: int
        args: argparse.ArgumentParser
        optimizer: torch.optim.Optimizer - Optimzer method
        scheduler: torch.optim.lr_scheduler._LRScheduler - Scheduler method
        train_iter: torch.utils.data.DataLoader
        val_iter: torch.utils.data.DataLoader
        Returns
        -------
        train_accuracy: float
        val_accuracy: float
    """
    train_acc = [0]
    val_acc = [0]
    for epoch in range(args.epochs):
        if args.test_model and not args.train_confidence:
            print("Training:")
            train_accuracy = evaluate(args, model, train_iter, num_preds)

            print("Validation:")
            val_accuracy = evaluate(args, model, val_iter, num_preds)

        elif args.train_confidence:

            print(f"epoch: {epoch + 1}")
            train_confidence(args, model, train_iter, optimizer, scheduler, loss_fun, model.module.num_classes)
            #evaluate(args, model, train_iter, num_preds)
            
            """
            if (epoch + 1) % 5 == 0:
                print("Training:")
                evaluate(args, model, train_iter, num_preds)
                
                print("Validation:")
                evaluate(args, model, val_iter, num_preds)
            """
            
            if epoch+1 == args.epochs:
                all_predictions, all_confidences = evaluate(args, model, val_iter, num_preds)

                if args.save:
                    print("\n")
                    if args.stacked:
                        model_name = f"Stacked_model_vit_pretrained_model_with_confidence.pt"
                    elif args.big_model:
                        model_name = f"big_model_with_confidence_bs128.pt"

                    state_dict = {
                        "model": model.state_dict(),
                        "num_preds": num_preds,
                        "optimizer": optimizer.state_dict()
                    }
                    torch.save(state_dict, model_name)
                    print(f"Model saved as {model_name} in epoch {epoch + 1}")
                    print("\n")
                
            
                
                return all_predictions, all_confidences
        else:
            #print(f"Validation accuracy before training:\n")
            #evaluate(args, model, val_iter, args.num_pred)
            print(f"epoch: {epoch + 1}")
            train(args, model, train_iter, optimizer, scheduler, loss_fun, args.num_classes)

            if (epoch + 1) % 5 == 0:
                if (epoch + 1) % 10 == 0:
                    print("Training:")
                    train_accuracy = evaluate(args, model, train_iter, num_preds)
                print("Validation:")
                val_accuracy = evaluate(args, model, val_iter, num_preds)

            elif args.big_model:
                val_accuracy = evaluate(args, model, val_iter, num_preds)
                if (epoch + 1) % 5 == 0:
                    print("Training:")
                    train_accuracy = evaluate(args, model, train_iter, num_preds)


            if args.save:
                print("\n")
                if args.stacked:
                    model_name = f"Stacked_model_fromScratch_residual_every3layer.pt"
                elif args.big_model:
                    model_name = f"big_model_residual_every3layer.pt"
                elif args.big_model_residual:
                    model_name = f"big_model_residual_every12layer.pt"
                elif args.horizontal:
                    model_name = f"Horizontal_model_pretrained.pt"
                elif args.monte_carlo:
                    model_name = f"Monte_carlo_1layer_model_pretrained.pt"
                elif args.last_layer:
                    model_name = f"Stacked_last1layer_fromscratch_model.pt"
                state_dict = {
                    "model": model.state_dict(),
                    "num_preds": num_preds,
                    "optimizer": optimizer.state_dict()
                }
                torch.save(state_dict, model_name)
                print(f"Model saved as {model_name} in epoch {epoch + 1}")
                print("\n")

    return

