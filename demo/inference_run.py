import torchvision.transforms as T
from torch import nn
import torch
from litdata import litdata
from model import Big_model, Big_model_confidence
import timm
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_val(num_samples, valdata):

    sampled_valdata = []
    for i in range(num_samples):
        img = np.random.randint(1, len(valdata))
        sampled_valdata.append(valdata[img])

    return sampled_valdata


def inference_confidence(model, threshold, val_loader):

    predictions, true_labels, confidences, layer_nums, execution_time = [],[],[],[],[]
    for val_data, label_true in val_loader:
        val_data, label_true = val_data.to(device), label_true.to(device)

        start_time = time.time()
        output, confidence, layer_num = model(val_data, threshold)
        end_time = time.time()



        predictions.append(output.argmax(dim=1).item())
        true_labels.append(label_true)
        confidences.append(round(torch.max(confidence, dim=1).values.item(), 3))
        layer_nums.append(layer_num)
        execution_time.append(end_time - start_time)

    return predictions, true_labels, confidences, layer_nums, execution_time

def inference_bigmodel(model, val_loader):

    predictions, true_labels, confidences, layer_nums, execution_time = [],[],[],[],[]
    for val_data, label_true in val_loader:
        val_data, label_true = val_data.to(device), label_true.to(device)

        start_time = time.time()
        output = model(val_data)
        end_time = time.time()

        predictions.append(output.argmax(dim=1).item())
        true_labels.append(label_true)
        execution_time.append(end_time - start_time)

    return predictions, true_labels, execution_time

def main():
    print("\n\n\n\n\n\n")
    data_path = "/fp/projects01/ec232/data/"
    big_model_name = "big_model_data_aug_90acc.pt"
    confidence_model_name = "big_model_with_confidence.pt" 

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
    dataset = "Caltech256"
    num_classes = 257
    num_samples = 10
    threshold = 0.95

    print("\nLoading data ...")
    valdata = litdata.LITDataset(dataset, data_path, train=False, override_extensions = ["jpg", "cls"]).map_tuple(*postprocess)
    val_sampled = sample_val(num_samples, valdata)
    val_loader_sampled = DataLoader(val_sampled, shuffle=False, batch_size = 1) 
    print("Loading data done")

    print("\nLoading models ...")
    pretrained_model = "vit_base_patch16_224"
    model = timm.create_model(pretrained_model, pretrained=True, num_classes = num_classes).to(device)
    print("\tPretrained based ViT loaded from timm")

    big_model = Big_model(model, num_pred=12)
    big_model = torch.nn.DataParallel(big_model)
    print("\n\tBig model created")
    model_save = torch.load(big_model_name, map_location='cpu')
    big_model.load_state_dict(model_save["model"])
    print("\tPretrained weights loaded to Big model")

    confidence_model = Big_model_confidence(big_model)
    confidence_model = torch.nn.DataParallel(confidence_model)
    print("\n\tBig confidence model loaded")
    model_save = torch.load(confidence_model_name, map_location='cpu')
    confidence_model.load_state_dict(model_save["model"])
    print("\tpretrained weights for confidence model loaded")
    print("Loading models done")

    print(f"\n\nConfidence threshold = {threshold}")

    print("\nEvaluating ...")
    time.sleep(3)
    big_model.eval()
    confidence_model.eval() 

    predictions_confidence, true_labels_confidence, confidences, layer_nums, execution_time_confidence = inference_confidence(confidence_model, threshold, val_loader_sampled)

    predictions_big_model, true_labels_big_model, execution_time_big_model = inference_bigmodel(big_model, val_loader_sampled)

    print("\n")
    for i in range(num_samples):
        print(f"Image {i + 1}:")
        print("Early stopping:")
        print(f"Early confidence stopping at layer {layer_nums[i]} / 12\nConfidence of predicting same as last layer = {confidences[i]}")
        saved_time = (1 - execution_time_confidence[i] / execution_time_big_model[i]) * 100
        print(f"Time saved by early confidence stopping = {saved_time:.2f}%")

        print("\nResult:")
        print(f"Early stop = {predictions_confidence[i]}")
        print(f"Last layer = {predictions_big_model[i]}")
        print(f"True label = {true_labels_big_model[i].item()}")
        
        print("\n")
    
    print("\nEvaluating done")

if __name__ == '__main__':
    main()
