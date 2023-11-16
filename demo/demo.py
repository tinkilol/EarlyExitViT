import torchvision.transforms as T
from torch import nn
import torch
from litdata import litdata
from model import Big_model, Big_model_confidence
import timm
from torch.utils.data import DataLoader
import streamlit as st
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels = {"0": "ak47", "1": "american-flag", "2": "backpack", "3": "baseball-bat", "4": "baseball-glove", "5": "basketball-hoop", "6": "bat", "7": "bathtub", "8": "bear", "9": "beer-mug", "10": "billiards", "11": "binoculars", "12": "birdbath", "13": "blimp", "14": "bonsai-101", "15": "boom-box", "16": "bowling-ball", "17": "bowling-pin", "18": "boxing-glove", "19": "brain-101", "20": "breadmaker", "21": "buddha-101", "22": "bulldozer", "23": "butterfly", "24": "cactus", "25": "cake", "26": "calculator", "27": "camel", "28": "cannon", "29": "canoe", "30": "car-tire", "31": "cartman", "32": "cd", "33": "centipede", "34": "cereal-box", "35": "chandelier-101", "36": "chess-board", "37": "chimp", "38": "chopsticks", "39": "cockroach", "40": "coffee-mug", "41": "coffin", "42": "coin", "43": "comet", "44": "computer-keyboard", "45": "computer-monitor", "46": "computer-mouse", "47": "conch", "48": "cormorant", "49": "covered-wagon", "50": "cowboy-hat", "51": "crab-101", "52": "desk-globe", "53": "diamond-ring", "54": "dice", "55": "dog", "56": "dolphin-101", "57": "doorknob", "58": "drinking-straw", "59": "duck", "60": "dumb-bell", "61": "eiffel-tower", "62": "electric-guitar-101", "63": "elephant-101", "64": "elk", "65": "ewer-101", "66": "eyeglasses", "67": "fern", "68": "fighter-jet", "69": "fire-extinguisher", "70": "fire-hydrant", "71": "fire-truck", "72": "fireworks", "73": "flashlight", "74": "floppy-disk", "75": "football-helmet", "76": "french-horn", "77": "fried-egg", "78": "frisbee", "79": "frog", "80": "frying-pan", "81": "galaxy", "82": "gas-pump", "83": "giraffe", "84": "goat", "85": "golden-gate-bridge", "86": "goldfish", "87": "golf-ball", "88": "goose", "89": "gorilla", "90": "grand-piano-101", "91": "grapes", "92": "grasshopper", "93": "guitar-pick", "94": "hamburger", "95": "hammock", "96": "harmonica", "97": "harp", "98": "harpsichord", "99": "hawksbill-101", "100": "head-phones", "101": "helicopter-101", "102": "hibiscus", "103": "homer-simpson", "104": "horse", "105": "horseshoe-crab", "106": "hot-air-balloon", "107": "hot-dog", "108": "hot-tub", "109": "hourglass", "110": "house-fly", "111": "human-skeleton", "112": "hummingbird", "113": "ibis-101", "114": "ice-cream-cone", "115": "iguana", "116": "ipod", "117": "iris", "118": "jesus-christ", "119": "joy-stick", "120": "kangaroo-101", "121": "kayak", "122": "ketch-101", "123": "killer-whale", "124": "knife", "125": "ladder", "126": "laptop-101", "127": "lathe", "128": "leopards-101", "129": "license-plate", "130": "lightbulb", "131": "light-house", "132": "lightning", "133": "llama-101", "134": "mailbox", "135": "mandolin", "136": "mars", "137": "mattress", "138": "megaphone", "139": "menorah-101", "140": "microscope", "141": "microwave", "142": "minaret", "143": "minotaur", "144": "motorbikes-101", "145": "mountain-bike", "146": "mushroom", "147": "mussels", "148": "necktie", "149": "octopus", "150": "ostrich", "151": "owl", "152": "palm-pilot", "153": "palm-tree", "154": "paperclip", "155": "paper-shredder", "156": "pci-card", "157": "penguin", "158": "people", "159": "pez-dispenser", "160": "photocopier", "161": "picnic-table", "162": "playing-card", "163": "porcupine", "164": "pram", "165": "praying-mantis", "166": "pyramid", "167": "raccoon", "168": "radio-telescope", "169": "rainbow", "170": "refrigerator", "171": "revolver-101", "172": "rifle", "173": "rotary-phone", "174": "roulette-wheel", "175": "saddle", "176": "saturn", "177": "school-bus", "178": "scorpion-101", "179": "screwdriver", "180": "segway", "181": "self-propelled-lawn-mower", "182": "sextant", "183": "sheet-music", "184": "skateboard", "185": "skunk", "186": "skyscraper", "187": "smokestack", "188": "snail", "189": "snake", "190": "sneaker", "191": "snowmobile", "192": "soccer-ball", "193": "socks", "194": "soda-can", "195": "spaghetti", "196": "speed-boat", "197": "spider", "198": "spoon", "199": "stained-glass", "200": "starfish-101", "201": "steering-wheel", "202": "stirrups", "203": "sunflower-101", "204": "superman", "205": "sushi", "206": "swan", "207": "swiss-army-knife", "208": "sword", "209": "syringe", "210": "tambourine", "211": "teapot", "212": "teddy-bear", "213": "teepee", "214": "telephone-box", "215": "tennis-ball", "216": "tennis-court", "217": "tennis-racket", "218": "theodolite", "219": "toaster", "220": "tomato", "221": "tombstone", "222": "top-hat", "223": "touring-bike", "224": "tower-pisa", "225": "traffic-light", "226": "treadmill", "227": "triceratops", "228": "tricycle", "229": "trilobite-101", "230": "tripod", "231": "t-shirt", "232": "tuning-fork", "233": "tweezer", "234": "umbrella-101", "235": "unicorn", "236": "vcr", "237": "video-projector", "238": "washing-machine", "239": "watch-101", "240": "waterfall", "241": "watermelon", "242": "welding-mask", "243": "wheelbarrow", "244": "windmill", "245": "wine-bottle", "246": "xylophone", "247": "yarmulke", "248": "yo-yo", "249": "zebra", "250": "airplanes-101", "251": "car-side-101", "252": "faces-easy-101", "253": "greyhound", "254": "tennis-shoes", "255": "toad", "256": "clutter"}

# path to the caltech256 folder
data_path = "/Users/amir/Documents/UiO/IN5310 – Advanced Deep Learning for Image Analysis/project3/"
# path to the big model
big_model_name = "/Users/amir/Documents/UiO/IN5310 – Advanced Deep Learning for Image Analysis/project3/big_model_data_aug_90acc.pt"
# path to the confidence model
confidence_model_name = "/Users/amir/Documents/UiO/IN5310 – Advanced Deep Learning for Image Analysis/project3/big_model_with_confidence.pt"  # feil modell, bare for å teste

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

dataset = "Caltech256_demo"
num_classes = 257

valdata = litdata.LITDataset(dataset, data_path, train=False, override_extensions = ["jpg", "cls"]).map_tuple(*postprocess)

base = "vit_base_patch16_224"

pretrained_model = base
model = timm.create_model(pretrained_model, pretrained=True, num_classes = num_classes).to(device)


big_model = Big_model(model, num_pred=12)
big_model = torch.nn.DataParallel(big_model)

model_save = torch.load(big_model_name, map_location='cpu')
big_model.load_state_dict(model_save["model"])


confidence_model = Big_model_confidence(big_model)
confidence_model = torch.nn.DataParallel(confidence_model)
model_save = torch.load(confidence_model_name, map_location='cpu')
confidence_model.load_state_dict(model_save["model"])
print("Loaded pretrained weights for confidence model")

big_model.eval()
confidence_model.eval()

def sample_val(num_samples, valdata):

    sampled_valdata = []
    for i in range(num_samples):
        img = np.random.randint(1, len(valdata))
        sampled_valdata.append(valdata[img])

    return sampled_valdata


def inference_confidence(model, threshold, val_loader):

    predictions, true_labels, confidences, layer_nums = [],[],[],[]
    start_time = time.time()
    for val_data, label_true in val_loader:
        val_data, label_true = val_data.to(device), label_true.to(device)
        
        output, confidence, layer_num = model(val_data, threshold)

        predictions.append(output.argmax(dim=1).item())
        true_labels.append(label_true)
        confidences.append(round(torch.max(confidence, dim=1).values.item(), 3))
        layer_nums.append(layer_num)
    end_time = time.time()

    execution_time =  end_time - start_time

    return predictions, true_labels, confidences, layer_nums, execution_time

def inference_bigmodel(model, val_loader):

    predictions, true_labels, confidences, layer_nums = [],[],[],[]
    start_time = time.time()
    for val_data, label_true in val_loader:
        val_data, label_true = val_data.to(device), label_true.to(device)
    
        output = model(val_data)
        predictions.append(output.argmax(dim=1).item())
        true_labels.append(label_true)

    end_time = time.time()

    execution_time =  end_time - start_time

    return predictions, true_labels, execution_time


def convert_seconds(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return minutes, remaining_seconds

def get_images(val_sampled, layer_nums, num_unqique_layers = 5):

    images = []
    indicies = []
    unique_layer_nums = []
    for idx, layer_num in enumerate(layer_nums):
        if layer_num not in unique_layer_nums and layer_num <12:
            unique_layer_nums.append(layer_num)
            images.append(val_sampled[idx][0])
            indicies.append(idx)

        if len(unique_layer_nums) == num_unqique_layers:
            unique_layer_nums.sort()  

            combined = list(zip(indicies, images))
            sorted_combined = sorted(combined, key=lambda x: unique_layer_nums.index(layer_nums[x[0]]))
            indicies, images = zip(*sorted_combined)

            indicies = list(indicies)
            images = list(images)
            
            
            return images, indicies
        
    unique_layer_nums.sort()  

    combined = list(zip(indicies, images))
    sorted_combined = sorted(combined, key=lambda x: unique_layer_nums.index(layer_nums[x[0]]))
    indicies, images = zip(*sorted_combined)

    indicies = list(indicies)
    images = list(images)

    #print("indicies: ", indicies)
    #print("images: ", images)
    return images, indicies


def reverse_postprocess(image_tensor):
    # Reverse the normalization
    reverse_normalize = T.Normalize(mean=[-mean/std for mean, std in zip(in_mean, in_std)],
                                    std=[1/std for std in in_std])
    
    original_image = reverse_normalize(image_tensor)

    # Convert the PyTorch tensor to a PIL image
    original_image = T.ToPILImage()(original_image)

    return original_image
##########################################################################################################################################################################


def main():

    st.title("Lightning inference time with ViTs: Early confidence stopping")
    threshold = st.number_input(label="Choose the confidence threshold", min_value=0.0, max_value=1.0, value = 0.95, step=0.01)
    num_samples = st.number_input(label="Choose number of images to test", min_value=1, max_value=len(valdata), value = 1, step=1)
    
    val_sampled = sample_val(num_samples, valdata)
    val_loader_sampled = DataLoader(val_sampled, shuffle=False, batch_size = 1)   

    run_demo = st.button("Run demo")

    if run_demo:
        with st.spinner("Processing..."):
            st.header('Performance indicators :chart_with_upwards_trend:', divider='blue')
            fig_col1, fig_col2 = st.columns(2)
            predictions_confidence, true_labels_confidence, confidences, layer_nums, execution_time_confidence = inference_confidence(confidence_model, threshold, val_loader_sampled)

            predictions_big_model, true_labels_big_model, execution_time_big_model = inference_bigmodel(big_model, val_loader_sampled)
            
            min_conf, sec_conf = convert_seconds(execution_time_confidence)
            min_big, sec_big = convert_seconds(execution_time_big_model)

            conf_perct_saved = (1-round(execution_time_confidence,2)/round(execution_time_big_model,2))*100


            with fig_col1:
                container = st.empty()
                if min_conf <10 and sec_conf <10:
                    container.metric("Time confidence model", f"{0}{round(min_conf)}:{0}{round(sec_conf)} ",  f"{conf_perct_saved:.2f}% faster")
                elif min_conf <10:
                    container.metric("Time confidence model", f"{0}{round(min_conf)}:{round(sec_conf)}", f"{conf_perct_saved:.2f}% faster")
                elif sec_conf <10:
                    container.metric("Time confidence model", f"{round(min_conf)}:{0}{round(sec_conf)}", f"{conf_perct_saved:.2f}% faster")
                else:
                    container.metric("Time confidence model", f"{round(min_conf)}:{round(sec_conf)}", f"{conf_perct_saved:.2f}% faster")
                
                accuracy_conf = st.empty()
                accuracy_conf.metric("Accuracy confidence model", accuracy_score(true_labels_confidence, predictions_confidence))

            
            with fig_col2:
                container2 = st.empty()
                if min_big <10 and sec_big <10:
                    container2.metric("Time regular model", f"{0}{round(min_big)}:{0}{round(sec_big)}")
                elif min_big <10:
                    container2.metric("Time regular model ", f"{0}{round(min_big)}:{round(sec_big)}")
                elif sec_big <10:
                    container2.metric("Time regular model", f"{round(min_big)}:{0}{round(sec_big)}")
                else:
                    container2.metric("Time regular model", f"{round(min_big)}:{round(sec_big)}")

                accuracy_big = st.empty()
                accuracy_big.metric("Accuracy regular model",accuracy_score(true_labels_big_model, predictions_big_model))

            similarity = st.empty()
            similarity.metric("How often the same prediction", accuracy_score(predictions_confidence, predictions_big_model))

            st.header('Example images :magic_wand:', divider='blue')
            images, indicies = get_images(val_sampled, layer_nums, num_unqique_layers = 5)
            #print("image: ", images)

            for i in range(len(images)):
                images[i] = reverse_postprocess(images[i])

            num_columns = len(images)

            # Create a layout with specified number of columns
            columns = st.columns(num_columns)

            # Display text and images in each column
            for i, column in enumerate(columns):
                # Apply custom CSS styling for the current column
                column.markdown(f"""
                    <style>
                    [data-testid=column]:nth-of-type({i + 1}) [data-testid=stVerticalBlock] {{
                        gap: 0rem !important;
                    }}
                    </style>
                    """, unsafe_allow_html=True)

                # Display image and text in the column
                column.image(images[i], use_column_width=True)

                column.write(f"<span style='font-size: small;'>Exit layer: {layer_nums[indicies[i]]}</span>", unsafe_allow_html=True)
                column.write(f"<span style='font-size: small;'>Confidence: {round(confidences[indicies[i]], 2)}</span>", unsafe_allow_html=True)
                column.write(f"<span style='font-size: small;'>Pred conf: {labels[str(predictions_confidence[indicies[i]])]}</span>", unsafe_allow_html=True)
                column.write(f"<span style='font-size: small;'>Pred reg: {labels[str(predictions_big_model[indicies[i]])]}</span>", unsafe_allow_html=True)
                column.write(f"<span style='font-size: small;'>True class: {labels[str(true_labels_confidence[indicies[i]].item())]}</span>", unsafe_allow_html=True)

            st.header('Early exiting overview :bar_chart:', divider='blue')


            counts = [0]*12
            accordance = [0]*12
            for i in range(len(predictions_confidence)):
                counts[layer_nums[i]-1] += 1
                accordance[layer_nums[i]-1] += int(predictions_confidence[i] == predictions_big_model[i])

            fig, ax = plt.subplots()
            ax = plt.bar(range(1, 13), counts)
            plt.xlabel('Prediction Layer at Exit')
            plt.ylabel('Number of Images')
            plt.title(f'ViT Exit Layers (Threshold = {threshold})')

            for layer, count in enumerate(counts):
                if count == 0:
                    plt.text(layer+1, count, f'{accordance[layer]:.2f}', ha='center', va='bottom')
                else:
                    plt.text(layer+1, count, f'{accordance[layer]/count:.2f}', ha='center', va='bottom')
            st.pyplot(fig)


if __name__ == '__main__':
    main()


