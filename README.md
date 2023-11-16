# Early Confidence Stopping in Inference Time (ECSIT)
## Lightning inference time with ViTs

*Project by Cornelius, Amir, and Katinka*

### Project Description

This project aims to significantly increase inference speed by introducing stacked Vision Transformers (ViTs) with early exiting capabilities based on classification confidence.

### Model Architecture

<figure>
    <img width="235" alt="Model Architecture" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/d92549f0-2436-4241-a757-66a189cb4cee">
</figure>

The model architecture of this project will be similar to the base pre-trained ViT’s from the timm library by having an input embedding layer, a transformer block consisting of multiple transformer blocks and a prediction head.   
The difference is that our model architecture will insert a separate prediction head on top of each transformer block and make a prediction. We also add a confidence indicator at each prediction head that will be used to calculate a confidence score of stopping the model at the current prediction head. The confidence threshold of stopping will be a hyperparameter.

When we created this model, we first fine-tuned the model weights to suit the dataset and the new model structure, then we added and trained the confidence weights to create a confidence indicator.

### Confidence
When each layer predicts the class of an image, we want to know how confident the layer is in its predictions. We therefore add and train a confidence indicator to each layer of the model, where we freeze all parameters of the model beside the confidence indicator.    
The confidence indicator per layer is simply calculated by a linear transformation of the prediction and calculating the softmax scores:
```math
\text{Confidence}_i = \text{softmax}(\text{linear layer}(\text{output\_prediction}_i))
```

For layer i, we evaluate the confidence level according to the final prediction of the model.    
Therefore, we use the prediction of the last layer as our confidence standard (“ground truth”). By using the model's final prediction as a confidence standard, we will see how confident each earlier layer is at predicting the same as the final layer, even if the final layer is wrong.    
The early confidence stopping is therefore not about stopping when predicting correctly in terms of the gold labels, but to stop early if the model is confident that the early layer prediction is the same as the last layer. So, if the model is confident, it would not make sense to send images further in the model since the result would end up being the same.    
The main goal is to see whether it is possible to apply early stopping by having a sufficiently strong confidence indicator.    
If the confidence is above a certain threshold at an early layer, we will assume that the layer will predict the same as the final layer with a confidence of at least C (threshold), and stop inference at that layer.


### Results

Our results are from a comprehensive data analysis using our chosen model trained with a confidence indicator. The following plots are the distributions of early exit layers for six different thresholds: 0.5, 0.7, 0.8, 0.9, 0.95 and 0.99.    
Above each bar is the percentage of how often the exited layer predicted the same class as the last layer. This shows how well the early confidence stopping did in terms of similarity to the predictions of the last layer. Additionally, the average exit layer and the overall similarity to the last layer as well as the current threshold is stated above each distribution.

<figure>
    <img width="235" alt="0.5" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/137831b9-0748-4351-967d-c0c887963d4b">
</figure>

<figure>
    <img width="235" alt="0.7" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/54d78f6d-5e27-466f-af36-0e7894172dbd">
</figure>

We clearly see a shift in the distribution towards the right (later layers) as we increase the threshold. Additionally, the similarity with the last layer and average exit layer increases as we increase the confidence threshold, which is what we were looking for.

<figure>
    <img width="235" alt="0.8" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/eff9d52e-f7f7-4ca9-862b-29416f10a190">
</figure>

<figure>
    <img width="235" alt="0.9" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/ce5db17d-60a1-4818-9a55-a232c96a2575">
</figure>

The results are not satisfactory in terms of similarity to the last layer for any confidence thresholds below 0.95, which gets a 90% similarity to the last layer, since we are looking for a reduction in inference time without compromising performance.

<figure>
    <img width="235" alt="0.95" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/4496a6ec-9562-4ea2-b845-46d3171af0ff">
</figure>

<figure>
    <img width="235" alt="0.99" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/ca89185a-8ae4-44a1-b70f-fb02b77f8e2b">
</figure>

Even with such a high confidence threshold, the model performs well in terms of early stopping. With a confidence threshold of 0.95, we get an average exit layer of 7.73 and a similarity to the last layer of 90%, which almost halves the number of layers the images have to pass through to get a prediction, while making 90% of the same predictions. This seems like a decent time/accuracy tradeoff.
The distribution with a confidence threshold of 0.99 has even better results. 97% similarity to the last layer and an average exit layer of 8.84, which means over 25% reduction in the total number of layers the images have to pass through. Additionally, with a 0.99 confidence threshold over half of the images exit early, which is great in terms of time-accuracy tradeoff.

<figure>
    <img width="235" alt="0.999" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/f685afb7-5888-41d1-876f-f3df6ee06f1d">
</figure>

<figure>
    <img width="235" alt="0.9999" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/4f9404d1-ae76-44c2-8287-411edfb9daa5">
</figure>

<figure>
    <img width="235" alt="1" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/9c34dc1d-4acd-4ad7-9c2c-924c581782de"> 
</figure>

Pushing the threshold even further, with extreme confidence thresholds of 0.999, 0.9999 and 1.0. We see that we actually benefit from it in terms of time and early stopping. Using 0.999 as confidence threshold gives almost the exact same predictions as the last layer, while having an average exit layer of 10.06, which means saved inference time while not compromising accuracy. With such extreme confidence thresholds, some images still exit at early layers like layer 3 and 4.

The same is true for even more extreme confidence thresholds like 0.9999 and 1.0. The difference is that less images exit early, so the average exit layer increases, while the similarity to the last layer gets slightly better. Even with a confidence threshold of 1.0, which technically means the model is perfectly confident in predicting the same as the last layer, the model exits early on some images. In fact, some images exit as early as layer 6 with perfect confidence and perfect similarity to the last layer.


### Code

In this repository, you can also find our code. 

To run the demo version of our project go to our group project folder on fox:

    /fp/projects01/ec232/g05/g05-p3

Once you are here, all you need to do is run:

    make

This will do the following:

    - Load the Caltech256 data from fox
    - Load all models needed to run a demonstration of how confidence stopping work
    - Pick 10 random images and run them through:
            - The confidence model with confidence threshold 0.95 for early stopping
            - The same model without confidence (for comparison)
    - For each image, the demonstration will print:
            - At what layer the confidence model stopped
            - Confidence of predicting the same as the last layer (original model) on a 0.0 - 1.0 scale
            - Time saved by stopping early using the confidence threshold
            - Early confidence stopping model prediction
            - Original model (last layer) prediction
            - True label

## Dataset
As Dataset we used Caltech256. To make the code work, you'd have to download that dataset.
