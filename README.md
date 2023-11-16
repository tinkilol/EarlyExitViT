# EarlyExitViT
Project by Cornelius, Amir, and Katinka.

## Project Description
This project aims to significantly increase inference speed by introducing stacked Vision Transformers (ViTs) with early exiting capabilities based on classification confidence.

## Further Results
For our choice of the model, we compared different batch sizes, confidence trainings (f.e. hinge), learning rates, weight decay and epochs. The results (amount of images that exit at certain layer) are shown below. 

To explain the parameters: df_128_hinge_lr001_wd02_50 would mean that the model uses batch size = 128, hinge embedding loss, learning rate = 0.01, weight decay = 0.2, and epochs = 50. The standard parameters are batch size = 64, cross-entropy loss, learning rate = 0.001, weight decay = 0.1, epochs = 10. If the parameter isn't given in the name, it is standard. 
<!-- First set of images -->
<figure>
    <img width="235" alt="df_bs128" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/f2de312d-05ab-45d5-b84f-15cbf4ee3d8a">
</figure>

<figure>
    <img width="235" alt="df_bs64_wd02" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/8e209c74-8b05-470a-8e4c-d5a6016ef19c">
</figure>

<figure>
    <img width="235" alt="df_bs64_lr001_epochs50" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/c9bba2af-551f-416d-9f1c-6981de46734c">
</figure>

<figure>
    <img width="235" alt="df_bs64_lr001_epochs10" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/107dbd99-097a-4a87-891d-3265a596494b">
</figure>

<figure>
    <img width="235" alt="df_bs64_hinge_epochs50" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/a63f4564-b7b2-44d6-afc7-1689ef51633c">
</figure>

<figure>
    <img width="235" alt="df_bs64_hinge" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/9f916ad1-e67d-4fa2-9eb1-ae6b23c27bbc">
</figure>

<figure>
    <img width="235" alt="df_bs64_epochs50" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/96aa65b3-272a-4843-9b0a-d6cb1e71a14c">
</figure>

<figure>
    <img width="235" alt="df_norm" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/46214170-7780-42b1-93da-901ad214dccc">
</figure>

<figure>
    <img width="235" alt="df_bs128_hinge_epochs50" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/f318d82d-fd3d-44bb-8544-58ed1dcca2d5">
</figure>

<figure>
    <img width="235" alt="df_bs128_hinge" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/e8f56005-d744-4163-ae9f-8235fc746b04">
</figure>

<figure>
    <img width="235" alt="df_bs128_epochs50" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/3479d30a-505c-413d-9e50-3995e652b898">
</figure>



<!-- Second set of images -->
<figure>
    <img width="235" alt="df_bs128_threshold_0_5" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/137831b9-0748-4351-967d-c0c887963d4b">
    <figcaption>df_bs128_threshold_0_5</figcaption>
</figure>

<figure>
    <img width="235" alt="df_bs128_threshold_0_7png" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/54d78f6d-5e27-466f-af36-0e7894172dbd">
    <figcaption>df_bs128_threshold_0_7png</figcaption>
</figure>

<figure>
    <img width="235" alt="df_bs128_threshold_0 8" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/eff9d52e-f7f7-4ca9-862b-29416f10a190">
    <figcaption>df_bs128_threshold_0_7png</figcaption>
</figure>

<figure>
    <img width="235" alt="df_bs128_threshold_0 9" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/ce5db17d-60a1-4818-9a55-a232c96a2575">
    <figcaption>df_bs128_threshold_0_7png</figcaption>
</figure>

<figure>
    <img width="235" alt="df_bs128_threshold_0 95" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/4496a6ec-9562-4ea2-b845-46d3171af0ff">
    <figcaption>df_bs128_threshold_0_7png</figcaption>
</figure>

<figure>
    <img width="235" alt="df_bs128_threshold_0 99" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/ca89185a-8ae4-44a1-b70f-fb02b77f8e2b">
    <figcaption>df_bs128_threshold_0_7png</figcaption>
</figure>

<figure>
    <img width="235" alt="df_bs128_threshold_0 999" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/f685afb7-5888-41d1-876f-f3df6ee06f1d">
    <figcaption>df_bs128_threshold_0_7png</figcaption>
</figure>

<figure>
    <img width="235" alt="df_bs128_threshold_0 9999" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/4f9404d1-ae76-44c2-8287-411edfb9daa5">
    <figcaption>df_bs128_threshold_0_7png</figcaption>
</figure>

<figure>
    <img width="235" alt="df_bs128_threshold_1" src="https://github.com/tinkilol/EarlyExitViT/assets/116383349/9c34dc1d-4acd-4ad7-9c2c-924c581782de"> 
    <figcaption>df_bs128_threshold_0_7png</figcaption>
</figure>




## Code
To run our project go to our group project folder on fox:

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
