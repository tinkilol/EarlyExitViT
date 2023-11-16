# EarlyExitViT
Project by Cornelius, Amir, and Katinka.

## Project Description
This project aims to significantly increase inference speed by introducing stacked Vision Transformers (ViTs) with early exiting capabilities based on classification confidence.

## Further Results
Further results are displayed below: 
![for_poster2](https://github.com/tinkilol/EarlyExitViT/assets/116383349/72bbad08-2ee8-4e3a-9443-4b2e69ed5b01)
![for_poster1](https://github.com/tinkilol/EarlyExitViT/assets/116383349/ee87a95a-0133-4110-a3a5-4a00eb27a99e)
![for_poster](https://github.com/tinkilol/EarlyExitViT/assets/116383349/e33a0831-3898-478b-bf48-eea5656441ee)
![for_poster9](https://github.com/tinkilol/EarlyExitViT/assets/116383349/f5b68a57-9be8-40ed-8157-203305761768)
![for_poster8](https://github.com/tinkilol/EarlyExitViT/assets/116383349/f4a9551b-39b0-41f0-aae7-eb0e88697695)
![for_poster7](https://github.com/tinkilol/EarlyExitViT/assets/116383349/53d8391a-8f9f-44b5-a813-0a5800fd6ec2)
![for_poster6](https://github.com/tinkilol/EarlyExitViT/assets/116383349/7b782f19-0d22-431e-9d31-c68bc03b373b)
![for_poster5](https://github.com/tinkilol/EarlyExitViT/assets/116383349/d9498e6b-d4f4-475b-8b65-1cce1f91f679)
![for_poster4](https://github.com/tinkilol/EarlyExitViT/assets/116383349/c055cf38-a569-45ba-97ee-4647d62c8e56)
![for_poster3](https://github.com/tinkilol/EarlyExitViT/assets/116383349/6d663b91-d2e8-443e-9a5b-2673ca59c7d6)

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
