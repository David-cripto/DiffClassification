# DiffClassification

## Few Shot Generative Classification

Traditional supervised classification approaches limit the scalability and training efficiency of neural networks because they require significant human effort and computational resources to partition the data.

## Concept

![image](https://github.com/David-cripto/DiffClassification/assets/78556639/cbe5f13e-c6f2-4021-bf86-dca3c87d5d6c)

Given a scenario where we possess a small labeled dataset alongside a larger unlabeled dataset, we can approach classification through the following steps:

- Train a generative model on the unlabeled dataset to learn the underlying data distribution.
- Utilize the trained generative model to extract more representative features for the labeled images, effectively enriching the feature space.
- Train a small neural network using the enriched features to make predictions for the corresponding labels.

This approach leverages the generative model to enable the small neural network to make accurate predictions despite the limited labeled data.


## Optimal time selection for diffusion model
In the framework of the diffusion model for feature aggregation, the choice of the optimal diffusion time step parameter becomes paramount in determining the temporal influence of features.
![image](https://github.com/David-cripto/DiffClassification/assets/78556639/1d92b604-84c0-4f7c-a8e0-d17c369e553a)
![image](https://github.com/David-cripto/DiffClassification/assets/78556639/3f8367d7-c394-4303-936a-24f88582d970)

As evident from the plots, the optimal step values for MNIST and CIFAR-10 are $100$ and $50$, respectively. Consequently, we set these timesteps as constants for subsequent experiments.

## Generation Quality

![image](https://github.com/David-cripto/DiffClassification/assets/78556639/2df88f3a-64eb-4017-97f3-5f75b50dcc58)


## MNIST Training Results

Training linear (Linear) and nonlinear (Linear+ReLU+Linear) models on features from generative models

![image](https://github.com/David-cripto/DiffClassification/assets/78556639/9397cc93-c248-461f-aace-6bbab676224d)

![image](https://github.com/David-cripto/DiffClassification/assets/78556639/6891d2ed-2740-4f14-ac96-df0d20d093c2)


## Models Comparison

![image](https://github.com/David-cripto/DiffClassification/assets/78556639/7af1b8f4-0df9-485e-b40a-d796c3ed97fb)

## CIFAR-10 Training Results

Training nonlinear (Linear+ReLU+Linear) model (since it was the best on MNIST) on features from generative models
![image](https://github.com/David-cripto/DiffClassification/assets/78556639/f77f8155-c96a-40db-8d07-d094e0d458dd)

## Models comparison

![image](https://github.com/David-cripto/DiffClassification/assets/78556639/8ea8a96e-5c4a-432f-8318-522985b39130)

## Scripts Usage

   ```bash
  extract_features_train_smallnet_cifar.py [-h] [-p PATH] [-d DEVICE] [-s SIZE_PER_CLASS] [-e EPOCHS]
options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path of weights
  -d DEVICE, --device DEVICE
                        Device for training
  -s SIZE_PER_CLASS, --size_per_class SIZE_PER_CLASS
                        Number of images per class
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
   ```
```bash
extract_features_train_smallnet_mnist.py [-h] [-p PATH] [-d DEVICE] [-s SIZE_PER_CLASS] [-e EPOCHS]

options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path of weights
  -d DEVICE, --device DEVICE
                        Device for training
  -s SIZE_PER_CLASS, --size_per_class SIZE_PER_CLASS
                        Number of images per class
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
```

 ```bash
train_vae_cifar.py [-h] [-d DEVICE] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LR] [-ld LATENT_DIM]

options:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        Device for training
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -lr LR, --lr LR       Learning rate
  -ld LATENT_DIM, --latent_dim LATENT_DIM
                        Laten space dimension
 ```

 ```bash
train_vae_mnist.py [-h] [-d DEVICE] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LR] [-ld LATENT_DIM]

options:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        Device for training
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -lr LR, --lr LR       Learning rate
  -ld LATENT_DIM, --latent_dim LATENT_DIM
                        Laten space dimension
 ```







