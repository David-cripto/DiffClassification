# DiffClassification

## Few Shot Generative Classification

Traditional supervised classification approaches limit the scalability and training efficiency of neural networks because they require significant human effort and computational resources to partition the data.

## Concept

![image](https://github.com/David-cripto/DiffClassification/assets/78556639/cbe5f13e-c6f2-4021-bf86-dca3c87d5d6c)

Given a scenario where we possess a small labeled dataset alongside a larger unlabeled dataset, we can approach classification through the following steps:

-Train a generative model on the unlabeled dataset to learn the underlying data distribution.
-Utilize the trained generative model to extract more representative features for the labeled images, effectively enriching the feature space.
-Train a small neural network using the enriched features to make predictions for the corresponding labels.
This approach leverages the generative model to enable the small neural network to make accurate predictions despite the limited labeled data.


## Optimal time selection for diffusion model
In the framework of the diffusion model for feature aggregation, the choice of the optimal diffusion time step parameter becomes paramount in determining the temporal influence of features.
![image](https://github.com/David-cripto/DiffClassification/assets/78556639/1d92b604-84c0-4f7c-a8e0-d17c369e553a)
![image](https://github.com/David-cripto/DiffClassification/assets/78556639/3f8367d7-c394-4303-936a-24f88582d970)

As evident from the plots, the optimal step values for MNIST and CIFAR-10 are $100$ and $50$, respectively. Consequently, we set these timesteps as constants for subsequent experiments.




