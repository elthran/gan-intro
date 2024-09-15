# GAN Getting Started - Image Generation with Generative Adversarial Networks

## Overview

This project is part of the Kaggle competition **"GAN Getting Started"**, which involves generating high-quality images using **Generative Adversarial Networks (GANs)**. The challenge focuses on understanding the process of using GANs for image generation tasks, specifically creating artistic images inspired by Monet's paintings.

GANs are a type of deep learning model consisting of two neural networks: a **generator** and a **discriminator**. The generator creates fake images, while the discriminator attempts to distinguish between real and fake images. Through iterative training, the generator improves its ability to create realistic images, while the discriminator sharpens its classification accuracy.

### Dataset

The dataset includes:
- 300 Monet paintings (`monet_jpg`)
- 7,028 photos (`photo_jpg`)
  
Each image is 256x256 pixels in size, and the challenge requires generating a large set of fake images that emulate Monet’s style. The goal is to produce between 7,000 to 10,000 high-quality images using the GAN model.

## Exploratory Data Analysis (EDA)

Before training the model, I performed some basic **EDA** to understand the dataset:
1. **Image Sizes**: All images are 256x256 pixels. Therefore, resizing isn't needed.
2. **Color Distribution**: Monet paintings typically have a distinct color palette compared to regular photos. This insight can be useful for training a GAN model that captures artistic style.
3. **Visualizations**: 
   - Histograms of pixel intensity values across Monet paintings and regular photos.
   - Comparisons of the RGB channels for photos and Monet paintings.

### Data Cleaning

No major cleaning steps were required, as the dataset was already well-structured and labeled. However, for memory efficiency during training, images were resized to 128x128 pixels initially and later scaled back to 256x256 for the output. Normalization was applied to rescale pixel values to the range [-1, 1].

## Model Architecture

### Generator

The generator takes a 100-dimensional noise vector as input and produces a 3-channel image of size 128x128. The architecture uses several transposed convolutional layers to progressively upsample the noise vector into a full-sized image. Batch normalization and ReLU activations are used between layers to stabilize the training process and enhance image quality.

**Generator Architecture:**
- Input: 100-dimensional noise vector
- Linear layer to map the noise into a 128x8x8 feature map
- 4 transposed convolutional layers with increasing output size: 16x16, 32x32, 64x64, and 128x128
- Tanh activation at the output layer to scale pixel values to the range [-1, 1]

### Discriminator

The discriminator is a standard convolutional neural network (CNN) that takes an image as input and outputs a single scalar value representing the probability that the image is real. It uses leaky ReLU activations and progressively downsamples the input image through a series of convolutional layers.

**Discriminator Architecture:**
- Input: 128x128 image
- 3 convolutional layers, each followed by leaky ReLU and downsampling
- Flatten the feature map into a single scalar output using a linear layer and sigmoid activation

The reasoning behind this architecture is that GANs perform well when the generator and discriminator have symmetric architectures in terms of their respective upsampling and downsampling layers. This design should help the generator learn how to produce realistic Monet-style images that can successfully fool the discriminator.

## Model Training

The model was trained using the **BCELoss** (Binary Cross-Entropy Loss) for both the generator and discriminator. Optimizers used were **Adam** with learning rates of 0.0002 and betas of (0.5, 0.999), which are commonly used for GAN training to stabilize the learning process.

Training was conducted for 100 epochs with the following procedure:
1. The discriminator was trained to differentiate between real Monet paintings and generated images.
2. The generator was trained to fool the discriminator by producing images that resemble Monet’s style.

## Hyperparameter Tuning

During the experiment, I ran several rounds of hyperparameter tuning to improve the quality of generated images:
1. **Batch Size**: Tried batch sizes of 32, 64, and 100. The best results were achieved with a batch size of 100, which provided a good balance between training speed and stability.
2. **Learning Rate**: Experimented with learning rates from 0.0001 to 0.0003. The optimal learning rate was found to be 0.0002, as a higher rate caused instability and mode collapse, while a lower rate led to slower convergence.
3. **Activation Functions**: Leaky ReLU with a slope of 0.2 was used in the discriminator to improve its ability to learn, while ReLU was used in the generator.
4. **Normalization**: Using batch normalization in the generator helped stabilize training and produced sharper images.

### Results

# Hyperparameter Tuning and Training Summary

In this section, I experimented with various combinations of batch sizes and learning rates to optimize the performance of the GAN. The goal was to achieve the lowest possible discriminator and generator loss, which would indicate better performance of the model in generating realistic images.
Hyperparameter Combinations Tested:

    Batch Sizes: 32, 64, 100
    Learning Rates: 0.0001, 0.0002, 0.0003
    Epochs: 10 for each combination

For each configuration, the generator and discriminator were trained and evaluated on their respective losses after each epoch. The generator loss measures how well it fools the discriminator, while the discriminator loss reflects its ability to distinguish between real and generated images.
Results Overview:

| Batch Size | Learning Rate | Generator Loss | Discriminator Loss |
|------------|---------------|----------------|--------------------|
| 32         | 0.0001        | 19.7705        | 1.9051             |
| 32         | 0.0002        | 25.0121        | 1.3033             |
| 32         | 0.0003        | 31.3805        | 0.9867             |
| 64         | 0.0001        | 14.8117        | 3.0773             |
| 64         | 0.0002        | 19.6375        | 2.0161             |
| 64         | 0.0003        | 22.8357        | 1.7054             |
| 100        | 0.0001        | 14.7467        | 2.9294             |
| 100        | 0.0002        | 19.6006        | 2.1460             |
| 100        | 0.0003        | 22.0088        | 1.7817             |

## Key Observations:

Learning Rate Impact:
    Across all batch sizes, increasing the learning rate generally led to a decrease in discriminator loss, indicating that the discriminator became better at distinguishing real from generated images.
    However, the generator loss increased with higher learning rates, which means the generator struggled more to fool the discriminator.

Batch Size Impact:
    A batch size of 100 with a learning rate of 0.0001 yielded the lowest overall generator loss (14.74) and discriminator loss (2.92). This suggests that a moderate batch size allowed for more stable training, balancing between the discriminator and generator performance.
    Batch size 32 with a learning rate of 0.0003 resulted in the lowest discriminator loss (0.98), but it also had the highest generator loss (31.38), which suggests that the discriminator was overly strong, and the generator could not catch up effectively.

## Best Performing Configuration:

The best configuration in terms of a balance between generator and discriminator loss was:

    Batch Size: 100
    Learning Rate: 0.0001
    This configuration resulted in a generator loss of 14.74 and a discriminator loss of 2.92, indicating a stable balance between the two networks.

## Next Steps:

For final image generation, the best-performing hyperparameter configuration (Batch Size: 100, Learning Rate: 0.0001) will be used. This combination ensures that the model achieves good performance without the generator or discriminator overpowering each other, which is crucial for producing high-quality images.

## Conclusion

This project provided a deep dive into generative modeling using GANs. The final model was able to generate 7,000 Monet-style images, and with hyperparameter tuning, the quality of these images was improved over time. Although the GAN architecture used in this project achieved decent results, there is still room for optimization and experimentation with alternative architectures and techniques.

