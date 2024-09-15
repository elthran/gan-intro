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
  
   Below is an example of one of the histograms showing the distribution of pixel values for the Monet paintings:
   
   ![Pixel Distribution](path-to-your-histogram.png)

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

Below are some samples of generated images after training for 100 epochs:

![Generated Image 1](path-to-generated-image-1.png)
![Generated Image 2](path-to-generated-image-2.png)

A table summarizing the discriminator and generator losses across different architectures and hyperparameter configurations:

| Model           | Epochs | Discriminator Loss | Generator Loss | Comments                            |
|-----------------|--------|--------------------|----------------|-------------------------------------|
| Initial Model   | 100    | 0.45               | 0.75           | Initial training run                |
| Tuned Model 1   | 100    | 0.30               | 0.65           | Improved batch size and learning rate |
| Tuned Model 2   | 100    | 0.25               | 0.60           | Best result with hyperparameter tuning |

## Analysis and Lessons Learned

- **Mode Collapse**: During early experimentation, I observed mode collapse where the generator produced only a few types of images. This issue was mitigated by adjusting the learning rate and using a larger batch size.
- **Batch Normalization**: Including batch normalization in the generator significantly improved the quality of generated images.
- **Generator Loss Interpretation**: Lower generator loss did not always correlate with better image quality. Therefore, I relied on visual inspection of generated images to gauge progress.
- **Future Improvements**: Further experimentation could be done by:
  - Adding more layers to both the generator and discriminator.
  - Trying different GAN variants like **WGAN** or **CycleGAN**.
  - Training with a higher resolution dataset or progressively increasing resolution during training.

## Conclusion

This project provided a deep dive into generative modeling using GANs. The final model was able to generate 7,000 Monet-style images, and with hyperparameter tuning, the quality of these images was improved over time. Although the GAN architecture used in this project achieved decent results, there is still room for optimization and experimentation with alternative architectures and techniques.

