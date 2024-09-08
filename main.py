# Import necessary libraries
import os
from random import sample

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Check for device availability (GPU/CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", device)


# Dataset class to load Monet and photo images
class Images(Dataset):
    def __init__(self, photo_path, monet_path, transform):
        self.photo_path = photo_path
        self.monet_path = monet_path
        self.transform = transform
        self.photos = os.listdir(photo_path)
        self.monets = os.listdir(monet_path)

        self.photos = sample(self.photos, min(50, len(self.photos)))
        self.monets = sample(self.monets, min(50, len(self.monets)))

        self.l_photo = len(self.photos)
        self.l_monet = len(self.monets)

    def __len__(self):
        return max(len(self.photos), len(self.monets))

    def __getitem__(self, idx):
        # Load and transform images
        photo = Image.open(self.photo_path + self.photos[idx % self.l_photo]).convert("RGB")
        monet = Image.open(self.monet_path + self.monets[idx % self.l_monet]).convert("RGB")

        photo = self.transform(photo)
        monet = self.transform(monet)

        return photo, monet


# Image transformations
transform = transforms.Compose([transforms.Resize((128, 128)),  # Rescale to 128x128 to reduce memory consumption
                                transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# Define dataset and dataloader
photo_path = 'data/photo_jpg/'
monet_path = 'data/monet_jpg/'
dataset = Images(photo_path, monet_path, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(# Input is a 100-dimensional noise vector
            nn.Linear(100, 128 * 8 * 8),  # First layer to map noise to a feature map
            nn.ReLU(True), nn.Unflatten(1, (128, 8, 8)),

            # Upsample to 16x16
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),

            # Upsample to 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),

            # Upsample to 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),

            # Upsample to 128x128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), nn.Tanh()
            # Output range [-1, 1] to match normalized image data
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(  # Input is a 3-channel image (128x128)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(), nn.Linear(128 * 16 * 16, 1), nn.Sigmoid()  # Output between 0 (fake) and 1 (real)
        )

    def forward(self, x):
        return self.model(x)


# Loss and optimizers
adversarial_loss = nn.BCELoss()

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training the GAN
epochs = 100
for epoch in range(epochs):
    running_d_loss = 0.0
    running_g_loss = 0.0
    for i, (photos, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = torch.ones((photos.size(0), 1), requires_grad=False).to(device)
        fake = torch.zeros((photos.size(0), 1), requires_grad=False).to(device)

        # Real photos to be processed by the discriminator
        real_photos = photos.to(device)

        # Sample random noise for the generator
        noise = torch.randn((photos.size(0), 100)).to(device)

        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()

        # Generate fake photos
        generated_photos = generator(noise)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(generated_photos), valid)
        running_g_loss += g_loss.item()

        # Backward propagation and optimization
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_photos), valid)
        fake_loss = adversarial_loss(discriminator(generated_photos.detach()), fake)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        running_d_loss += d_loss.item()

        # Backward propagation and optimization
        d_loss.backward()
        optimizer_D.step()

    # Print progress at the end of each epoch
    print(f"Epoch [{epoch + 1}/{epochs}] | Discriminator loss: {running_d_loss / len(dataloader):.4f} | Generator loss:"
          f" {running_g_loss / len(dataloader):.4f}")

    # Save some generated images after every epoch (optional)
    if epoch % 10 == 0:
        noise = torch.randn(25, 100).to(device)
        gen_imgs = generator(noise)
        gen_imgs = gen_imgs.view(gen_imgs.size(0), 3, 128, 128)
        gen_imgs = gen_imgs.permute(0, 2, 3, 1).cpu().detach().numpy()

        fig, axs = plt.subplots(5, 5, figsize=(10, 10))
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(gen_imgs[i * 5 + j] * 0.5 + 0.5)  # Rescale to [0, 1]
                axs[i, j].axis('off')
        plt.show()

