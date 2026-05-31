# Face Image Inpainting with GAN

A deep learning project for **face image inpainting** using a **Generative Adversarial Network (GAN)** built with **TensorFlow/Keras**.  
The model learns to reconstruct the missing central region of a face image after a square mask is applied.

---

## Educational Purpose

This project was designed by me for students in the **Principles of Computer Vision** course as an introductory hands-on example of **image inpainting using GANs**.

The main goal of this project is to help students become familiar with:

- the basic idea of **Generative Adversarial Networks (GANs)**
- the interaction between a **Generator** and a **Discriminator**
- preprocessing images for an inpainting task
- defining and understanding adversarial loss functions
- applying deep learning concepts to a practical computer vision problem

This project is primarily intended as an **educational starting point**, not as a fully optimized or state-of-the-art inpainting system.  
The focus is on understanding the workflow and core concepts behind GAN-based reconstruction rather than achieving the best possible visual performance.

Although the current implementation demonstrates the essential structure of a GAN for face inpainting, there is still significant room for improvement.  
For example, the project could be extended by:

- using a more advanced generator architecture such as **U-Net**
- adding **reconstruction losses** such as L1 or perceptual loss
- improving training stability
- using more flexible or irregular masks
- evaluating results with metrics such as **PSNR** or **SSIM**
- training for longer and tuning hyperparameters more carefully

In summary, this project was created to provide students with an initial practical understanding of GANs in the context of computer vision, while also leaving space for further experimentation, discussion, and future improvement.


---

## Overview

Image inpainting is the task of restoring missing or damaged parts of an image in a visually realistic way.  
In this project, the center region of a face image is masked with a black square, and a GAN is trained to predict and reconstruct the missing content.

The notebook uses a CelebA-style face dataset and demonstrates the full workflow:

- downloading the dataset
- loading and visualizing images
- masking the center region
- preparing training/validation/test splits
- building generator and discriminator models
- defining adversarial loss functions
- training the GAN for reconstruction

---

## Project Idea

The main objective is to train a model that can infer missing facial details from surrounding context.

Given:

- an **input image with a missing center**
- the corresponding **original image**

the generator tries to reconstruct the full image, while the discriminator tries to determine whether an image is real or generated.

This adversarial setup encourages the generator to produce outputs that are not only close to the original image but also visually realistic.

---

## Dataset

This project uses the following Kaggle dataset:

- **Dataset name:** `kushsheth/face-vae`
- **Accessed via:** `kagglehub`
- **Image folder:** `img_align_celeba/img_align_celeba`

---
## Pipeline

The overall project pipeline is:

- Download the face dataset from Kaggle
- Load image files from the dataset directory
- Resize images to a fixed resolution
- Apply a square mask to the center of each image
- Build input-target pairs:
- Input: masked image
- Target: original image
- Split the dataset into training, validation, and test sets
- Train a GAN:
- Generator reconstructs missing content
- Discriminator distinguishes real from generated images
- Evaluate visual reconstruction quality

---

## Preprocessing
The preprocessing stage prepares data for training.

Steps
- Read images from the dataset directory
- Resize each image to 128 × 128
- Apply a center mask of fixed size
- Convert processed samples to NumPy arrays
- Input/Target Format

This makes the task a supervised image-to-image translation problem.

#### Masking Strategy
A black square mask is placed in the center of the image.

This removes a meaningful portion of the face and forces the model to reconstruct it using surrounding visual context.

---

## Model Architecture
The project uses a standard GAN setup with two networks:

- Generator
- Discriminator

The generator receives the masked image and predicts the missing content.
Its goal is to produce an output image that resembles the original unmasked face. 
This network learns contextual understanding of facial structure such as skin, nose, mouth, and surrounding textures.

The discriminator is a binary classifier that learns to distinguish between:

- real images from the dataset
- fake/generated images produced by the generator

Its output is a probability score indicating whether the input is real or generated.

The discriminator forces the generator to produce more realistic outputs over time.

 ---
 
 ## Loss Functions

This project trains a GAN for face inpainting using **Binary Cross-Entropy (BCE)** adversarial losses.

### Notation
- `x` : masked input image
- `y` : real (ground-truth) image
- `G(·)` : Generator
- `D(·)` : Discriminator output in `[0, 1]` (probability of being real)
- `E[·]` : expectation over the data distribution


---

### Binary Cross-Entropy (BCE)

**Formula (plain text):**

BCE(t, p) = - [ t * log(p) + (1 - t) * log(1 - p) ]

**Where:**
- `t` in `{0, 1}` is the target label
- `p` in `(0, 1)` is the predicted probability

