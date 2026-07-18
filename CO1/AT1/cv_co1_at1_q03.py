# ==========================================================
# Image Aliasing and Quantization Demonstration
# Google Colab
# ==========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Upload image
uploaded = files.upload()

image_name = list(uploaded.keys())[0]

# Read image
img = cv2.imread(image_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ----------------------------------------------------------
# 1. Original Image
# ----------------------------------------------------------

# ----------------------------------------------------------
# 2. Aliasing (Low Sampling)
# Downsample heavily and enlarge again
# ----------------------------------------------------------

small = cv2.resize(img, (img.shape[1]//8, img.shape[0]//8),
                   interpolation=cv2.INTER_NEAREST)

aliased = cv2.resize(small,
                     (img.shape[1], img.shape[0]),
                     interpolation=cv2.INTER_NEAREST)

# ----------------------------------------------------------
# 3. Quantization
# Reduce image to only 16 intensity levels
# ----------------------------------------------------------

levels = 16

quantized = np.floor(img / (256/levels)) * (256/levels)
quantized = quantized.astype(np.uint8)

# ----------------------------------------------------------
# 4. Corrective Approach
# Anti-aliasing using Gaussian Blur before Downsampling
# ----------------------------------------------------------

blurred = cv2.GaussianBlur(img, (5,5), 0)

small_corrected = cv2.resize(
    blurred,
    (img.shape[1]//8, img.shape[0]//8),
    interpolation=cv2.INTER_AREA
)

corrected = cv2.resize(
    small_corrected,
    (img.shape[1], img.shape[0]),
    interpolation=cv2.INTER_CUBIC
)

# ----------------------------------------------------------
# Display Results
# ----------------------------------------------------------

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(aliased)
plt.title("Aliasing (Low Sampling)")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(quantized)
plt.title("Quantization (16 Levels)")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(corrected)
plt.title("Corrected (Anti-Aliasing)")
plt.axis("off")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Observation
# ----------------------------------------------------------

print("\nCAUSE OF ALIASING")
print("- Image was sampled at a very low resolution.")
print("- High-frequency details are lost, producing jagged edges.")

print("\nQUANTIZATION EFFECT")
print("- Pixel intensity values are reduced to only 16 levels.")
print("- This causes visible banding and loss of smooth gradients.")

print("\nCORRECTIVE APPROACH")
print("- Apply Gaussian Blur before downsampling.")
print("- Use INTER_AREA for shrinking.")
print("- Use INTER_CUBIC when enlarging.")
print("- This reduces aliasing artifacts.")