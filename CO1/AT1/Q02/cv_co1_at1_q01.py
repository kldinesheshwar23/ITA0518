# Downsample the image to a tiny 50x25 resolution to simulate a very low sampling rate (bad camera sensor)
low_res = cv2.resize(img, (50, 25), interpolation=cv2.INTER_NEAREST) # INTER_NEAREST simply drops pixels without blending
print("Low-Res Shape:", low_res.shape)         # The image is now only 25 pixels high and 50 pixels wide

# Scale the tiny image back up to 400x200 so we can actually see the blocky pixelation effect on screen
pixelated = cv2.resize(low_res, (400, 200), interpolation=cv2.INTER_NEAREST) # Upscale using Nearest Neighbor to keep it blocky

# Create a figure with a specific size (width=10, height=4 inches)
plt.figure(figsize=(10, 4))                                            # Create a visualization window to hold our images

# Create the left subplot (1 row, 2 columns, 1st position)
plt.subplot(1, 2, 1)                                                   # Position this image within a grid layout
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Display the original, high-quality image
plt.title('Original (High Sampling)')                                  # Set a descriptive title for this plot
plt.axis('off')                                                        # Hide axis ticks and coordinates

# Create the right subplot (1 row, 2 columns, 2nd position)
plt.subplot(1, 2, 2)                                                   # Position this image within a grid layout
plt.imshow(cv2.cvtColor(pixelated, cv2.COLOR_BGR2RGB)) # Display the blurred/blocky low-sampling image
plt.title('Pixelated (Low Sampling)')                                  # Set a descriptive title for this plot
plt.axis('off')                                                        # Hide axis ticks and coordinates
plt.show()                                     # Render both subplots to the screen
