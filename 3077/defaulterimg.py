from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. List Your Image Filenames ---
# IMPORTANT: Change the file extension (.png, .jpg, etc.) to match your files.
image_filenames = [
    'img1.png',
    'img2.png',
    'img3.png',
    'img4.png'
]

# This list will store your loaded image data
loaded_images = []

print("⚙️  Starting to load images...")

# --- 2. Loop Through Filenames and Load Images ---
for filename in image_filenames:
    try:
        if os.path.exists(filename):
            # Open the image file
            img = Image.open(filename)
            
            # --- FIX STARTS HERE ---
            
            # 1. Convert image to a standard color format (RGB)
            img = img.convert('RGB')
            
            # 2. Resize the image to a uniform size (e.g., 128x128 pixels)
            # You can change this size if you need to.
            img = img.resize((128, 128))
            
            # --- FIX ENDS HERE ---

            # Convert the standardized image to a numerical array
            img_array = np.array(img)
            
            # Add the image data to our list
            loaded_images.append(img_array)
            print(f"✅ Successfully loaded and resized '{filename}'")
        else:
            print(f"❌ Error: Could not find '{filename}'. Please check the upload and filename.")
            
    except Exception as e:
        print(f"⁉️  Could not process '{filename}'. Error: {e}")

# This line should now work without error!
loaded_images = np.array(loaded_images)

print(f"\n✅ All images loaded. Total images: {len(loaded_images)}")
print(f"Data shape: {loaded_images.shape}") # e.g., (4, 128, 128, 3) for 4 images


# --- 3. Display the Loaded Images to Verify ---
if len(loaded_images) > 0:
    print("\nDisplaying loaded images:")
    fig, axes = plt.subplots(1, len(loaded_images), figsize=(15, 5))
    
    for i, img_array in enumerate(loaded_images):
        ax = axes[i]
        ax.imshow(img_array)
        ax.set_title(image_filenames[i])
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()