import os
from PIL import Image
from count_jpgs_in_directory import count_jpgs_in_directory

'''
This script processes images in a specified directory, padding them to square dimensions and resizing them to 224x224 pixels.
'''

def pad_to_square(image):
    width, height = image.size
    if width == height:
        return image
    
    max_side = max(width, height)
    new_image = Image.new("RGB", (max_side, max_side), (0, 0, 0))  # Create a black canvas
    
    # Paste the original image onto the canvas
    left = (max_side - width) // 2
    top = (max_side - height) // 2
    new_image.paste(image, (left, top))
    
    return new_image

def process_images(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(subdir, file)
                try:
                    with Image.open(file_path) as img:
                        # Pad the image to square
                        img = pad_to_square(img)
                        
                        # Resize image to 224x224
                        img = img.resize((224, 224))
                        
                        # Save the image (you can change the directory or filename as needed)
                        img.save(file_path)
                        print(f"Processed and saved: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Directory to scan for images
directory = "./ColHis-IDS_restructured_244"

process_images(directory)
print(count_jpgs_in_directory(directory))
