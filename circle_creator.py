from PIL import Image, ImageDraw
import os
import random

# Set the size of the image
image_size = (256, 256)


# Number of images to generate
num_images = 5000

for i in range(num_images):
    # Create a new image with white background
    img = Image.new('L', image_size, 'white')
    draw = ImageDraw.Draw(img)

    # Generate random coordinates for the circle's center
    x = random.randint(0, image_size[0])
    y = random.randint(0, image_size[1])
    # Generate a random radius
    radius = random.randint(1, 100)

    # Draw the circle
    left_up_point = (x-radius, y-radius)
    right_down_point = (x+radius, y+radius)
    draw.ellipse([left_up_point, right_down_point], outline='black', fill='black')

    # Save the image
    img.save(f'test_photos/circle_{i+1}.png')

print(f'{num_images} images saved in photos folder')
