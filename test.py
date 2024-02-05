from PIL import Image
import io
import random

# Define the tuple with image path, rect, and tile flags
image_data = ('./world_builder_data/projects/testproject/map_metadata/3/./tilesets/../../../../../assets_for_world_builder/MiniWorldSprites/Ground/Shore.png', 
              (48, 0, 16, 16), 
              {'flipped_horizontally': 0, 'flipped_vertically': 0, 'flipped_diagonally': 0})

# Extract path and rect from the tuple
image_path, rect, flags = image_data

# Open the image
image = Image.open(image_path)

# The rect is in the form (x, y, width, height), but PIL's crop method expects (left, upper, right, lower)
# So, we need to convert it
crop_box = (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])

# Crop the image
cropped_image = image.crop(crop_box)

# Create an in-memory buffer
buffer = io.BytesIO()

# Save the cropped image to the buffer in PNG format
cropped_image.save(buffer, format='PNG')

# Display the cropped image (this will only work in environments that support image display, like Jupyter notebooks)
buffer.seek(0)
display_image = Image.open(buffer)

print(f"cropped image width: {cropped_image.width}")
print(f"cropped image height: {cropped_image.height}")

display_image.show()

# Assuming the cropped_image is already created from the previous step

# Define the size of the new buffer based on the tile size (16x16 pixels) and the number of tiles (15x10)
buffer_width_tiles = 15  # 15 tiles wide
buffer_height_tiles = 10  # 10 tiles high
tile_width = 16
tile_height = 16

# Calculate the pixel dimensions of the buffer
buffer_width_px = buffer_width_tiles * tile_width
buffer_height_px = buffer_height_tiles * tile_height

# Create a new white image for the buffer
new_image_buffer = Image.new('RGB', (buffer_width_px, buffer_height_px), 'white')

# We'll now paste the cropped image onto the buffer at random cell positions
# Let's do this 5 times for demonstration, ensuring unique cell locations for each paste
occupied_cells = set()

for _ in range(50):
    while True:
        # Randomly select a cell
        cell_x = random.randint(0, buffer_width_tiles - 1)
        cell_y = random.randint(0, buffer_height_tiles - 1)
        
        # Check if the cell is already occupied
        if (cell_x, cell_y) not in occupied_cells:
            occupied_cells.add((cell_x, cell_y))
            break
    
    # Calculate the pixel position based on the cell position
    px_x = cell_x * tile_width
    px_y = cell_y * tile_height

    # Paste the cropped image onto the buffer at the calculated position
    new_image_buffer.paste(cropped_image, (px_x, px_y))

# Display the resulting image buffer
new_image_buffer.show()

