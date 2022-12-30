import os
from torchvision.io import read_image

my_dir = r"C:\Users\Administrator\Desktop\fx-mnist\data\caltech101\101_ObjectCategories"

for cat_dir_name in os.listdir(my_dir):
    cat_dir = os.path.join(my_dir, cat_dir_name)
    for image_name in os.listdir(cat_dir):
        image_path = os.path.join(cat_dir, image_name)
        img = read_image(image_path)
        if img.shape[0] != 3:
            os.remove(image_path)

print("done")