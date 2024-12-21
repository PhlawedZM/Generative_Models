import os
import torch
import PIL.Image as image
import torch.utils.data as data

r"""
    This returns a dataset used for Style Transfer GANs in the theory of cGAN's.
    Idea behind it is to get your input images and your style you want to turn into images as a dataset.
    
    input_folder(path): Folder containing images that you want to change the style of.
    
    stylized_folder(path): Folder containing images of the style you want.
    
    transform(torchvision.transform): Ideally you want to use ToTensor(), and some way of Normalizing.
"""


class CGANImageFolder(data.Dataset):
    def __init__(self, input_folder, stylized_folder, transform=None):
        self.input_folder = input_folder
        self.stylized_folder = stylized_folder
        self.transform = transform

        self.input_images = os.listdir(input_folder)
        self.stylized_images = os.listdir(stylized_folder)

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_folder, self.input_images[idx])
        stylized_image_path = os.path.join(self.stylized_folder, self.stylized_images[idx])

        input_image = image.open(input_image_path).convert('RGB')
        stylized_image = image.open(stylized_image_path).convert('RGB')

        if self.transform:
            input_image = self.transform(input_image)
            stylized_image = self.transform(stylized_image)

        return input_image, stylized_image


def padded_collate_fn(batch):
    width = 1
    height = 2

    padded_images = []
    for input_img, stylized_img in batch:

        max_width = max(input_img.size(width), stylized_img.size(width))
        max_height = max(input_img.size(height), stylized_img.size(height))
        max_size = max(max_width, max_height)

        padding_input = torch.zeros(3, max_size, max_size)
        padding_stylized = torch.zeros(3, max_size, max_size)

        # Getting the center
        input_start_height = int((max_size - input_img.size(width)) / 2)
        input_start_width = int((max_size - input_img.size(height)) / 2)
        stylized_start_height = int((max_size - stylized_img.size(width)) / 2)
        stylized_start_width = int((max_size - stylized_img.size(height)) / 2)

        # Merging the tensor using slices
        padding_input[:, input_start_height:input_start_height + input_img.size(width), input_start_width:input_start_width + input_img.size(height)] = input_img
        padding_stylized[:, stylized_start_height:stylized_start_height + stylized_img.size(width), stylized_start_width:stylized_start_width + stylized_img.size(height)] = stylized_img

        padded_images.append((padding_input, padding_stylized))

    return padded_images
