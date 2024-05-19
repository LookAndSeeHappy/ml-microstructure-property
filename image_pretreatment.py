from PIL import Image
import os

def cut_image(image,item_width,item_height,overlap):
    # item_width: the width of window images
    # item_height:the height of window images
    width, height = image.size
    num_width = int(1)
    num_height = int(1)
    if width != item_width:
        num_width = int((width - item_width)//(item_width - overlap) + 1)
    if height != item_height:
        num_height = int((height - item_height) // (item_height - overlap) + 1)
    box_list = []
    for i in range(num_width):
        for j in range(num_height):
            # (left, upper, right, lower)
            box = (i*(item_width-overlap),j*(item_height-overlap),i*(item_width-overlap)+item_width,j*(item_height-overlap)+item_height)
            box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    return image_list

def save_images(image_list,name):
    index = 1
    for image in image_list:
        # the path of document where you want to save images
        image.save('YOUR_PATH', 'PNG')
        index += 1

def geturlPath():
    # the path of images you want to process
    path = r'YOUR_PATH'
    dirs = os.listdir(path)
    for dir in dirs:
        pa = path+dir
        if not os.path.isdir(pa):
            yield pa,dir

def convert_images_to_3_channel(input_path, output_path):
    """
    Convert images in a directory to 3-channel RGB format and save them in another directory.

    Args:
    input_path (str): Path to the directory containing images to be converted.
    output_path (str): Path to the directory where converted images will be saved.
    """

    # List all files in the input directory
    all_images = os.listdir(input_path)

    # Iterate through each image in the directory
    for image in all_images:
        # Construct the full path to the image
        image_path = os.path.join(input_path, image)

        # Open the image using PIL
        img = Image.open(image_path)

        # Convert the image to RGB format (3 channels)
        img = img.convert("RGB")

        # Define the path to save the converted image
        save_path = output_path

        # Save the converted image
        img.save(os.path.join(save_path, image))
        
if __name__ == '__main__':
    for file_path,name in geturlPath():
        image = Image.open(file_path)
        # remove the text part of images to get raw images
        # image_list = cut_image(image, 1280, 960, 0)
        
        # acquire window image
        image_list = cut_image(image,340,340,170)
        save_images(image_list,name)
