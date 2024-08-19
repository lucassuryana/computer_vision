from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def create_mask(path, color_threshold):
    """
    create a binary mask of an image using a color threshold
    args:
    - path [str]: path to image file
    - color_threshold [array]: 1x3 array of RGB value
    returns:
    - img [array]: RGB image array
    - mask [array]: binary array
    """
    # load the image
    img = np.array(Image.open(path).convert('RGB'))

    # create the mask by comparing the image pixles with color_threshold
    mask = np.all(img <= color_threshold, axis = -1)

    # convert the boolean mask into an integer mask (0 or 1)
    return img, mask


def mask_and_display(img, mask):
    """
    display 3 plots next to each other: image, mask and masked image
    args:
    - img [array]: HxWxC image array
    - mask [array]: HxW mask array
    """
    # create a copy of image to apply mask
    masked_image = np.copy(img)
    # apply the mask
    masked_image[mask == 1] = 0
    f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(24,9))
    ax1.imshow(img)
    ax1.set_title('Original image', fontsize = 30)
    ax2.imshow(mask)
    ax2.set_title('Mask', fontsize = 30)
    ax3.imshow(masked_image)
    ax3.set_title('Masked image', fontsize = 30)
    plt.show()


if __name__ == '__main__':
    path = 'data/images/segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    color_threshold = [128, 128, 128]
    img, mask = create_mask(path, color_threshold)
    mask_and_display(img, mask)