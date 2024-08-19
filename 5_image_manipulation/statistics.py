import numpy as np
import seaborn as sns
import glob
from PIL import Image
import matplotlib.pyplot as plt
from utils import check_results

def calculate_mean_std(image_list):
    """
    calculate mean and std of image list
    args:
    - image_list [list[str]]: list of image paths
    returns:
    - mean [array]: 1x3 array of float, channel wise mean
    - std [array]: 1x3 array of float, channel wise std
    """
    mean = []
    std = []
    for path in image_list:
        img = np.array(Image.open(path).convert('RGB'))
        R, G, B = img[..., 0], img[..., 1], img[..., 2]
        mean.append(np.array([np.mean(R), np.mean(G), np.mean(B)]))
        std.append(np.array([np.std(R), np.std(G), np.std(B)]))
    
    total_mean = np.mean(mean, axis = 0)
    total_std = np.mean(std, axis = 0)
    return total_mean, total_std


def channel_histogram(image_list):
    """
    calculate channel wise pixel value
    args:
    - image_list [list[str]]: list of image paths
    """
    red = []
    green = []
    blue = []                    
    for path in image_list:
        img = np.array(Image.open(path).convert('RGB'))
        R, G, B = img[..., 0], img[..., 1], img[..., 2]
        red.extend(R.flatten())            
        green.extend(G.flatten())                                
        blue.extend(B.flatten())            

    plt.figure()
    sns.kdeplot(red, color='r')
    sns.kdeplot(green, color='g')
    sns.kdeplot(blue, color='b')
    plt.show()             
                    
if __name__ == "__main__": 
    image_list = glob.glob('data/images/*')
    mean, std = calculate_mean_std(image_list)
    check_results(mean, std)
    channel_histogram(image_list[:2])
    