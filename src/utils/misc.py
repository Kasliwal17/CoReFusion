import matplotlib.pyplot as plt
import os

# normalizing target image to be compatible with tanh activation function
def normalize_data(data):
    data *= 2
    data -= 1
    return data

def unnormalize_data(data):
    data += 1
    data /= 2
    return data

def preprocessing_fn(x):
        x=x/255
        x=x-0.485
        x=x/0.229
        return x

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def list_img(dir1):
    lst = []
    for root, dirs, files in os.walk(dir1):
        lst.extend(files)
    lst = sorted(lst)
    for x in range(len(lst)):
        lst[x]= dir1+ '/'+ lst[x]
    return lst
