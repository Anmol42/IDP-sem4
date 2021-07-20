from skimage.io import imread_collection
from skimage.color import rgb2lab,lab2rgb
from skimage.transform import resize

def get_img_data(path):
    train_ds = imread_collection(path)
    images = torch.zeros(len(train_ds),3,128,128)
    for i,im in enumerate(train_ds):
        im = resize(im, (128,128,3),
                           anti_aliasing=True)
        image = rgb2lab(im)
        image = torch.Tensor(image)
        image = image.permute(2,0,1)
        images[i]=image
    
    return images

def normalize_data(data):
    data[:,0] = data[0]/100
    data[:,1:] = data[1:]/110
    return data

