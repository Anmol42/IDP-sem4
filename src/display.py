import torch
import generator 
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab,lab2rgb
def tensor_to_pic(tensor : torch.Tensor) -> np.ndarray:
    tensor[0] *=  100
    tensor[1:]*=  128
    image = tensor.permute(1,2,0).detach().cpu().numpy()
    image = lab2rgb(image)
    return image

def show_images(n,dataset = images,gen=gen_model,dis=dis_model) -> None:
  gen_model.eval()
  dis_model.eval()
  z = torch.randn(1,256).to('cuda')
  #z = torch.ones_like(z)
  image_tensor = dataset[n].to('cuda')
  gen_tensor = gen(image_tensor[0].unsqueeze(0).unsqueeze(0),z)[0]
  image = tensor_to_pic(image_tensor)
  #print(torch.sum(gen_tensor))
  gen_image = tensor_to_pic(gen_tensor)
  to_be_shown = np.concatenate((gen_image,image),axis=1)
  plt.figure(figsize=(9,9))
  plt.imshow(to_be_shown)
  plt.show()

