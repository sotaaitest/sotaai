import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import mxnet as mx
from skimage import color
import tensorflow as tf


def ds_numpy2torch(X, y):
    ''' Transform a numpy array to a torch dataloader '''
    x_tensor = Tensor(X)
    y_tensor = Tensor(y)
    pt_dataset = TensorDataset(x_tensor, y_tensor)  # create your datset
    pt_dataloader = DataLoader(pt_dataset)  # create your dataloader
    return pt_dataloader


def ds_numpy2mxnet(X, y):
    X_mxnet = mx.nd.array(X)
    X_mxnet = mx.nd.transpose(X_mxnet, (0,3,1,2) )
    y_mxnet = mx.nd.array(y)
    mx_dataset = mx.gluon.data.ArrayDataset(X_mxnet, y_mxnet)
    return mx_dataset


def ds_torch2numpy(dataloader):
    X = dataloader.data
    y = np.asarray(dataloader.targets,dtype = np.int32)
    if len(X.shape)==3:
        X = X.reshape(*X.shape,1)
    if isinstance(X,Tensor):
        X = X.numpy()
    '''
    tmp = dataloader.data
    if isinstance(tmp[:][0],np.ndarray):
        X = tmp[:][0]
    else:
        X = tmp[:][0].numpy()
        '''
    return X, y


def ds_mxnet2numpy(dataloader):
    X = dataloader[:][0].asnumpy()
    X = X.transpose(0, 3, 1, 2)
    y = dataloader[:][1]
    return (X, y)


def ds_tf2numpy(dataloader):
    ds = list(dataloader.as_numpy_iterator())
    return ds


def ds_torch2mxnet(dataloader): 
    ''' Transform a torch dataloader to a mxnet dataloader '''
    X, y = ds_torch2numpy(dataloader)
    mx_dataset = ds_numpy2mxnet(X, y)
    return mx_dataset

def resize(numpy_data,img_size):
    #shape [batch,height,width,channels]
    batch=512
    imgs = tf.constant(numpy_data)
    n_batches = np.floor(numpy_data.shape[0]/512).astype(np.int8)
    print(n_batches)
    obt_batches = []
    for n in range(n_batches):
        start = n*batch
        end = batch*(n+1)
        resized = tf.image.resize(imgs[start:end], [img_size,img_size])
        obt_batches.append(resized.numpy())
    start = ( n_batches - 1 )*batch
    resized = tf.image.resize(imgs[start:], [img_size,img_size])
    obt_batches.append(resized.numpy())
    return np.concatenate(obt_batches)


def ds_mxnet2torch(dataloader):
    ''' Transform a mxnet dataloader to a torch dataloader. To access only dataset, run pt_dataloader.dataset '''
    X, y = ds_mxnet2numpy(dataloader)
    pt_dataloader = ds_numpy2torch(X, y)
    return pt_dataloader


def change_channels(dataset):
    '''
    Input: numpy array of shape (N,H,W,C)
    Output: numpy array of shape (N,H,W,C*)
    '''
    # Convert to numpy array
    if isinstance(dataset, mx.nd.NDArray):
        tmp_dataset = ds_mxnet2numpy(dataset)
    elif isinstance(dataset, torch.Tensor):
        tmp_dataset = ds_torch2numpy(dataset)
    elif isinstance(dataset, np.array):
        tmp_dataset = dataset
    else:
        type_ds = str(type(dataset))
        return print('Type '+type_ds+' not supported')

    img = color.rgb2gray(
        tmp_dataset) if dataset.shape[-1] == 3 else color.gray2rgb(dataset)

    # If dataset was not numpy, transform back to original type
    if isinstance(dataset, mx.nd.NDArray):
        img = ds_numpy2mxnet(dataset)
    elif isinstance(dataset, torch.Tensor):
        img = ds_numpy2torch(dataset)

    return img


def transform(image, resize: int = -1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    Input: image of mxnet NDArray type, shape (height, width, channel)
    Output: image of mxnet NDArray type, shape (1, channel, height, width)
    '''
    # images should be in range [0,1]
    # with shape (N x 3 x H x W) also needs to be dtype = float32

    img = image
    if resize != -1:
        img = mx.image.resize_short(image, resize)
        cropped, _ = mx.image.center_crop(img, (resize, resize))
    else:
        long_side = max(image.shape)
        cropped, _ = mx.image.center_crop(img, (long_side, long_side))

    if mean != None:
        normalized = mx.image.color_normalize(cropped.astype(np.float32)/255,
                                              mean=mx.nd.array(mean),
                                              std=mx.nd.array(std))
    else:
        normalized = cropped
    # the network expect batches of the form (N,3,224,224)
    # Transposing from (height, width, 3) to (3,height, width)
    transposed = normalized.transpose((2, 0, 1))
    # change the shape from (3, height, width) to (1, 3, height, width)
    batchified = transposed.expand_dims(axis=0)
    batchified = batchified.astype('float32')
    return batchified


'''
#PIL TO TENSOR AND TENSOR TO PIL
pil_img = Image.open(img)
print(pil_img.size)

pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
print(pil_to_tensor.shape)

tensor_to_pil = transforms.ToPILImage()(pil_to_tensor.squeeze_(0))
print(tensor_to_pil.size)
'''


# TO DO:
# check the shape of input for all the functions
# change channels using torch, mxnet functionality
# normalize a one channel image in transform_mxnet()
