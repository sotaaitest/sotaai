

# # Code to find the minimal acceptable size for models


from math import floor, ceil


def conv1d_out(layer, img_shape: int):
    width = img_shape

    padding = layer._kwargs['pad']
    dilation = layer._kwargs['dilate']
    kernel_size = layer._kwargs['kernel']
    stride = layer._kwargs['stride']

    out_width = floor((width+2*padding-dilation*(kernel_size-1)-1)/stride)+1
    return out_width


def conv2d_out(layer, img_shape: tuple):
    height = img_shape[0]
    width = img_shape[1]

    padding = layer._kwargs['pad']
    dilation = layer._kwargs['dilate']
    kernel_size = layer._kwargs['kernel']
    stride = layer._kwargs['stride']

    out_height = floor(
        (height+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
    out_width = floor(
        (width+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
    return (out_height, out_width)


def conv3d_out(layer, img_shape):
    depth = img_shape[0]
    height = img_shape[1]
    width = img_shape[2]

    padding = layer._kwargs['pad']
    dilation = layer._kwargs['dilate']
    kernel_size = layer._kwargs['kernel']
    stride = layer._kwargs['stride']

    out_depth = floor(
        (depth+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
    out_height = floor(
        (height+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
    out_width = floor(
        (width+2*padding[2]-dilation[2]*(kernel_size[2]-1)-1)/stride[2])+1
    return (out_depth, out_height, out_width)


def conv1dtranspose_out(layer, img_shape):
    width = img_shape

    padding = layer._kwargs['pad']
    output_padding = layer.outpad[0]
    kernel_size = layer._kwargs['kernel']
    strides = layer._kwargs['stride']

    out_width = (width-1)*strides-2*padding+kernel_size+output_padding
    return out_width


def conv2dtranspose_out(layer, img_shape):
    height = img_shape[0]
    width = img_shape[1]

    padding = layer._kwargs['pad']
    output_padding = layer.outpad[0]
    kernel_size = layer._kwargs['kernel']
    strides = layer._kwargs['stride']

    out_height = (height-1)*strides[0]-2 * \
        padding[0]+kernel_size[0]+output_padding[0]
    out_width = (width-1)*strides[1]-2*padding[1] + \
        kernel_size[1]+output_padding[1]

    return (out_height, out_width)


def conv3dtranspose_out(layer, img_shape):
    depth = img_shape[0]
    height = img_shape[1]
    width = img_shape[2]

    padding = layer._kwargs['pad']
    output_padding = layer.outpad[0]
    kernel_size = layer._kwargs['kernel']
    strides = layer._kwargs['stride']

    out_depth = (depth-1)*strides[0]-2*padding[0] + \
        kernel_size[0]+output_padding[0]
    out_height = (height-1)*strides[1]-2 * \
        padding[1]+kernel_size[1]+output_padding[1]
    out_width = (width-1)*strides[2]-2*padding[2] + \
        kernel_size[2]+output_padding[2]

    return (out_depth, out_height, out_width)


def pool1d_out(layer, img_shape: int):
    ''' Calculate output of a MaxPool1D or an AvgPool1D layer'''
    width = img_shape

    padding = layer._kwargs['pad']
    pool_size = layer._kwargs['kernel']
    strides = layer._kwargs['stride']
    if layer._kwargs['pooling_convention'] == 'full':
        ceil_mode = True
    else:
        ceil_mode = False

    if ceil_mode:
        out_width = ceil((width+2*padding-pool_size)/strides)+1
    else:
        out_width = floor((width+2*padding-pool_size)/strides)+1

    return out_width


def pool2d_out(layer, img_shape):
    ''' Calculate output of a MaxPool2D or an AvgPool2D layer'''
    height = img_shape[0]
    width = img_shape[1]

    padding = layer._kwargs['pad']
    pool_size = layer._kwargs['kernel']
    strides = layer._kwargs['stride']
    if layer._kwargs['pooling_convention'] == 'full':
        ceil_mode = True
    else:
        ceil_mode = False

    if ceil_mode:
        out_height = floor((height+2*padding[0]-pool_size[0])/strides[0])+1
        out_width = floor((width+2*padding[1]-pool_size[1])/strides[1])+1
    else:
        out_height = floor((height+2*padding[0]-pool_size[0])/strides[0])+1
        out_width = floor((width+2*padding[1]-pool_size[1])/strides[1])+1

    return (out_height, out_width)


def pool3d_out(layer, img_shape):
    ''' Calculate output of a MaxPool3D or an AvgPool3D layer'''
    depth = img_shape[0]
    width = img_shape[1]
    height = img_shape[2]

    padding = layer._kwargs['pad']
    pool_size = layer._kwargs['kernel']
    strides = layer._kwargs['stride']
    if layer._kwargs['pooling_convention'] == 'full':
        ceil_mode = True
    else:
        ceil_mode = False

    if ceil_mode:
        out_depth = ceil((depth+2*padding[0]-pool_size[0])/strides[0])+1
        out_height = ceil((height+2*padding[1]-pool_size[1])/strides[1])+1
        out_width = ceil((width+2*padding[2]-pool_size[2])/strides[2])+1
    else:
        out_depth = floor((depth+2*padding[0]-pool_size[0])/strides[0])+1
        out_height = floor((height+2*padding[1]-pool_size[1])/strides[1])+1
        out_width = floor((width+2*padding[2]-pool_size[2])/strides[2])+1

    return (out_depth, out_height, out_width)


def reflectionpad2d_out(layer, img_shape):
    height = img_shape[0]
    width = img_shape[1]

    padding = layer._padding[-1]
    out_height = height + 2*padding
    out_width = width + 2*padding
    return (out_height, out_width)


def find_func_4_layer(layer):
    l_type = str(type(layer))

    if 'Global' in l_type:
        return None
    # Get type of layer by getting only name of it after the dot
    l_type = l_type.rsplit('.')[-1]
    l_type = l_type.rsplit("'")[0].lower()
    if 'pool' in l_type:
        l_type = l_type[3:]

    l_type = l_type + '_out'
    return l_type


def feature_min_size(mod):
    '''
    Function that finds the minimum size that the model requires
    Input: model

    '''
    layers = mod.features
    k = []
    for l in layers:
        try:
            k.append(l.weight)
        except:
            continue
    k.reverse()  # start looking from the end of the model since sizes are descending towards the end
    for l in k:
        if len(l.shape) < 3:
            continue
        else:
            return l.shape[2]


def inf_bound_size(mod, img_shape):
    layers = mod.features
    shape = img_shape
    i = 0
    minn = feature_min_size(mod)
    out_vectors = []
    for layer in layers:
        if 'conv_layers' in str(type(layer)):
            func = find_func_4_layer(layer)
            shape = eval(func)(layer, shape)
            out_vectors.append(shape[0])
        # if shape<1: return shape,i
        i += 1
    return minn, out_vectors


def rec_step(mod, img_size, delta):
    if '1D' in str(mod.features[0]):
        im_shape = img_size
    if '2D' in str(mod.features[0]):
        im_shape = (img_size, img_size)
    if '3D' in str(mod.features[0]):
        im_shape = (img_size, img_size, img_size)

    minn, feat_vectors = inf_bound_size(mod, im_shape)
    if min(feat_vectors) > minn:
        img_size -= delta
        incr = False
    else:
        img_size += delta
        incr = True
    return img_size, incr


def find_im_size(mod):
    deltas = [50, 25, 10, 1]
    i = 1
    delta = deltas[0]
    img_size = 200
    img_size, incr = rec_step(mod, img_size, delta)
    inc_temp = incr
    while delta != 1:
        while incr == inc_temp:
            img_size, inc_temp = rec_step(mod, img_size, delta)
        delta = deltas[i]
        i += 1
        incr = inc_temp

    while inc_temp != incr:
        img_size, inc_temp = rec_step(mod, img_size, delta)

    if incr:
        img_size += 1

    return img_size
