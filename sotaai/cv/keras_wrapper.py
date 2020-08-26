import tensorflow.keras.applications as models
import tensorflow.keras.datasets as dset
# Models that do not contain classifier_activation as an argument
differ = ["ResNet50", "ResNet101", "ResNet152", "DenseNet121",
          "DenseNet169", "DenseNet201", "NASNetMobile", "NASNetLarge"]
# MobileNet models
mobile_nets = ["MobileNet", "MobileNetV2"]
DATASETS = {'classification': [
    'mnist', 'cifar10', 'cifar100', 'fashion_mnist']}
MODELS = ['InceptionResNetV2',
          'InceptionV3',
          'ResNet101V2',
          'ResNet152V2',
          'ResNet50V2',
          'VGG16',
          'VGG19',
          'Xception'] + differ + mobile_nets


def tasks():
    return list(DATASETS.keys())


def available_datasets(task: str = 'all'):
    if task == "all":
        def flatten(l): return [item for sublist in l for item in sublist]
        flat_ds = flatten(DATASETS.values())
        return flat_ds

    return DATASETS[task]


def available_models(task: str = 'all'):
    return MODELS

# MODELS ARE NOT COMPILED FOR TRAINING!!


def load_model(model_name,
               pretrained=False,
               alpha=1.0,
               depth_multiplier=1,
               dropout=0.001,
               include_top=True,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    """Load a model with specific configuration.
    Args:
      model_name (string): name of the model/algorithm.
      include_top: whether to include the fully-connected layer at the top of the network.
      weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
      input_shape: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (299, 299, 3). It should have exactly 3 inputs channels, and width and height should be no smaller than 71. E.g. (150, 150, 3) would be one valid value.
      pooling: Optional pooling mode for feature extraction when include_top is False.
        None means that the output of the model will be the 4D tensor output of the last convolutional block.
        avg means that global average pooling will be applied to the output of the last convolutional block, and thus the output of the model will be a 2D tensor.
        max means that global max pooling will be applied.
      alpha: Controls the width of the network. This is known as the width multiplier in the MobileNet paper. - If alpha < 1.0, proportionally decreases the number of filters in each layer. - If alpha > 1.0, proportionally increases the number of filters in each layer. - If alpha = 1, default number of filters from the paper are used at each layer. Default to 1.0.
      depth_multiplier: Depth multiplier for depthwise convolution. This is called the resolution multiplier in the MobileNet paper. Default to 1.0.
      dropout: Dropout rate. Default to 0.001.
      classes: optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.
      classifier_activation: A str or callable. The activation function to use on the "top" layer. Ignored unless include_top=True. Set classifier_activation=None to return the logits of the "top" layer.

    """
    if pretrained:
        weights = 'imagenet'
    else:
        weights = None
    # Load the trainer and return.
    trainer = getattr(models, model_name)
    if model_name in differ:
        return trainer(weights, input_tensor, input_shape, pooling, classes)
    elif model_name == "MobileNet":
        return trainer(weights, alpha, depth_multiplier, dropout, input_tensor, input_shape, pooling, classes)
    elif model_name == "MobileNetV2":
        return trainer(weights, alpha, input_tensor, input_shape, pooling, classes)
    return trainer(weights, input_tensor, input_shape, pooling, classes, classifier_activation)


def load_dataset(dataset_name):
    """
    Args:
      dataset_name (string): name of dataset. Available for computer vision task (only classification in Keras): mnist, cifar10,cifar100,fashion_mnist
    Returns:
      (x_train,y_train),(x_test,y_test)
    """
    ds = getattr(dset, dataset_name)
    return ds.load_data()


"""
def run(model, dataset):
    '''
    Input: numpy array of shape (N,H,W,C)
    Output: output of models
    '''
    in_shape = model.input.shape
"""
