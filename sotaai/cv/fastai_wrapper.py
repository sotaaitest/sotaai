from fastai.vision import *

MODELS = [
    'alexnet',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'mobilenet_v2',
    'resnet101',
    'resnet152',
    'resnet18',
    'resnet34',
    'resnet50',
    'squeezenet1_0',
    'squeezenet1_1',
    'vgg11_bn',
    'vgg13_bn',
    'vgg16_bn',
    'vgg19_bn',
    'xresnet101',
    'xresnet152',
    'xresnet18',
    'xresnet18_deep',
    'xresnet34',
    'xresnet34_deep',
    'xresnet50',
    'xresnet50_deep'
    #--------------------- Customize models -----------------#
    # 'Darknet',
    # 'DynamicUnet',
    # 'ResNet',
    # 'SqueezeNet',
    # 'WideResNet',
    # 'xception' #not tested and no pretrained model available
]

DATASETS = {'classification': ['CALTECH_101',
                               # 'CARS',
                               'CIFAR',
                               'CIFAR_100',
                               # 'CUB_200_2011',#Caltech UCSD Birds
                               'DOGS',
                               # 'FLOWERS',
                               # 'FOOD',
                               'IMAGENETTE',
                               'IMAGENETTE_160',
                               'IMAGENETTE_320',
                               'IMAGEWOOF',
                               'IMAGEWOOF_160',
                               'IMAGEWOOF_320',
                               # 'LSUN_BEDROOMS',
                               'MNIST',
                               'MNIST_SAMPLE',
                               'MNIST_TINY',
                               'MNIST_VAR_SIZE_TINY',
                               'PETS',
                               'SKIN_LESION'],
            'key-point detection': ['BIWI_SAMPLE'],  # , 'BIWI_HEAD_POSE'],
            'object detection': ['COCO_SAMPLE',
                                 'COCO_TINY'],
            'multi-label classification': ['PLANET_SAMPLE',
                                           'PLANET_TINY'],
            'segmentation': ['CAMVID',
                             'CAMVID_TINY']
            }


''' Not yet wrapped

 #'PASCAL_2007',                                     #Checksum does not match. No download
 #'PASCAL_2012',                                     #Checksum does not match. No download
 'FLOWERS',
 'BIWI_HEAD_POSE', #
 'CARS',#Classification
 'CUB_200_2011', #Classification, detection            #Caltech UCSD Birds
 'FOOD',#Classification
 'LSUN_BEDROOMS',#Classification
'''


def load_model(model_name: str, pretrained: bool = False):
    trainer = getattr(models, model_name)
    return trainer(pretrained=pretrained)


def load_dataset(dataset_name, destination="datasets"):
    url_dataset = getattr(URLs, dataset_name)
    path = untar_data(url_dataset, dest=destination)
    # Classification
    if dataset_name == "MNIST":
        data = ImageDataBunch.from_folder(
            path, train="training", test="testing")

    if dataset_name in ["MNIST_SAMPLE", "MNIST_TINY", "MNIST_VAR_SIZE_TINY", "CIFAR", "CIFAR_100", "DOGS"]:
        if dataset_name == "DOGS":
            data = ImageDataBunch.from_folder(
                path, test="test1", no_check=True)
        data = ImageDataBunch.from_folder(path, test="test", no_check=True)

    if dataset_name == 'SKIN_LESION':
        data = ImageDataBunch.from_folder(path, no_check=True)

    if dataset_name == 'PETS':
        path_img = path/'images'
        fnames = get_image_files(path_img)
        re = r'/([^/]+)_\d+.jpg$'
        data = ImageDataBunch.from_name_re(
            path_img, fnames, re, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    #learn = cnn_learner(data, models.resnet18, metrics=accuracy)
    # learn.fit(1)
    # Multicategory
    if dataset_name in ["PLANET_SAMPLE", "PLANET_TINY"]:
        data = ImageDataBunch.from_csv(
            path, folder="train", suffix=".jpg", label_delim=" ")

    if dataset_name == 'CALTECH_101':
        parent_path = Path('/'.join(path.parts[1:-1]))
        data = ImageDataBunch.from_folder(
            parent_path/"101_ObjectCategories", valid_pct=0.2, no_check=True)

    if dataset_name in ['IMAGENETTE', 'IMAGENETTE_160', 'IMAGENETTE_320', 'IMAGEWOOF', 'IMAGEWOOF_160', 'IMAGEWOOF_320']:
        data = ImageDataBunch.from_folder(path, valid='val')

    if dataset_name in ['COCO_SAMPLE', 'COCO_TINY']:
        if dataset_name == 'COCO_SAMPLE':
            annot = path/'annotations/train_sample.json'
            imgs_folder = 'train_sample'
        else:
            annot = path/'train.json'
            imgs_folder = 'train'
        images, lbl_bbox = get_annotations(annot)
        img2bbox = dict(zip(images, lbl_bbox))
        def get_y_func(o): return img2bbox[o.name]
        data = (ObjectItemList.from_folder(path / imgs_folder)
                .random_split_by_pct()
                .label_from_func(get_y_func)
                .transform(get_transforms(), tfm_y=True, padding_mode='zeros', size=128,)
                .databunch(bs=16, collate_fn=bb_pad_collate))

    """
    if dataset_name == "FLOWERS":
        df_train=pd.read_table(path.ls()[3],names=["name","label"],sep=" ",index_col=False)
        df_valid=pd.read_table(path.ls()[2],names=["name","label"],sep=" ",index_col=False)
        df_test=pd.read_table(path.ls()[1],names=["name","label"],sep=" ",index_col=False)

        data_train=ImageDataBunch.from_df(path,df_train, valid_pct=0, no_check=True, size=224)
        data_test=ImageDataBunch.from_df(path,df_test, valid_pct=0, no_check=True, size=224)
        data_valid=ImageDataBunch.from_df(path,df_valid, valid_pct=0, no_check=True, size=224)
        return data_train, data_valid, data_test
    """
    if dataset_name == "BIWI_SAMPLE":
        fn2ctr = pickle.load(open(path/"centers.pkl", "rb"))
        data = (PointsItemList.from_folder(path)
                .split_by_rand_pct(seed=42)
                .label_from_func(lambda o: fn2ctr[o.name])
                .transform(get_transforms(), tfm_y=True, size=(120, 160))
                .databunch()
                .normalize(imagenet_stats))

    if dataset_name in ['CAMVID', "CAMVID_TINY"]:
        path_lbl = path/"labels"
        path_img = path/'images'
        codes = np.loadtxt(path/'codes.txt', dtype=str)
        def get_y_fn(x): return path_lbl/f'{x.stem}_P{x.suffix}'
        data = (SegmentationItemList.from_folder(path_img)
                .split_by_rand_pct()
                .label_from_func(get_y_fn, classes=codes)
                .transform(get_transforms(), tfm_y=True, size=128)
                .databunch(bs=16, path=path)
                .normalize(imagenet_stats))

    return data
