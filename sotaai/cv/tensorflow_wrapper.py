#import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow as tf

DATASETS = {'video':    ['bair_robot_pushing_small',  # without as_supervised
                         'moving_mnist',  # without as_supervised
                         # 'robonet', #Apache beam
                         'starcraft_video',  # without as_supervised
                         'ucf101'  # needs ffmpeg. Instalado en ADA
                         ],
            'object detection':     ['coco',  # without as_supervised
                                     'kitti',  # without as_supervised
                                     'open_images_challenge2019_detection',  # 594 GB
                                     'open_images_v4',  # need 565 GB space
                                     'voc',  # good without as_supervised
                                     # 'waymo_open_dataset', #Apache beam
                                     'wider_face'  # as_supervised. need PIL
                                     ],
            'classification':       ['beans',  # good
                                     # 'bigearthnet', #apache beam dataset: https://www.tensorflow.org/datasets/beam_datasets#generating_a_beam_datasetfor
                                     'binary_alpha_digits',  # good
                                     'caltech101',  # good
                                     'caltech_birds2010',  # good
                                     'caltech_birds2011',  # good
                                     'cars196',  # good
                                     'cats_vs_dogs',  # good
                                     'cifar10',  # good
                                     'cifar100',  # good
                                     # 'cifar10_1', #checksum does not match, although URL does exist and works properly
                                     'cifar10_corrupted',  # good
                                     'citrus_leaves',  # good
                                     'cmaterdb',  # good
                                     'colorectal_histology',  # good
                                     'colorectal_histology_large',  # without as suppervised
                                     'curated_breast_imaging_ddsm',  # manual download. run download to see instructions
                                     'cycle_gan',  # good
                                     'deep_weeds',  # good
                                     # manual download.
                                     'diabetic_retinopathy_detection',
                                     # 'dtd', #wrong checksum. long download btw.Artifact https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz,
                                     'emnist',  # good
                                     'eurosat',  # good
                                     'fashion_mnist',  # good
                                     # 'food101',   #wrong checksum
                                     'geirhos_conflict_stimuli',  # good
                                     'horses_or_humans',  # good
                                     'i_naturalist2017',  # not enough disk space
                                     'imagenet2012',  # manual
                                     'imagenet2012_corrupted',  # manual
                                     'imagenet2012_subset',  # manual
                                     'imagenet_resized',  # good
                                     'imagenette',  # good
                                     'imagewang',  # good
                                     'kmnist',  # good
                                     'lfw',  # good
                                     'malaria',  # good
                                     'mnist',  # good
                                     'mnist_corrupted',  # good
                                     'omniglot',  # good
                                     # 'oxford_flowers102', #wrong checksum
                                     # 'oxford_iiit_pet',#wrong checksum
                                     # 'patch_camelyon',#wrong checksum
                                     'places365_small',  # good
                                     'quickdraw_bitmap',  # good
                                     'resisc45',  # manual
                                     'rock_paper_scissors',  # good
                                     'smallnorb',  # good
                                     'so2sat',  # good
                                     'stanford_dogs',  # good
                                     # without as_supervised argument. Runs perfectly without it
                                     'stanford_online_products',
                                     'stl10',  # killed process out of memory. Was good in server
                                     'sun397',
                                     'svhn_cropped',  # good
                                     'tf_flowers',  # good
                                     'uc_merced',  # good
                                     'vgg_face2',  # manual
                                     # 'visual_domain_decathlon' #checksum wrong
                                     ],
            'other': [  # 'abstract_reasoning',#apache_beam
    'aflw2k3d',  # without as_supervised
    'binarized_mnist',  # without as_supervised
    'celeb_a',  # without as supervised
    'celeb_a_hq',  # manual
    'chexpert',  # manual
    'cityscapes',  # manual
    'clevr',  # without as_supervised
    'coil100',  # good
    'div2k',  # good
    'downsampled_imagenet',  # without as_supervised
    'dsprites',  # without as_supervised
    'duke_ultrasound',  # without as_supervised.Try GCS instead of downloading!
    'flic',  # without as supervised
    'lost_and_found',  # without as supervised
    'lsun',  # without as_supervised. Needs tensorflow_io
    'scene_parse150',  # good
    'shapes3d',  # without as_supervised
    'the300w_lp'  # without as_supervised
]
}


def available_datasets(task: str = 'all'):
    if task == "all":
        def flatten(l): return [item for sublist in l for item in sublist]
        flat_ds = flatten(DATASETS.values())
        return flat_ds

    return DATASETS[task]


# tfds.list_builders()

# models usually expect tf.float32, but tfds provides tf.uint8, hence let's normalize the image
def normalize_img(image, label):
    return tf.cast(image, tf.float32)/255., label


def load_dataset(name_dataset):
    _, ds_info = tfds.load(name_dataset, with_info=True)
    try:
        ls = list(ds_info.splits.keys())
        ds = tfds.load(
            name_dataset,
            split=ls,
            shuffle_files=True,
            as_supervised=True)

    except:
        ls = list(ds_info.splits.keys())
        ds = tfds.load(
            name_dataset,
            split=ls,
            shuffle_files=True)

    ds_dic = {}
    for i in range(len(ls)):
        if ls[i] == 'test':
            ds[i] = ds[i].map(
                normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            #ds[i] = ds[i].batch(64)
            # ds[i].prefetch(tf.data.experimental.AUTOTUNE)
            ds_dic[ls[i]] = ds[i]

        else:
            ds[i] = ds[i].map(
                normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            #ds_train = ds_train.cache()
            #ds[i] = ds[i].shuffle(ds_info.splits['train'].num_examples)
            #ds[i] = ds[i].batch(64)
            # ds[i].prefetch(tf.data.experimental.AUTOTUNE)
            ds_dic[ls[i]] = ds[i]

    return ds_dic
