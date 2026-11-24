import torch
CFG = {
    "data_root": r"E:\Pycharm\Baseline\sifar-pytorch-main\data\images",  # 你的图片根目录
    "train_ann": "train.txt",  # 刚才生成的标注文件
    "val_ann": "val.txt",  # 自行拆分的验证集标注
    "image_size": 224,
    "batch_size": 16,  # 显卡内存小的话改8
    "lr": 1e-4,
    "epochs": 20,
    "num_classes": 2,  # 二分类
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

DATASET_CONFIG = {
    'st2stv2': {
        'num_classes': 174,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 3,
        'label_file': 'categories.txt'
    },
    'mini_st2stv2': {
        'num_classes': 87,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'test_list_name': 'mini_test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 3,
    },
    'st2stv1': {
        'num_classes': 174,
        'train_list_name': 'training_256.txt',
        'val_list_name': 'validation_256.txt',
        'test_list_name': 'testing_256.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 3,
        'label_file': 'something-something-v1-labels.csv'
},
    'kinetics400': {
        'num_classes': 400,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30,
        'label_file': 'image/kinetics-400_label.csv'
    },
    'mini_kinetics400': {
        'num_classes': 200,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'test_list_name': 'mini_test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'charades': {
        'num_classes': 157,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'diva': {
        'num_classes': 19,
        'train_list_name': 'DIVA_GT_RGB_TSM_train.txt',
        'val_list_name': 'DIVA_GT_RGB_TSM_validate.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:08d}.jpg',
        'filter_video': 0
    },
    'diva_pvi': {
        'num_classes': 8,
        'train_list_name': 'DIVA_PVI_GT_RGB_TSM_train.txt',
        'val_list_name': 'DIVA_PVI_GT_RGB_TSM_validate.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:08d}.jpg',
        'filter_video': 0
    },
    'moments': {
        'num_classes': 339,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'mini_moments': {
        'num_classes': 200,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'ucf101': {
        'num_classes': 101,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'hmdb51': {
        'num_classes': 51,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'jester': {
        'num_classes': 27,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
}


def get_dataset_config(dataset, use_lmdb=False):
    ret = DATASET_CONFIG[dataset]
    num_classes = ret['num_classes']
    train_list_name = ret['train_list_name'].replace("txt", "lmdb") if use_lmdb \
        else ret['train_list_name']
    val_list_name = ret['val_list_name'].replace("txt", "lmdb") if use_lmdb \
        else ret['val_list_name']
    test_list_name = ret.get('test_lmdb_name', None)
    if test_list_name is not None:
        test_list_name = test_list_name.replace("txt", "lmdb")
    filename_seperator = ret['filename_seperator']
    image_tmpl = ret['image_tmpl']
    filter_video = ret.get('filter_video', 0)
    label_file = ret.get('label_file', None)

    return num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, \
           image_tmpl, filter_video, label_file
