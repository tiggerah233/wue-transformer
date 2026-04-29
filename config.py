import torch


class Config:
    # 数据配置
    DATA_PATHS = {
        'vegetation_index': '/root/autodl-tmp/ndvi',
        'lai': '/root/autodl-tmp/裁剪后的lai数据500米',
        'precipitation': '/root/autodl-tmp/CHIRPS_Precip_2001_2020-20250906T072123Z-1-001/CHIRPS_Precip_2001_2020',
        'temperature': '/root/autodl-tmp/裁剪后的气温数据0.1°',
        'solar_radiation': '/root/autodl-tmp/裁剪后太阳辐射0.1°',
        'population': '/root/autodl-tmp/裁剪后人口数据100米°',
        'terrain': '/root/autodl-tmp/dem',
        'co2': '/root/autodl-tmp/CO2hebing.csv',
        'wue': '/root/autodl-tmp/wueanzhibei',
        'study_area': '/root/autodl-tmp/空间范围2/黄土.shp'  # 添加研究区掩膜路径
    }

    # 模型参数
    INPUT_CHANNELS = {
        'high_res': 1,  # 修改为1，因为只有LAI数据
        'medium_res': 3,  # 修改为3，植被指数、人口、地形
        'low_res': 3  # 修改为3，降水、气温、太阳辐射
    }
    HIDDEN_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 6
    DROPOUT = 0.1

    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据参数
    TEMPORAL_RESOLUTION = 'monthly'
    SPATIAL_RESOLUTIONS = {
        'high_res': (500, 500),  # LAI数据
        'medium_res': (1000, 1000),  # 植被指数、人口、地形
        'low_res': (1000, 1000)  # 降水、气温、太阳辐射(重采样后)
    }

    # 正则化参数
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.1
    DROPOUT_RATE = 0.1

    # 输出配置
    OUTPUT_DIR = 'results/'
    SAVE_MODEL = True

    # 新增配置：NaN处理和掩膜配置
    NAN_HANDLING_METHOD = 'interpolation'  # 'interpolation', 'knn', 'mask_only'
    USE_STUDY_AREA_MASK = True

    # 新增配置：分块训练配置
    USE_PATCH_TRAINING = True
    PATCH_SIZE = (512, 512)
    STRIDE = (256, 256)
    TIME_CHUNK_SIZE = 12

    # 新增配置：内存管理
    GRADIENT_CLIPPING = True
    MAX_GRAD_NORM = 1.0
    CLEANUP_INTERVAL = 10  # 每10个批次清理一次内存