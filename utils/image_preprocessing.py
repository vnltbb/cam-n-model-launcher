import os
import numpy as np
from PIL import Image
import tensorflow as tf

# 백본 이름별로 전처리 함수 매핑
from tensorflow.keras.applications import (
    efficientnet, resnet, densenet, mobilenet
)

SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

def get_model_architecture_name(model_path):
    """
    저장된 모델 파일명에서 아키텍처 이름 추출
    예시: saved_models/EfficientNetB0_seed99.keras → EfficientNetB0
    """
    filename = os.path.basename(model_path)
    model_name = filename.split('_')[0]
    return model_name

def get_preprocessing_function(model_name):
    """
    모델 이름에 따른 전처리 함수 반환
    """
    name = model_name.lower()
    if "efficientnet" in name:
        return efficientnet.preprocess_input
    elif "resnet" in name:
        return resnet.preprocess_input
    elif "densenet" in name:
        return densenet.preprocess_input
    elif "mobilenet" in name:
        return mobilenet.preprocess_input
    else:
        raise ValueError(f"지원되지 않는 모델 이름: {model_name}")

def load_and_preprocess_image(image_path, target_size, preprocess_func):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = preprocess_func(img_array)
    return img_array

def load_images_from_path(input_path, model_path, target_size=(224, 224)):
    """
    이미지 파일 또는 폴더에서 이미지들을 로드하고 모델에 맞게 전처리

    Args:
        input_path (str): 입력 이미지 또는 폴더 경로
        model_path (str): 저장된 .keras 모델 경로
        target_size (tuple): 모델 입력 이미지 크기 (w, h)

    Returns:
        images_np (np.ndarray): 전처리된 이미지 배열
        filenames (list[str]): 이미지 파일명 리스트
    """
    model_name = get_model_architecture_name(model_path)
    preprocess_func = get_preprocessing_function(model_name)

    image_list = []
    filenames = []

    if os.path.isdir(input_path):
        for fname in sorted(os.listdir(input_path)):
            if fname.endswith(SUPPORTED_FORMATS):
                fpath = os.path.join(input_path, fname)
                img = load_and_preprocess_image(fpath, target_size, preprocess_func)
                image_list.append(img)
                filenames.append(fname)
    elif os.path.isfile(input_path) and input_path.endswith(SUPPORTED_FORMATS):
        img = load_and_preprocess_image(input_path, target_size, preprocess_func)
        image_list.append(img)
        filenames.append(os.path.basename(input_path))
    else:
        raise ValueError(f"❌ 지원되지 않는 입력 형식: {input_path}")

    images_np = np.stack(image_list, axis=0)
    return images_np, filenames
