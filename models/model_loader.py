import tensorflow as tf
import os

def load_model(model_path):
    """
    지정된 경로에서 Keras 모델(.keras) 파일을 로드합니다.

    Args:
        model_path (str): 모델 파일 경로

    Returns:
        keras.Model: 로드된 모델 객체
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print(f"✅ 모델 로드 완료: {model_path}")
    return model
