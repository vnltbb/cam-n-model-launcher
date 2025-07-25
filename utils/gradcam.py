import tensorflow as tf
import numpy as np

def make_gradcam_heatmap(img_array, model, layer_name, class_index=None):
    """
    Grad-CAM heatmap 생성

    Args:
        img_array (np.ndarray): shape = (1, H, W, 3) 전처리된 이미지
        model (tf.keras.Model): 분류 모델
        layer_name (str): 시각화할 convolution layer 이름
        class_index (int): 대상 클래스 인덱스 (None일 경우 예측값 최대 클래스 사용)

    Returns:
        heatmap (np.ndarray): shape = (H, W), 0~1로 정규화된 heatmap
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    # gradient 계산
    grads = tape.gradient(class_channel, conv_outputs)

    # 평균화된 gradient weight
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # conv feature map에 weight 곱하기
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 정규화
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
