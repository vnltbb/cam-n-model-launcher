import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def save_classification_result(img_array, top_preds, probs, save_dir, input_name, input_folder=None):
    """
    분류 결과 시각화 및 저장

    Args:
        img_array (np.ndarray): (224, 224, 3) 형태의 원본 이미지
        top_preds (list[str]): 상위 예측 클래스명
        probs (list[float]): 예측 확률 (0~1)
        save_dir (str): 저장 경로 (results/classification/)
        input_name (str): 입력 이미지 파일명
        input_folder (str): 입력 폴더명 (없으면 None)
    """
    os.makedirs(save_dir, exist_ok=True)

    if input_folder:
        save_dir = os.path.join(save_dir, input_folder)
        os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_array.astype("uint8"))
    ax.axis('off')

    # 상단에 top-1~3 클래스 결과 표기
    title = f"Top-1: {top_preds[0]} ({probs[0]*100:.1f}%)\n"
    title += f"Top-2: {top_preds[1]} ({probs[1]*100:.1f}%)\n"
    title += f"Top-3: {top_preds[2]} ({probs[2]*100:.1f}%)"
    ax.set_title(title, fontsize=8)

    fname = f"{os.path.splitext(input_name)[0]}_classification.png"
    save_path = os.path.join(save_dir, fname)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 분류 결과 저장 완료: {save_path}")

def save_gradcam_result(orig_image, heatmap, save_dir, model_name, layer_name, input_name):
    """
    Grad-CAM 결과 저장

    Args:
        orig_image (np.ndarray): 원본 이미지
        heatmap (np.ndarray): 0~1로 정규화된 heatmap
        save_dir (str): 저장 디렉토리 (results/grad_cam/)
        model_name (str): 모델 이름
        layer_name (str): convolution layer 이름
        input_name (str): 원본 이미지 이름
    """
    os.makedirs(save_dir, exist_ok=True)

    # heatmap을 컬러로 변환 후 이미지와 합성
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_image.astype("uint8"), 0.6, heatmap_color, 0.4, 0)

    # 파일명 생성
    base_name = os.path.splitext(input_name)[0]
    fname = f"{model_name}_{layer_name}_{base_name}_gradcam.png"
    save_path = os.path.join(save_dir, fname)

    # 저장
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"✅ Grad-CAM 저장 완료: {save_path}")
