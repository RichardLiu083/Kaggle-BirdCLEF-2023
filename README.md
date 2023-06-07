# Kaggle-HuBMAP-HPA-Hacking-the-Human-Body
**2022/09/23 - Silver Medal - Top 4%**  
[Competition Link](https://www.kaggle.com/competitions/hubmap-organ-segmentation)
![image](https://github.com/RichardLiu083/Kaggle-HuBMAP-HPA-Hacking-the-Human-Body/blob/main/img/Rank.png)

## Solution
![image](https://github.com/RichardLiu083/Kaggle-HuBMAP-HPA-Hacking-the-Human-Body/blob/main/img/Inference%20Pipeline.png)

## Insight
- Split the main task into two subtask (predict lung type and the others), which makes the model more stable while training and validation. 
  Probably because lung type have weak signal and also noisy label.
- CNN model is better than transformer model on lung type prediction.
- Transformer model is better than CNN model while predicting the other types.
- Stain Normalization can reduce the difference between HPA and Hubmap data.
- Scale adjustment is crucial since Hubmap data have different pixel size compare to HPA data.
- Doing Mosaic、Cutmix augmentation on lung type data will lead to performence drop.
- Multi-classes model is worse than single class model.

## Model
- EfficientNet_b7 * 2 (Unet decoder)
- Coat_medium (Daformer decoder)
- Segformer_b5 * 2 (different image size)
- Beit_base

## Augmentation
- Stain Normalization (helps a lot)
- Mosaic
- Cutmix
- H、V Flip
- HueSaturationValue
- ShiftScaleRotate
- CoarseDropout
- Blur
- GaussNoise

## Training
- 100 epochs
- lr 3e-4 for CNN, 6e-5 for Transformer
- bce + dice loss

## Validation
- 5 fold 
- only choose best model from validation

## Inference
- TTA * 8 (Flip、Rotate)
- choosing lower threshold for Hubmap data
- 2 model (CNN) for lung type prediction, 4 model (Transformer) for the others. (total 6 model)

## Top place method which I missed
- Pseudo label on GTEX portal data.
- Using full dataset to create one model (no validation).
- SWA or model fusion in the same training pipeline.
- Stain Normalization while inference (not only training).
