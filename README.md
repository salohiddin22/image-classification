# Simple-Image-Classification-Model-Code

Classify objects using DL-based image classification model [resnet50](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py) [(paper)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf), test the model performance on unseen images during training, and perform model analysis using [GradCAM](https://github.com/jacobgil/pytorch-grad-cam).

### Create virtual environment
```python
conda create -n <ENV_NAME> python=3.9
conda activate <ENV_NAME>
pip install -r requirements.txt
```

### Sample data

![Capture]


file:///mnt/C6E2920FE29203B9/din/Deep_l/1_classification/Object-Classification-Modified/experiment/resnet/resnet50_64/3_cmatrix.png

### Run training 
```
python main.py  --root="/classification/1_data/nuts/"  -bs=32 -mn='resnet50' -d='cuda:0' -ld="resnet50_dir" -eps=100
```

### Results


### GradCAM


# image-classification
