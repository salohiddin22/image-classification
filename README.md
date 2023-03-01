# Object-Classification-and-Classifier-Analysis

Classify objects using DL-based image classification model [rexnet_150](https://github.com/clovaai/rexnet) [(paper)](https://arxiv.org/pdf/2007.00992.pdf), test the model performance on unseen images during training, and perform model analysis using [GradCAM](https://github.com/jacobgil/pytorch-grad-cam).

### Create virtual environment
```python
conda create -n <ENV_NAME> python=3.9
conda activate <ENV_NAME>
pip install -r requirements.txt
```

### Sample data

![Capture]

### Run training 
```
python main.py  --root="/classification/1_data/nuts/"  -bs=32 -mn='resnet50' -d='cuda:0' -ld="resnet50_dir" -eps=100
```

### Results


### GradCAM


# image-classification
