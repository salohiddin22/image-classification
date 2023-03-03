# Simple-Image-Classification-Model-Code

Classify objects using DL-based image classification model [resnet50](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py) [(paper)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf), test the model performance on unseen images during training, and perform model analysis using [GradCAM](https://github.com/jacobgil/pytorch-grad-cam).

### Create virtual environment
```python
conda create -n <ENV_NAME> python=3.9
conda activate <ENV_NAME>
pip install -r requirements.txt
```

### Sample data

![image](https://user-images.githubusercontent.com/89576437/222754944-3e69f284-a8df-4561-bb76-40509e91a17f.png)


### Run training 
```
python main.py  --root="/classification/1_data/nuts/"  -bs=32 -mn='resnet50' -d='cuda:0' -ld="resnet50_dir" -eps=20
```

### Results
![image](https://user-images.githubusercontent.com/89576437/222756441-9ec039e7-97ad-463a-bd29-21ba26848476.png)


### GradCAM
![grad-cam](https://user-images.githubusercontent.com/89576437/222756688-5a4ad976-1610-40c1-8af9-ade0b34365b8.png)
![image](https://user-images.githubusercontent.com/89576437/222756740-9b3383c5-d2f8-4a90-b464-2dc34a8ae15b.png)
![image](https://user-images.githubusercontent.com/89576437/222756797-18caed1a-31ed-4066-b518-c68ed8c0ef16.png)
![image](https://user-images.githubusercontent.com/89576437/222756825-45c33d87-e98a-4cec-a2e6-c27f552049aa.png)
![image](https://user-images.githubusercontent.com/89576437/222756860-96a50e26-1b05-40e1-81ce-ddda51b01601.png)
![image](https://user-images.githubusercontent.com/89576437/222756883-d16b32b3-e82e-4ce8-aae3-7746d6be9ff7.png)
![image](https://user-images.githubusercontent.com/89576437/222756919-834e9beb-ea9e-4cf9-a915-007dac229336.png)
![image](https://user-images.githubusercontent.com/89576437/222756958-167775f4-539b-4090-b97a-b5df25a10ce7.png)





