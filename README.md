# YOLOv2 in PyTorch
Another PyTorch implementation of YOLOv2 object detection algorithm. 
I tried to make it a bit cleaner than some other implementations.

- There is a Jupyter notebook that you can use to test your own images or run the pretrained models on your camera.
- I tested this on PyTorch 0.4.1 but it should also work with 0.4.0.
- Training is not implemented. I started working on it but I never got to finish it.

## How to run the notebook?
- You need to download pretrained weights in order to run the notebook. You can download them here: <br>
  [YOLOv2 608x608	COCO](https://pjreddie.com/media/files/yolov2.weights) <br>
  [Tiny YOLO	VOC 2007+2012](https://pjreddie.com/media/files/yolov2-tiny-voc.weights)
- After that, you need to create a folder named weights and put them inside this folder.
- Now you should be able to run it if you have the required packages installed.

#### An easy way to get required packages installed

1. You should have Anaconda installed on your machine:
  https://conda.io/docs/user-guide/install/index.html
2. Download environment.yml file by running this command:
```bash
wget https://raw.githubusercontent.com/furkanu/yolov2-pytorch/master/environment.yml
```
3. Then, run the command below to create the conda environment with the required packages installed. The environment will be  named "yolov2-pytorch" but you can change it by editing the first line of the environment.yml file.
```bash
conda env create -f environment.yml
```
4. After your environment has been created successfully, you can run these commands to add a kernel that you can select when running the notebook.
```bash
#replace "yolov2-pytorch" with your environment name if you changed it.
source activate yolov2-pytorch 
python -m ipykernel install --user --name yolov2-pytorch --display-name "yolov2-pytorch"
```
## References
This project took inspiration and/or code from these projects and courses/tutorials:
- Fast.ai
- https://github.com/ayooshkathuria/PyTorch-YOLO-v2
- https://github.com/marvis/pytorch-yolo2
- https://pjreddie.com/darknet/yolov2/
- Deeplearning.ai > Convolutional Neural Networks > Week 3 > Car detection for Autonomous Driving
- https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/


