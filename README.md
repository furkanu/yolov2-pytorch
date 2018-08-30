# YOLOv2 in PyTorch
Another PyTorch implementation of YOLOv2 object detection algorithm. 
I tried to make it a bit cleaner than some other implementations.
* There is a Jupyter notebook that you can use to test your own images or run the pretrained models on your camera.
* I tested this on PyTorch 0.4.1 but it should also work with 0.4.0.
* Training is not implemented. I started working on it but I never got to finish it.

## How to run the notebook?
<ul>
  <li>
        <b>
            You need to download pretrained weights in order to run the notebook. You can download them here: <br />
             YOLOv2 608x608	COCO: https://pjreddie.com/media/files/yolov2.weights <br />
             Tiny YOLO	VOC 2007+2012: https://pjreddie.com/media/files/yolov2-tiny-voc.weights
        </b>
  </li>
  <li>
      After that, you need to create a folder named weights and put them inside this folder.
  </li>
  <li>
      Now you should be able to run it if you have the required packages installed.
   </li>
   <li>
       I am planning to add an environment.yml so that you can easily create a conda environment to run it.
   </li>
</ul>

## References
This project took inspiration and/or code from these projects and courses/tutorials:
- Fast.ai
- https://github.com/ayooshkathuria/PyTorch-YOLO-v2
- https://github.com/marvis/pytorch-yolo2
- https://pjreddie.com/darknet/yolo/
- Deeplearning.ai > Convolutional Neural Networks > Week 3 > Car detection for Autonomous Driving
- https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
