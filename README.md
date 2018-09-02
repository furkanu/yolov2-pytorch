# YOLOv2 in PyTorch
Another PyTorch implementation of YOLOv2 object detection algorithm. 
I tried to make it a bit cleaner than some other implementations.
<ul>
<li>There is a Jupyter notebook that you can use to test your own images or run the pretrained </li>
<li>models on your camera. </li>
<li>I tested this on PyTorch 0.4.1 but it should also work with 0.4.0. </li>
<li>Training is not implemented. I started working on it but I never got to finish it. </li>
</ul>

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
</ul>

#### An easy way to get required packages installed
<ol>
    <li>
        You should have Anaconda installed on your machine:
        https://conda.io/docs/user-guide/install/index.html
    </li>
    <li>
        Download environment.yml file by running this command:
        <pre>
            wget https://raw.githubusercontent.com/furkanu/yolov2-pytorch/master/environment.yml
        </pre>
    </li>
    <li>
        Then, run the command below to create the conda environment with the required packages installed. The environment will be named "yolov2-pytorch" but you can change it by editing the first line of the environment.yml file.
        <pre>
            conda env create -f environment.yml
        </pre>
    </li>
    <li>
        After your environment has been created successfully, you can run these commands to add a kernel that you can select when running the notebook.
        <pre>
            source activate yolov2-pytorch #or whatever you named your environment
            python -m ipykernel install --user --name yolov2-pytorch --display-name "yolov2-pytorch"
        </pre>
    </li>
</ul>




## References
This project took inspiration and/or code from these projects and courses/tutorials:
<ul>
    <li>Fast.ai </li>
    <li>https://github.com/ayooshkathuria/PyTorch-YOLO-v2 </li>
    <li>https://github.com/marvis/pytorch-yolo2 </li>
    <li>https://pjreddie.com/darknet/yolov2/ </li>
    <li>Deeplearning.ai > Convolutional Neural Networks > Week 3 > Car detection for Autonomous Driving </li>
    <li>https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/ </li>
</ul>

