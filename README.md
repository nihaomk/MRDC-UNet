# MRDC-UNet

The BlindPath-SegDataset\* \*dataset proposed in our paper is released [here](https://drive.google.com/drive/folders/1k78G5lKxYbCd5OBBBmxZbRqGweGI5u3n)

![image](https://github.com/nihaomk/MRDC-UNet/blob/main/images/Figure%204.jpeg)


MRD module, using a multi-branch learning strategy, the original input feature map x with its feature tensor obtained through the processing of each branch, spliced with x in the channel dimension to form a fused feature tensor out, the fused feature tensor out is then further feature integration and dimensionality reduction processed through a one-dimensional convolutional layer on the channel dimension, and finally through Residual links and Relu get result.


![image](https://github.com/nihaomk/MRDC-UNet/blob/main/images/Figure%207.jpeg)

<p align="center">MRDC-Net Network Architecture </p>                                                        

## Setup environment

    pip install -r requirements.txt

## Setup data

The `data` folder should be like this:

```
data  
├── BR-SEG
│   ├── images  
│   │   ├── ####.jpg  
│   ├── masks  
│   │   ├── ####.jpg  
│   ├── predict 
│   │   ├── images
│   │   │   ├── ####.jpg  
│   │   ├── masks
│   │   │   ├── ####.jpg  
│   │   ├── predict.txt

```

## How to train

    python train.py

**Note**: Currently we only support for single-gpu training.

## How to test

    python test.py

We trained our model using this BlindPath-SegDataset, and got similar results with our paper. The final weights can be download [here](https://drive.google.com/file/d/1xi6iNWcXmTTyvB_XhEMskEkyIgg3nFVl/view?usp=drive_link).
