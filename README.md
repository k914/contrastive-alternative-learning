# Best of Both Worlds: See and Understand Clearly in the Dark (ACM MM2022)

![image](https://github.com/k914/contrastive-alternative-learning/blob/main/Figure/first.PNG)

The architecture and training stategy
=====
![image](https://github.com/k914/contrastive-alternative-learning/blob/main/Figure/flow.png)

Codes
=====
### Requirements

* Python 3.6
* Pytorch==1.14.0
* Cuda 11.1

### How to use
* For training      
The traning codes will be released with the journal version.
* For testing    
For low-light image enhancement, run test_en.py with enhance.pth.  
For dark face detection/nighttime segmentation, run test_det.py.
Trained detection model is in https://drive.google.com/file/d/10zb5uC7j0N7fspG2HXljKQEYhRQUUaTQ/view?usp=drive_link.   
For nighttime semantic segmentation, visual results (ACDC night val set) of trained segmentation model are in https://drive.google.com/drive/folders/11qfyxLXjH_sQF1t4-4sRCEzHAZ0BywgJ?usp=drive_link. 
