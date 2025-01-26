## Enhanced segmentation of optic disc and cup using attention-based U-Net with dense dilated series convolutions


## Table of contents
* [Abstract](#abstract)
* [Research Paper Link](#research-paper-link)
* [Dataset info](#dataset-info)
* [Project Files](#project-files)
* [UI](#ui)

## Abstract
Delineating the boundaries of the optic disc and cup regions is a critical pre-requisite for glaucoma screening because it allows for precise measurement of key parameters, such as cup-to-disc ratio, which is a critical indicator of optic nerve head damage, a hallmark of glaucoma progression. Accurate segmentation enables early detection and monitoring of the disease, aiding in timely intervention to prevent vision loss. The main contribution of this research work is to develop an automated process to isolate and demarcate the optic disc and cup from retinal fundus images. To prevent the blood vessels from interfering with the segmentation process, a novel method is used for vessel mask generation and vessel inpainting. Most of the research works have used based encoder–decoder models like U-Net architecture or handcrafted feature extraction techniques such as hough transform, fuzzy clustering, etc. The proposed model has made significant modifications to the U-Net model. (1) Dual attention mechanism at every layer of decoder and (2) dense dilated series convolutions as skip connections to generate higher level feature map. The proposed model achieved benchmark accuracies - Dice score of 95.95% and IoU score of 92.22% for optic disc segmentation averaged over fivefold. For the task of outlining the optic cup region, it attained a Dice score of 88.7% and IoU of 79.72%.
	
## Research Paper Link

### Published in Springer Nature **Neural Computing and Applications** journal (Q1, 5Y IF: 4.8)

Click on this [link](https://link.springer.com/epdf/10.1007/s00521-025-10989-x?sharing_token=ySeNL4ePGQMRHvNAEC5fSPe4RwlQNchNByi7wbcMAY471hlVSY6YfCOXX-KdBIWWVndAiGRtDc8J2sHxfPizUVXS4ozSbSqv7aGtLEOP7vL3icnA1JmYEBLvqmW05muQ6jskI6ty_iq0a64u5CEe4VUgdFBoA6rC8aCbz91VwKY%3D) to view the paper

## Dataset info

Pre-processed dataset link:- Click [here](https://www.kaggle.com/datasets/sohamkumar1703/enhanced-fundus-image-dataset-for-glaucoma)

Data is collected from:

| Dataset | Number of Images |
|---------|------------------|
|  REFUGE |      800         |
|  G1020  |      790         |
|  ORIGA  |      650         |
|  PAPILA |      488         |
|  DRISHTI|      101         |
|  CRFO   |      44          |


## Project Files

* [process.py](https://github.com/sohamkumar1703/Final_Year_Project/blob/main/process.py) -- contains function for pre-processing
* [report.py](https://github.com/sohamkumar1703/Final_Year_Project/blob/main/report.py) -- contains the model architecture and mask generation and glaucoma indication functions
* [app.py](https://github.com/sohamkumar1703/Final_Year_Project/blob/main/app.py) -- contains functions for the UI 


## UI

* Create a virtual environment and install the necessary libraries
* Run app.py

![image](https://github.com/user-attachments/assets/09a79cb9-0f89-4926-bbd1-366646a3f4f4)

Click on **Browse**, upload the image anf then click on **Upload**

![image](https://github.com/user-attachments/assets/446181b2-ef94-4941-840c-ce3de1abc166)

![image](https://github.com/user-attachments/assets/84ad1a4d-3bc9-4647-94a2-9d8cdec30e50)

   

