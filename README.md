# Region Proposal Object Detection
Final Project for COMSE 6998: Practical Deep Learning Systems Performance. Region Proposal Object Detection Comparative Performance Analysis.

This project has been implemented using Keras and executed on the K80 GPU on Google Colab. We implement an object detection pipeline as follows:
- Download the Food101 dataset.
- Finetune different classifier architectures on the dataset after initializing with ImageNet weights(transfer learning).
- Tune hyperparameters of selective search and run on the image.
- Pick the best classifier based on performance metrics. Classify each proposal.
- Apply Non-Maxima Suppression to filter redundant proposal windows.
- Return final object detection result.

The repo contains:
- Python script for object detection.
- Jupyter notebooks for training and downloading Classifier architectures in the 'Training Logs and Notebooks' folder. 
- Notebook to implement the pipeline in Google Colab as an alternative to the Python script.
- Directory containing source code of Android App

# Instructions 

First, pick a classifier architecture of your choice, and execute the corresponding Jupyter Notebook in the 'Training Logs and Notebooks' folder. Download and store the trained model. Execute the Jupyter Notebook for object detection or run the script by following the instructions below:

Clone the Repo:

	git clone https://github.com/vm2656/COMSE6998_Final_Project.git

Install Requirements by opening a terminal and running:

	pip install -r requirements.txt
  
Download the Food101 dataset from https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ and extract it.

Open terminal and run the following command:

	python region_proposal_detection.py --image example.png

To filter by label, enter the following:
     
    python region_proposal_detection.py --image example.png --filter pizza


# Tables 
![](https://github.com/vm2656/COMSE6998_Final_Project/blob/main/Images/Table1.png)

![](https://github.com/vm2656/COMSE6998_Final_Project/blob/main/Images/Table2.png)

![](https://github.com/vm2656/COMSE6998_Final_Project/blob/main/Images/Table3.png)

# Example Results

After training till about 92% validation accuracy, some of the results produced by the model are shown below:

![](https://github.com/vm2656/COMSE6998_Final_Project/blob/main/Images/afternonmax3.png) 

![](https://github.com/vm2656/COMSE6998_Final_Project/blob/main/Images/afternonmax4.png)

![](https://github.com/vm2656/COMSE6998_Final_Project/blob/main/Images/afternonmax5.png)

As you can see, the bounding boxes aren't perfect, but the classification works. For further work, these labelled images can be fed to a YOLO, or an RCNN object detection pipeline can be implemented for better accuracy.  
  
The app was used to classify images using the live camera feed using the tflite model. An example still for the same:
![](https://github.com/vm2656/COMSE6998_Final_Project/blob/main/Images/app_donut.png)

# Credits
Credits to the tutorial on transfer learning found in here:

https://medium.com/@manasnarkar/transfer-learning-getting-started-9cebf5855a08

Credits to the app building tutorial found here:
https://developers.google.com/learn/topics/on-device-ml#build-your-first-on-device-ml-app

Dataset:

https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
