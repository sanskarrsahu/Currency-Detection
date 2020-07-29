Currency Detection system
This project focusses on telling the currency and the denomination of the notes taken as a image. It can take input via saved images and as well as via web cam. It uses opencv and python. It consists of two folders:- 
1. Sample- It consists of the dataset images and is currently holding only indian and romanian currency images.
2. Backup- Consists of some sampled images that can be browsed and selected via gui to identify the currency and denomination.
Then the folder consists of four python files which includes the coding part of it in which As a feature extractor I have used ORB along with BFmatcher and the GUI is written in tkinter with ttk widgets(system theme):-
1. util.py- In this the images present the sample folder is being trained and prepapred.
2. Config.py- Is used for basic purpose like close key aor for adjusting image contrast, etc. Includes the operations performed on image while its training and comparision.
3. Detector.py- In this the image taken as input is compared with all the images stored as dataset and accordingly the result is decided.
4. Gui.py- In this a graphical user interface is given to take image as input.

You should run gui.py file to take input and taking the input can take some time depending on the performance of system as the functions used here after taking the input compares it with the stored images and it also trains all the dataset images and then compare it with the input image thus taking time.
The light conditions will probably effect the result and the images in the dataset is not enough to perfectly identify the input image so till now it is recommended to give proper light conditions to effective result. 
The result can be observed on the console log and when taking input via can it can be observed on the console log as well as on the live screen.
We are not submitting lagre dataset as the dataset is large lot of data will be required to upload as well as download it. The original dataset is aroung 4.2 gb but we are only including only important dataset images. So when we include more and more images in the Dataset the efficiency of the result will definately increase and thus providing more effective result as compared to now.
