# SmileRecognition

This is an end-to-end deep learning computer vision project

Please clone the repostory with the data.

You will then want unzip the images.zip file from the data subfolder and extract it right at the root of the data subfolder.

You should have the following subfolders:

data

   testing_images

   training_images

  test_results       <----- output of running the prediction.py


If for any any reason the data is inaccessible, you can also find a copy of the data here at https://drive.google.com/file/d/1ka0c6LqDLwovePRRzeuJFLx7AcXcz8Ji/view?usp=drive_link

Before running the script enter the following command in the terminal : pip install -r requirements.txt

Run these scripts in the following order by typing in the terminal  : python <fileName.py>

1) separateData.py
2) createModel.py
3) predict.py  (to run this last file type : python predict.py `<file_path>`)
