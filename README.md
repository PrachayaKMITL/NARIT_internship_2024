# AI for sky camera weather precaution.
## Abstract
This project is a collaboration with NARIT to develop a machine learning model capable of predicting sky conditions across five classes, ranging from clear to overcast. We have selected the Random Forest algorithm for classification. For feature extraction, we utilize the Gray Level Co-occurrence Matrix (GLCM) and statistical values, resulting in a DataFrame with 12 features. To enhance model accuracy, the datasets undergo scaling and normalization. The model achieves an impressive accuracy of approximately 99.28% for daytime predictions and around 96.04% for nighttime predictions. For practical use, the model requires the same feature extraction and scaling processes employed during training.
## image data
Image data is provided by NARIT's sky camera system. Each image was taken by sky camera every 2-3 minutes. Camera resolution may varies by system setting and configurtion. Each image filename is tick value, which can then convert to datetime for futher usage.
<p align="center">
  <img src="output_png\638543809344295124.png" alt="Sample sky camera Image" width="500"/>
  <br/>
  <b><u>Figure : Sky camera image</u></b> 
</p>

## classes
we have seperate sky condition to 5 statuses.

1. Clear : No cloud visible or slight cloud covered around 0-1 / 8 parts of the sky.
<p align="center">
<img src="output_png\638531421743182513.png" alt="Sample sky camera Image" width="500"/>
<br/>
<b><u>Figure : Clear sky example</u></b> 
</p>

2. Partly cloudy : Scatter Cumulus cloud or slight high cloud cover the sky covered around 2-3 / 8 parts of the sky.
<p align="center">
<img src="output_png\638529697331896570.png" alt="Sample sky camera Image" width="500"/>
<br/>
<b><u>Figure : Partly cloudy example</u></b> 
</p>

3. Mostly cloudy : More cloud covered in the sky, bigger Nimbus cloud or big cumulus covered around 4-5 / 8 parts of the sky.
<p align="center">
<img src="output_png\638528996505324368.png" alt="Sample sky camera Image" width="500"/>
<br/>
<b><u>Figure : Mostly cloudy example</u></b> 
</p>

4. Cloudy : Most sky is covered, only slight part of the sky visible, cloud covered around 6-7 / 8 parts of the sky.
<p align="center">
<img src="output_png\638529065537845343.png" alt="Sample sky camera Image" width="500"/>
<br/>
<b><u>Figure : Cloudy example</u></b> 
</p>

5. Overcast : Cloud covered the whole sky, no visible sky in sight, cloud covered 8/8 parts of the sky.
<p align="center">
<img src="output_png\638528938276605125.png" alt="Sample sky camera Image" width="500"/>
<br/>
<b><u>Figure : Overcast example</u></b> 
</p>

These condition will use in data labeling and others operation. But since the model will be created, we can leave prediction to the model for predict. (Explain later)
## Definition
In this project, some definition is marked for general commandment and agreements for users to understand.
since most of components and objects used in this project are tecnically specific to astronomical and meteorolocgical terms.
lists of important definition willl be mark down below. 

### Programming definitions
1. Sky camera images : image capture from sky camera. Use for dataset creation, validation and Model testing.
2. Masking : Computer vision operation of obstacle deletion from an image, use `Opencv-Python`
3. GLCM : Stands for Gray Level Cooccurance Matrix, is feature extraction algorithm which extract value by comparing two adjecent pixels with given distance and angle value.
4. Statistic : Mean, skewness and standard deviation calculate from an image or images. Especially Red and Blue channel of image. 
5. Feature extraction : Meaningful values extracted from an image. including GLCM and statical value from an image.
6. Model : Mathematical model from machine learning algorithms. Use for classification sky in classesfrom class 0-4.

### Scientific definitions
1. Sky camera : Camera modules which have capability to capture image of 180 degrees from horizon to zenith angle of 90 degrees.
2. Oktas : Oktas is standard cloud coverage measurements unit. Oktas divide sky into 8 parts and measure cloud coverage by eye or measuring device out of 8 part.
3. Daytime : Daytime in this project refers to time duration after sunrise and before sunset. This time duration needed for seperation between day and nightime.
4. Zenith : The zenith angle is the angle between the vertical direction (directly overhead) and the line of sight to the sun or another celestial object.     

## Methodology
I've been working on the machine learning model, which I aim to present scalability and customability
While I previously create an unsupervised model based on dataset generated from raw image data.
I created some important method for further training and testing to evaluation of dataset in training and testing model, heres step of completing the model 

1. Execute `Write_label.py` to generate label.
2. Visual inspect image from each folder, delete defect image.
3. Use image folder path to train the model in `training.py`.
4. Use model from `training.py` then to use it in prediction.

## Dependencies
All dependencies of this project has been written in `requirement.txt` user can install all dependencies using <code>pip install -r requirement.txt</code>.
User may use higher or lower version of libraries. However, `onnxruntime` and `skl2onnx` library should preserve as marked version.

## Disclaimers
This project is cross-language project. Since the model will be firstly deployed in Python, then use the deployed model in 
C/C++, the model need feature extraction as we trained (I will provide this part in C)
Since the model use the different approach to result of 5 classes, the process needs additional feature extraction which is 
not traditional CNN. I did think about feature extract using end-end Neural network, but since neural network model is also need \
a little bit of preprocessing that data before and give quite the same output this model could be a little bit cheaper considering 
training customability and reduce time complexity.

## Usage
I've written some important programs for generating and creating of the dataset and training. For generating dataset, the step will be explain 
below. 

1. Categorize images by running `Write_label.py`, then program will write labels for each categories by categorizing it.
    However, user may need to recheck each folder to ensure categorization correctness.
2. After finish inspection, user may run programm `Buildataset.py` to generate dataset to the folder. Programm instruction 
    was written inline in the program itself.
3. Using generated dataset to train the model(The model is interchangable, see Sklearn documentation for others model).
    and scaling before training. The model will fit the data and give out probability.
4. After training, program will display evaluation scores and ask user to save the model. User can retrain the model or
    save the model as .onnx files as a globally support machine learning model file type.
5. User can test `Model.onnx` on `Test_model.py` by include path of the model. Run the code from onnxruntime and inspect
    output
After done folloing steps, user may use the model in other languages. In python, the model will be loaded using `onnxruntime` Python library.
For C/C++, user may need to install libraries in that languages.

## Masks
In folder 'masks', are mask generated from `Mask_creator.py` that included both thresholding methods and manual masking method.
both is able to generate decent quality of mask. If image cannot perform a good performance, user may need to consider creating a 
new mask. Default masks that included in folder mask are

#### **international observatories**
1. `mask_Australia.png` : Springbook observatory, Springbook, Australia.
2. `mask_Chile.png` : PROMPT8 (CTIO), Coquimbo, Chile.
3. `mask_US.png` : Sierra remote observatories, California, USA.
4. `mask_China.png` : Gao Mei Gu observatory, Lijiandg, China.

#### **Domestic observatory (Thailand)**
1. `mask_Astropark.png` : Princess Siridhorn Astropark, Maerim, Chiangmai.
2. `mask_TNO.png` : Thai National Observatory, Chomtong, Chiangmai.
3. `mask_Khonkaen.png` : Khonkaen Observatory, Ubolratana, Khonkaen.
4. `mask_Korat.png` : Nakhonratchasima Observatory, Muang, Nakhon Ratchasima.
5. `mask_Chachoengsao.png` : Chachoengsao Observatory, Plaeng Yao, Chachoengsao.
6. `mask_Songkla.png` : Songkla Observatory, Muang, Songkla.

These are complete list of observatory with sky camera data available.However, if there's any change in camera placement or 
field of view, user may need to create a newer version of the mask.

## Example in C
from all these steps, the model will be create and test in Python environment. However in real application, model may be use in varieties of scenario and environment. As mention above, 
.onnx models are versatile for programming languages. So, I will conduct a test of prediction
model in C to evaluate model performances and programming constrain in C.

## Models 
In models folder, I created machine learning model especially for astropark, Chiangmai. The model acheives ~99.28 % accuracy for Day model and ~96.04% for night model. Since the model was generated from Astropark specific dataset, the model would not be versatile for other camera configuration or additional system (**Please be aware on this**)
<p align="center">
  <img src="output_png\confusion_mat_day.png" alt="Sample Image" width="500"/>
  <br/>
  <b><u>Figure : Confusion matrix for Daytime</u></b> 
</p>

<p align="center">
  <img src="output_png\confusion_mat_night.png" alt="Sample Image" width="500"/>
  <br/>
  <b><u>Figure : Confusion matrix for Nighttime</u></b> 
</p>

## Prediction
Since this model is trained with scaling method to ensure model performance, scaling model also save into `scaler_params.csv` and `scaler_params.txt` to use in C or other languages. Since default Python filename is not suitable in other languages. User may further develop model and be using other normalization/scaling feature in the future but please be aware of unsupport language which may not support model file<br><br>
<b>Prediction of the model</b> requires feature extraction programs and onnxruntime library. This can be done by installation of onnxruntime library in that language (if not Python). The model export to .onnx file can perform without remarkable error in the program and prediction deficiency. 

<b>In this repository</b>, I include the `scaler_param.py` and `Astropark_day.onnx` which is scaler parameter files and model file. To use the model file, user need to scale all parameter down with this following formula.

### Standard Scaler

$$
\huge Z = \frac{(x-\mu)}{\sigma}
$$

$$
\large Z = Z \ score  \\   
x = input \ data \\
\mu = mean \ value \\
\sigma = standard \ deviation \\
$$

This formula is a standard deviation formula to convert large scaler data to Z-score value with respect to mean and stadard deviation of each feature. All function for calculation of this Z-score value is included in `Test/Test_C/header folder` 

### Classification model
As mention earlier, classification model that use in this project is RandomForest classifier. This classifier has capability of prediction in probability of each class, which will be compared with other class probability and find the highest. Thus, model can be describe by .onnx graph <b>like this</b><br>
<p align="center">
  <img src="output_png\Model_map.png" alt="Sample Image" width="500"/>
  <br/>
  <b><u>Figure : RandomForest classifier structure graph</u></b> 
</p>
<br>As you can see the model can gives out both probability and output label. User can use output_probability to calculate percentage of each class (eg. Clear 60 percent) or else.<br>
