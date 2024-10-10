# AI for sky camera weather precaution.
## Abstract
This project is cooperative project with NARIT. The project is assigned to create machine learning model that have capability of making prediction of sky in 

## classes
we have seperate sky condition to 5 statuses.

    1. Clear : No cloud visible or slight cloud covered around 0-1 / 8 parts of the sky.
    2. Partly cloudy : Scatter Cumulus cloud or slight high cloud cover the sky covered around 2-3 / 8 parts of the sky.
    3. Mostly cloudy : More cloud covered in the sky, bigger Nimbus cloud or big cumulus covered around 4-5 / 8 parts of the sky.
    4. Cloudy : Most sky is covered, only slight part of the sky visible, cloud covered around 6-7 / 8 parts of the sky.
    5. Overcast : Cloud covered the whole sky, no visible sky in sight, cloud covered 8/8 parts of the sky.

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
new mask. Default masks that included in folder mask are>

##### international observatories
    1. `mask_Australia.png` : Springbook observatory, Springbook, Australia.
    2. `mask_Chile.png` : PROMPT8 (CTIO), Coquimbo, Chile.
    3. `mask_US.png` : Sierra remote observatories, California, USA.
    4. `mask_China.png` : Gao Mei Gu observatory, Lijiandg, China.

##### Domestic observatory (Thailand)
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