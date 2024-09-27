Hello, My name is Prachaya Makeragsakij from King Mongkut's institute of technology Ladkrabang

    This is my internship project currently working in 2024 with NARIT.
This project aim to create a working algorithm that is able to predict and classified of sky in each conditions
we have seperate sky condition to 5 statuses.

    1. Clear : No cloud visible or slight cloud covered around 0-1 / 8 parts of the sky.
    2. Partly cloudy : Scatter Cumulus cloud or slight high cloud cover the sky covered around 2-3 / 8 parts of the sky.
    3. Mostly cloudy : More cloud covered in the sky, bigger Nimbus cloud or big cumulus covered around 4-5 / 8 parts of the sky.
    4. Cloudy : Most sky is covered, only slight part of the sky visible, cloud covered around 6-7 / 8 parts of the sky.
    5. Overcast : Cloud covered the whole sky, no visible sky in sight, cloud covered 8/8 parts of the sky.

These condition will use in data labeling and others operation. But since the model will be created, we can leave prediction to the model for predict. (Explain later)

## Methodology

    I've been working on the machine learning model, which I aim to present scalability and customability
While I previously create an unsupervised model based on dataset generated from raw image data.
I created some important method for further training and testing to evaluation of dataset in training and testing model, heres step of completing the model 

    1. Execute `Write_label.py` to generate label.
    2. Visual inspect image from each folder, delete defect image.
    3. Use image folder path to train the model in `training.py`.
    4. Use model from `training.py` then to use it in prediction.

## Prinmary source
    For all of this to happen, I would like to thanks to Scikit-learn developing team and Opencv team, who contribute many valuable
methods for the program and create other functions

## Disclaimers
    This project is cross-language project. Since the model will be firstly deployed in Python, then use the deployed model in 
C/C++, the model need feature extraction as we trained (I will provide this part in C)
    Since the model use the different approach to result of 5 classes, the process needs additional feature extraction which is 
not traditional CNN. I did think about feature extract using end-end Neural network, but since neural network model is also 