## AI for sky camera weather precaution 
</br>
This file is the user manual to correctly use programs in this Github repository. All programs,test files, source code and models is included in this repository also. User may use pretrained model for sky predicition, or create own model.
However, if user wants/needs to include further processing or steps,<span style="color: red;"> <B><u>Please note that all steps in training must include in prediction step</u> </B> </span>. Otherwise, the model will not be able to predict
correctly.

## Write_label.py
This programs is use for initially labeling image into classes. The method is to read sky image and evaluate sky in Oktas scale (See [WMO_oktas](URL "https://worldweather.wmo.int/oktas.html") for more information).
From this criterion, we can assign label to each image and classify it into classes