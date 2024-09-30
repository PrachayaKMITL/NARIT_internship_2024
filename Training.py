import onnx
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix
from skl2onnx import to_onnx
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import os
import pandas as pd
from ConstructDataset import Builddataset
import json
import time

#Enter dataset path | Folder with subfolder 
'''
Folder structure :

Folder 
    |- Subfolder 1 | Ex. GLCM [2,0]
        |- file1.csv
        |- file2.csv
        |- file3.csv
    |- Subfolder 2 | Ex. GLCM [1,0] (Optional)
    .
    .
    |- Subfolder n

'''
start = time.time()
dataset_path = input('Enter dataset path : ')
df = list()
folders = os.listdir(dataset_path)
for folder in folders:
    subfolder = os.path.join(dataset_path,folder)
    df.append(Builddataset().concateDataset(folder_name=subfolder))
dataframe = pd.concat(df)
print("---Done construct dataframe---")
'''
This part is changable, see Sklearn documentation for more information.
However, each scaler/normalizer model has different features and parameters
So, each scaler might need different paramter to export.
'''
raw_data = dataframe.drop(columns=['label'])
label = dataframe['label']
#Scaler operation
scaler = StandardScaler()
scaler.fit(raw_data)
#transform to a new data
X = scaler.transform(raw_data)
y = label
'''
Split data into training and testing for the dataframe
Parameter canbe adjusted | test_size and train_size (See Sklearn.model_selection for more information)
'''
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
#training RandomforestClassifier (optional)
clf = RandomForestClassifier(n_estimators=200,criterion='entropy')
clf.fit(X_train,y_train)
print("---Fitting model---")
#Estimate validation score 
scores = cross_val_score(estimator=clf,X=X_train,y=y_train)
print("Cross-validation scores : ", scores)
print("Mean accuracy : ", scores.mean())
print("Execution time : ",time.time() - start)
'''
Save model in .onnx (Open Neural Network Exchange) 
For the use in C please refer to www.onnx.ai for onnxruntime and skl2onnx
'''
save = input("Save model? (Y/n) : ")
if (save == 'Y') or (save == 'y'):
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = skl2onnx.convert_sklearn(clf, initial_types=initial_type, target_opset=12)
    with open(r"C:\Users\ASUS\Documents\NARIT_internship_2024\NARIT_internship_2024\models\Classification_model\classifier_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    scaler_params = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist()
}
    with open(r'C:\Users\ASUS\Documents\NARIT_internship_2024\NARIT_internship_2024\models\Scaler\scaler_params.json', 'w') as f:
        json.dump(scaler_params, f, indent=4)
if (str(input("Clear console ? (Y/n)")) == 'Y') or (str(input("Clear console ? (Y/n)")) == 'y'):
    os.system('cls')


