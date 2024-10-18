import os
import time
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import json
from src.ConstructDataset import Builddataset
# Set up logging
log_dir = "logs/Training_log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"model_training_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Start timer
start = time.time()

# Dataset loading
dataset_path = input('Enter dataset path: ')
logging.info(f"Dataset path entered: {dataset_path}")
df = list()

# List folders and build the dataset
folders = os.listdir(dataset_path)
for folder in folders:
    subfolder = os.path.join(dataset_path, folder)
    logging.info(f"Processing folder: {subfolder}")
    df.append(Builddataset().concateDataset(folder_name=subfolder))

# Concatenate all dataframes
dataframe = pd.concat(df)
logging.info("---DataFrame construction completed---")
print("---DataFrame construction completed---")

# Feature and label extraction
raw_data = dataframe.drop(columns=['label'])
label = dataframe['label']

# Scaler operation
scaler = StandardScaler()
scaler.fit(raw_data)
logging.info(f"Scaler fit completed. Mean: {scaler.mean_}, Scale: {scaler.scale_}")

# Transform the raw data
X = scaler.transform(raw_data)
y = label

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
logging.info("Data split into training and test sets completed.")

# Train RandomForest Classifier (optional)
clf = RandomForestClassifier(n_estimators=200, criterion='entropy')
clf.fit(X_train, y_train)
logging.info("---Model training completed---")
print("---Model training completed---")

# Log and print model parameters
logging.info(f"RandomForest Parameters: {clf.get_params()}")
print(f"RandomForest Parameters: {clf.get_params()}")

# Estimate validation score
scores = cross_val_score(estimator=clf, X=X_train, y=y_train)
logging.info(f"Cross-validation scores: {scores}")
logging.info(f"Mean accuracy: {scores.mean()}")
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean()}")

# Execution time logging and print
execution_time = time.time() - start
logging.info(f"Execution time: {execution_time} seconds")
print(f"Execution time: {execution_time} seconds")

# Save model and scaler if requested
save = input("Save model? (Y/n): ")
if save.lower() == 'y':
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = skl2onnx.convert_sklearn(clf, initial_types=initial_type, target_opset=12)
    
    model_path = r"C:\Users\ASUS\Documents\NARIT_internship_2024\NARIT_internship_2024\models\Classification_model\classifier_model.onnx"
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logging.info(f"Model saved to: {model_path}")
    print(f"Model saved to: {model_path}")

    # Save scaler parameters
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    scaler_path = r'C:\Users\ASUS\Documents\NARIT_internship_2024\NARIT_internship_2024\models\Scaler\scaler_params.json'
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=4)
    logging.info(f"Scaler parameters saved to: {scaler_path}")
    logging.info(f"Scaler parameters: {scaler_params}")  # Log scaler parameters
    print(f"Scaler parameters saved to: {scaler_path}")
    print(f"Scaler parameters: {scaler_params}")

# Clear console option
if input("Clear console? (Y/n): ").lower() == 'y':
    os.system('cls')
    logging.info("Console cleared")
    print("Console cleared")

logging.info("Process finished.")
print("Process finished.")
