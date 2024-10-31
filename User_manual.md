# AI for sky camera weather precaution 
</br>
This file is the user manual to correctly use programs in this Github repository. All programs,test files, source code and models is included in this repository also. User may use pretrained model for sky predicition, or create own model.
However, if user wants/needs to include further processing or steps,<span style="color: red;"> <B><u>Please note that all steps in training must include in prediction step</u> </B> </span>. Otherwise, the model will not be able to predict
correctly.

# source code
In src folder, there're a few source files, which is develop for readability and adjustability of programs. There're mostly for development of testing in older development of the program (unsupervised learning) but function that is currently in use will add manual for
readability inside each function. Description of each file is
1. ClassPrediction.py : For prediction from dataset with given model and evaluation of models(Older version).
2. ConstructDataset.py : For concatenation of dataset with multiple subfolders.
3. feature_extraion.py : For calculation GLCM and statistical values of image without importing Skimage 
4. ModelTraining.py : <span style='color: red'>~~Currently unused~~</span>
5. preprocessing.py : Use for image preprocessing and load images.
6. TotalCalculation.py : Use for calculation of time and sunrise,sunset.

## Write_label.py
This programs is use for initially labeling image into classes. The method is to read sky image and evaluate sky in Oktas scale (See [WMO_oktas](URL "https://worldweather.wmo.int/oktas.html") for more information).
From this criterion, we can assign label to each image and classify it into classes.<br>
### sky classes
1. Clear : Bright blue sky. No or little cloud visible (0 - 12.5 Percents).
2. Partly cloudy : Little cloud visible.Around 2 - 3 parts of the sky cover in clouds (12.6 - 37.5 Percents).
3. Mostly cloudy : More cloud cover the sky. Around 4 - 5 parts of the sky cover in clouds (37.6 - 62.5 Percents).
4. Cloudy : Almost the whole sky is cover in clouds. Around 6 - 7 parts cover in clouds (62.6 - 87.5 Percents).
4. Overcast : The whole sky cover in clouds. All 8 parts of the sky cover in clouds (87.6 - 100 Percents).<br>

**Input** : Image directory<br>
**Output** : Classified image based on cloud percentages<br>
### Program details
<details>
<summary>Parameters</summary> 

| Parameter Name                | Description                                                                                           |
|-------------------------------|-------------------------------------------------------------------------------------------------------|
| `start_date`                 | The start date for the time duration calculation, formatted as 'YYYY-MM-DD'.                        |
| `end_date`                   | The end date for the time duration calculation, formatted as 'YYYY-MM-DD'.                          |
| `location`                   | A list containing latitude and longitude coordinates for the location (e.g., `[latitude, longitude]`). |
| `days`                        | The total number of days calculated between `start_date` and `end_date`.                            |
| `LSTM`                        | An instance of the `SunPosition` class for calculating the Local Solar Time Mean (LSTM).            |
| `EoT`                         | The Equation of Time, calculated based on the day number for solar position adjustments.            |
| `TC`                          | The Time Correction Factor calculated using longitude, LSTM, and EoT.                                |
| `dec`                         | The solar declination angle calculated for the specified day.                                       |
| `sunrise1`                   | The calculated sunrise time for the specified location and day.                                      |
| `sunset1`                    | The calculated sunset time for the specified location and day.                                       |
| `mask_directory`             | The file path to the mask image used for preprocessing input images.                                 |
| `classes_map`                | A dictionary mapping cloud classification labels to their corresponding folder names.               |
| `image_directory`            | The directory containing images to be processed and categorized.                                     |
| `output_directory`           | The directory where processed images will be saved based on classification.                          |
| `mode`                        | A string that specifies whether to process 'day' or 'night' images based on sunlight times.         |
| `percentage`                  | A list storing the calculated cloud coverage percentages for the classified images.                 |
| `classifier`                 | A list of classifications for the images based on their cloud coverage percentages.                  |


</details>
<details>
<summary>Functions</summary> 

| Function Name                  | Description                                                                                           |
|-------------------------------|-------------------------------------------------------------------------------------------------------|
| `timeConvertion().time_duration(start_date, end_date, include_end_date=True)` | Calculates the number of days between the specified start and end dates.                            |
| `SunPosition.LSTM(time_zone_offset)`        | Initializes the LSTM value based on the provided time zone offset for solar calculations.           |
| `SunPosition.calculate_EoT(day)`             | Calculates the Equation of Time for the given day to adjust solar time.                            |
| `SunPosition.TimeCorrectionFactor(Longitude, LSTM, EoT)` | Computes the Time Correction Factor based on location and solar calculations.                      |
| `SunPosition.declination(day)`               | Calculates the solar declination angle for the specified day.                                     |
| `SunPosition.DaytimeInfo(latitude, declination, TC)` | Determines sunrise and sunset times based on latitude, declination, and Time Correction Factor.     |
| `preprocessData().load_images_and_preprocess(path, mask, apply_crop_sun=False)` | Loads images from a specified path and preprocesses them using the given mask.                    |
| `prediction().CloudRatio(image, mask)`       | Calculates the cloud coverage ratio for a given image using the specified mask.                    |
| `prediction().classify_sky(image, ratio)`    | Classifies the sky condition of a given image based on the calculated cloud coverage ratio.        |
| `os.makedirs(target_folder, exist_ok=True)`   | Creates the target folder for storing classified images if it does not already exist.              |
| `cv2.resize(image, (1036, 705))`             | Resizes the given image to the specified dimensions (1036x705 pixels).                             |
| `cv2.imwrite(write_path, write_image)`        | Saves the processed image to the specified file path.                                             |

</details>

## Builddataset.py
This programs is use to generate dataset from image in each categoies. Read image in each categogies, Extract features from images (Textural feature and Spectral features) as list.
### Features 
<details>
<summary>Textural features</summary>

| Property       | Description                                                                                     |
|----------------|------------------------------------------------------------------------------------------------|
| `contrast`     | Measures the local intensity variation. High contrast values indicate significant intensity changes. |
| `dissimilarity`| Captures how different the pairs of pixels are; it increases with differences in gray levels.    |
| `homogeneity`  | Assesses how similar the pixel pairs are. Higher values indicate more uniform textures.         |
| `energy`       | Represents the sum of squared elements in the GLCM (Gray Level Co-occurrence Matrix), indicating texture uniformity. Higher energy values mean less texture complexity. |
| `correlation`  | Measures the linear dependency of gray levels in the image. High values indicate a predictable pattern. |
| `ASM`          | Also known as Angular Second Moment, it reflects the texture uniformity by summing the squared elements of the GLCM. Higher ASM indicates a more uniform texture. |
</details>
<details>
<summary>Spectral features</summary>

| Feature         | Description                                                                                   |
|-----------------|-----------------------------------------------------------------------------------------------|
| `intensity`     | Average intensity of the blue channel (`B`) across the image. It provides a measure of brightness. |
| `chan_b`        | Mean value of the blue channel (`B`). Useful for analyzing blue hues in the image.             |
| `chan_r`        | Mean value of the red channel (`R`). Useful for analyzing red hues in the image.               |
| `skewness`      | Skewness of the blue channel (`B`). It indicates the asymmetry in the pixel distribution of the channel. |
| `std`           | Standard deviation of the blue channel (`B`). Measures the spread or variability of pixel values in the channel. |
| `diff`          | Mean difference between the red channel (`R`) and the blue channel (`B`). Highlights color contrasts between red and blue tones. |
| `statistical`   | Combined features including skewness, standard deviation, difference, and mean values of red and blue channels for detailed statistical analysis. |

</details>

From this criterion, we can assign label to each image and classify it into classes.<br><br>
**Input** : Image directory<br>
**Output** : Classified image based on cloud percentages<br>
### Program details
<details>
<summary>Parameters</summary> 

| **Parameter**         | **Description**                                                                                       |
|-----------------------|-----------------------------------------------------------------------------------------------------|
| `sky_cam`              | User input for selecting configuration based on location (e.g., specific observatory location).     |
| `path`                 | Directory path to the folder containing images to be processed.                                     |
| `GLCM_param`           | User input list containing distance and angle parameters for the GLCM calculation.                  |
| `location`             | Coordinates of the selected sky camera location, loaded from configuration.                         |
| `time_zone`            | Time zone associated with the selected sky camera location, loaded from configuration.              |
| `start_date`           | The start date for the dataset, loaded from configuration.                                          |
| `mask_path`            | File path to the mask image for preprocessing images.                                               |
| `output_directory`     | Directory path to save the processed dataset CSV files.                                             |
| `properties`           | List of GLCM properties used for feature extraction (e.g., contrast, dissimilarity, etc.).         |
| `output_folder`        | Subdirectory path within the output directory for storing specific results based on GLCM parameters.|
| `folders`              | List of subdirectories within the main `path`, each representing a class of images.                 |
| `class_folder`         | Full path to a specific image class folder, containing images to be processed.                      |
| `GLCM`                 | List of computed GLCM features for the processed images.                                            |
| `Filename`             | List of image filenames that were processed.                                                        |
| `Intensity`            | List of intensity values calculated from the images.                                                |
| `skewness`             | List of skewness values calculated from the images.                                                 |
| `std`                  | List of standard deviation values calculated from the images.                                       |
| `diff`                 | List of difference values between channels calculated from the images.                              |
| `chan_r`               | List of mean red channel values calculated from the images.                                         |
| `chan_b`               | List of mean blue channel values calculated from the images.                                        |
| `images`               | Preprocessed images loaded from a single image path using a mask.                                   |
| `gray`                 | Grayscale converted versions of the preprocessed images.                                            |
| `glcm`                 | Computed GLCM features for a single grayscale image.                                                |
| `sky_cat`              | Name of the sky condition class derived from the folder name.                                       |
| `output_filename`      | Filename for the output CSV file that stores the dataset for a specific class.                      |
| `output_path`          | Full file path to where the CSV file will be saved.                                                 |




</details>
<details>
<summary>Functions</summary> 

| **Function Name**                                | **Description**                                                                                           |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `preprocessData().load_single_image`             | Loads a single image from a specified path, applies a mask, and optionally crops out the sun. Returns the preprocessed image and its name. |
| `preprocessData().computeGlcm`                   | Computes the Gray Level Co-occurrence Matrix (GLCM) for a grayscale image based on specified distance and angle parameters. Outputs GLCM features. |
| `Builddataset().Statistical`                     | Computes statistical metrics like mean intensity, skewness, standard deviation, and channel differences for a set of input images. |
| `preprocessData().getDataframe`                  | Constructs a DataFrame with GLCM properties, statistical features, and intensity for a set of images. The DataFrame is used to organize and save the dataset. |
| `os.makedirs`                                    | Creates directories if they do not exist, used for organizing logs and output folders.                     |
| `cv2.imread`                                     | Loads an image from a file, with an option to read in grayscale mode (used here to read mask images).      |
| `gc.collect`                                     | Clears memory by running garbage collection after processing a batch of images, ensuring efficient memory usage. |
| `json.load`                                      | Reads configuration data from a JSON file, providing settings like paths, parameters, and mask information for image processing. |
| `logging.info`                                   | Logs informational messages to a log file, documenting the progress and status of the dataset generation.  |
| `time.strftime`                                  | Formats the current date and time for use in file naming (e.g., timestamped log files).                    |
| `os.listdir`                                     | Retrieves a list of folders or files within a directory, used to iterate through image datasets.           |
| `cv2.cvtColor`                                   | Converts images from one color space to another, in this case, from RGB to grayscale.                      |
| `pd.DataFrame.to_csv`                            | Saves a DataFrame as a CSV file to a specified path, used for creating output datasets.                    |
                                         |

</details>

## Training.py
This programs is aim to train machine learning model using Python on Sklearn framework. Each run of model accuracy and validation score would have variability due to 
model inconsistancy and dataset size. Range of accuracy from a decent dataset could be around 85-98 %. <span style="color: red;"> <B><u>Please not that training dataset size and dataset quality have large effect on model accuracy.
</u> </B> </span> (See [sklearn documentation](URL "https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html") for more information).<br><br>

**Input** : Dataset<br>
**Output** : Model file (.onnx) and parameters<br>

### Features 
<details>
<summary>Textural features</summary>

| Property       | Description                                                                                     |
|----------------|------------------------------------------------------------------------------------------------|
| `contrast`     | Measures the local intensity variation. High contrast values indicate significant intensity changes. |
| `dissimilarity`| Captures how different the pairs of pixels are; it increases with differences in gray levels.    |
| `homogeneity`  | Assesses how similar the pixel pairs are. Higher values indicate more uniform textures.         |
| `energy`       | Represents the sum of squared elements in the GLCM (Gray Level Co-occurrence Matrix), indicating texture uniformity. Higher energy values mean less texture complexity. |
| `correlation`  | Measures the linear dependency of gray levels in the image. High values indicate a predictable pattern. |
| `ASM`          | Also known as Angular Second Moment, it reflects the texture uniformity by summing the squared elements of the GLCM. Higher ASM indicates a more uniform texture. |
</details>
<details>
<summary>Spectral features</summary>

| Feature         | Description                                                                                   |
|-----------------|-----------------------------------------------------------------------------------------------|
| `intensity`     | Average intensity of the blue channel (`B`) across the image. It provides a measure of brightness. |
| `chan_b`        | Mean value of the blue channel (`B`). Useful for analyzing blue hues in the image.             |
| `chan_r`        | Mean value of the red channel (`R`). Useful for analyzing red hues in the image.               |
| `skewness`      | Skewness of the blue channel (`B`). It indicates the asymmetry in the pixel distribution of the channel. |
| `std`           | Standard deviation of the blue channel (`B`). Measures the spread or variability of pixel values in the channel. |
| `diff`          | Mean difference between the red channel (`R`) and the blue channel (`B`). Highlights color contrasts between red and blue tones. |
| `statistical`   | Combined features including skewness, standard deviation, difference, and mean values of red and blue channels for detailed statistical analysis. |

</details>

From this criterion, we can assign label to each image and classify it into classes.<br><br>
**Input** : Image directory<br>
**Output** : Classified image based on cloud percentages<br>
### Program details
<details>
<summary>Parameters</summary> 

| **Parameter**         | **Description**                                                                                       |
|-----------------------|-----------------------------------------------------------------------------------------------------|
| `sky_cam`              | User input for selecting configuration based on location (e.g., specific observatory location).     |
| `path`                 | Directory path to the folder containing images to be processed.                                     |
| `GLCM_param`           | User input list containing distance and angle parameters for the GLCM calculation.                  |
| `location`             | Coordinates of the selected sky camera location, loaded from configuration.                         |
| `time_zone`            | Time zone associated with the selected sky camera location, loaded from configuration.              |
| `start_date`           | The start date for the dataset, loaded from configuration.                                          |
| `mask_path`            | File path to the mask image for preprocessing images.                                               |
| `output_directory`     | Directory path to save the processed dataset CSV files.                                             |
| `properties`           | List of GLCM properties used for feature extraction (e.g., contrast, dissimilarity, etc.).         |
| `output_folder`        | Subdirectory path within the output directory for storing specific results based on GLCM parameters.|
| `folders`              | List of subdirectories within the main `path`, each representing a class of images.                 |
| `class_folder`         | Full path to a specific image class folder, containing images to be processed.                      |
| `GLCM`                 | List of computed GLCM features for the processed images.                                            |
| `Filename`             | List of image filenames that were processed.                                                        |
| `Intensity`            | List of intensity values calculated from the images.                                                |
| `skewness`             | List of skewness values calculated from the images.                                                 |
| `std`                  | List of standard deviation values calculated from the images.                                       |
| `diff`                 | List of difference values between channels calculated from the images.                              |
| `chan_r`               | List of mean red channel values calculated from the images.                                         |
| `chan_b`               | List of mean blue channel values calculated from the images.                                        |
| `images`               | Preprocessed images loaded from a single image path using a mask.                                   |
| `gray`                 | Grayscale converted versions of the preprocessed images.                                            |
| `glcm`                 | Computed GLCM features for a single grayscale image.                                                |
| `sky_cat`              | Name of the sky condition class derived from the folder name.                                       |
| `output_filename`      | Filename for the output CSV file that stores the dataset for a specific class.                      |
| `output_path`          | Full file path to where the CSV file will be saved.                                                 |




</details>
<details>
<summary>Functions</summary> 

| **Function Name**                                | **Description**                                                                                           |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `preprocessData().load_single_image`             | Loads a single image from a specified path, applies a mask, and optionally crops out the sun. Returns the preprocessed image and its name. |
| `preprocessData().computeGlcm`                   | Computes the Gray Level Co-occurrence Matrix (GLCM) for a grayscale image based on specified distance and angle parameters. Outputs GLCM features. |
| `Builddataset().Statistical`                     | Computes statistical metrics like mean intensity, skewness, standard deviation, and channel differences for a set of input images. |
| `preprocessData().getDataframe`                  | Constructs a DataFrame with GLCM properties, statistical features, and intensity for a set of images. The DataFrame is used to organize and save the dataset. |
| `os.makedirs`                                    | Creates directories if they do not exist, used for organizing logs and output folders.                     |
| `cv2.imread`                                     | Loads an image from a file, with an option to read in grayscale mode (used here to read mask images).      |
| `gc.collect`                                     | Clears memory by running garbage collection after processing a batch of images, ensuring efficient memory usage. |
| `json.load`                                      | Reads configuration data from a JSON file, providing settings like paths, parameters, and mask information for image processing. |
| `logging.info`                                   | Logs informational messages to a log file, documenting the progress and status of the dataset generation.  |
| `time.strftime`                                  | Formats the current date and time for use in file naming (e.g., timestamped log files).                    |
| `os.listdir`                                     | Retrieves a list of folders or files within a directory, used to iterate through image datasets.           |
| `cv2.cvtColor`                                   | Converts images from one color space to another, in this case, from RGB to grayscale.                      |
| `pd.DataFrame.to_csv`                            | Saves a DataFrame as a CSV file to a specified path, used for creating output datasets.                    |
                                         |

</details>