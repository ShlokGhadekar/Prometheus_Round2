# Prometheus_Round2
This repository is made for file submission of prometheus tech team recruitments round 2(software domain)

TASK 1
Ball Detection Dataset and Model Training

Methodology

1. Dataset Creation

Roboflow Dataset Generation

The Ball Detection dataset was created using Roboflow by uploading and annotating images.

Bounding boxes were used to mark the balls in the images for precise annotation.

The dataset was limited to 20 images, ensuring high-quality annotations for each image.

The dataset was split into training (13 images) and validation (4 images) sets to evaluate model performance.

Exporting the Dataset

Once annotation was completed, the dataset was exported in the YOLOv8 format.

The dataset included .txt files for labels and .jpg files for images, following YOLO’s format (class, x_center, y_center, width, height).

2. Model Training

Environment Setup

Google Colab was used to leverage free GPU resources for model training.

YOLOv8 Setup

The YOLOv8 model was selected due to its efficiency and accuracy in object detection tasks.

Training Configuration

The model was trained using YOLOv8’s built-in functionalities.

Configurations included input size, batch size, and epochs for optimized training.

The Adam optimizer was used to adjust model weights during training.

Both training and validation data were used to monitor performance.

3. Model Evaluation and Fine-Tuning

Metrics

Key metrics monitored included mean average precision (mAP) and Intersection over Union (IoU).

These metrics helped determine how well the model was detecting objects (balls) in images.

Hyperparameter Tuning

If the validation performance was unsatisfactory, hyperparameters such as learning rate and batch size were adjusted and retraining was performed.

4. Inference and Testing

Model Inference

The trained model was tested on new, unseen images to assess its generalization capabilities.

Test images were processed through the model, and predictions were visualized with bounding boxes.

Prediction Results

The output included bounding box coordinates and class labels (ball).

The model successfully detected balls in test images, confirming effective training.

5. Model Deployment

Model Saving

The trained model was saved in a .pt format (PyTorch), making it suitable for future deployment.

Download and Usage

The model was downloaded for future use in real-time applications or further testing.

6. Future Improvements

Dataset Expansion

Increasing the dataset size would enhance the model’s ability to generalize in real-world scenarios.

Data Augmentation

Techniques such as flipping, rotation, and scaling could improve dataset diversity.

Hyperparameter Tuning

Further adjustments in learning rate, batch size, and epochs could enhance model accuracy.

This methodology ensures a structured approach to dataset creation, model training, evaluation, and deployment for ball detection using YOLOv8.

Task 2: Player Rating Prediction

Methodology and Findings

1. Data Exploration

The first step in the analysis was to explore the available data in the European Soccer Database. The database contains multiple tables, including information about players, matches, teams, and leagues. The focus was on the Player_Attributes table, which provides details such as player ratings, attributes, and related information.

The dataset structure was analyzed by checking available columns, identifying missing values, and removing duplicates.

Summary statistics were computed for key variables such as player rating, potential, and physical attributes (height, weight, etc.) to gain insights into the data distribution.

2. Visualizations

Three key visualizations were generated to better understand the data:

Age Distribution: A histogram of player ages was created to analyze the age demographics and understand if younger or older players tend to have better ratings.

Rating Distribution: A histogram of overall player ratings helped identify any skewness or imbalance in the rating data.

Correlation Heatmap: A heatmap of feature correlations helped determine the relationships between different player attributes (such as passing, dribbling, and shooting) and their potential impact on player ratings.

These visualizations provided meaningful insights and guided the feature selection process.

Machine Learning Part: Player Rating Prediction

Feature Selection Justification

The following features were chosen based on their potential influence on player ratings:

Age: Older players might have more experience, which could influence their ratings.

Height and Weight: These physical attributes can affect performance, especially in aspects such as strength and stamina.

Value: A player’s market value often reflects their overall skill and performance level.

These features were selected as they likely have a direct or indirect impact on the player’s overall rating.

Model Choice Explanation

A Linear Regression model was chosen for predicting player ratings due to the following reasons:

Simplicity: Linear regression is an effective baseline model for regression tasks where relationships between features and the target variable are assumed to be linear.

Interpretability: The model coefficients provide insights into how each feature influences the predicted player rating.

Given that the task involves predicting a continuous variable (overall rating), linear regression was an appropriate choice.

Basic Performance Metrics

After training the model, the following performance metrics were recorded:

Mean Absolute Error (MAE): The MAE was 1.58, indicating that, on average, the model’s predictions were off by 1.58 rating points.

R-squared (R²): The R² value was 0.466, meaning that approximately 46.6% of the variance in player ratings was explained by the chosen features. While reasonable, this suggests that additional features could improve the model’s performance.

Conclusion

This analysis provided valuable insights into the factors influencing player ratings in the European Soccer Database. The predictive model demonstrated that player ratings can be estimated with reasonable accuracy using a simple linear regression approach. However, further improvements can be made by:

Incorporating additional relevant features such as passing, dribbling, and defensive attributes.

Experimenting with more advanced models such as Decision Trees or Neural Networks.

Applying feature engineering techniques to better capture player performance characteristics.

This project successfully showcased a structured approach to player rating prediction using data exploration, visualization, and machine learning.



