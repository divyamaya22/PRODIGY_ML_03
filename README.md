# PRODIGY_ML_03
Pet Breed Classification Using CNN
📌 Objective
This project aims to classify different pet breeds using Convolutional Neural Networks (CNNs). By training a deep learning model on the Oxford-IIIT Pet Dataset, the system can identify various dog and cat breeds from images.

📂 Dataset
The dataset consists of 7,394 images of 37 different pet breeds, each labeled accordingly. The images include various poses, lighting conditions, and occlusions, making it a great dataset for training a robust classifier.

🏗 Project Workflow
Data Preprocessing

Load and clean the dataset (remove corrupted images).
Resize images to 128×128 pixels for model compatibility.
Normalize pixel values for better model performance.
Model Architecture (CNN)

Convolutional layers extract features from pet images.
MaxPooling layers reduce spatial dimensions.
Fully connected layers classify the breed.
Softmax activation provides probability scores for each breed.
Training & Optimization

Loss function: Categorical Cross-Entropy
Optimizer: Adam
Metrics: Accuracy
Model Testing & Evaluation

Validate on test images to measure accuracy.
Use sample images to verify classification performance.
Testing on New Images

Load external pet images.
Preprocess and classify them using the trained model.
🚀 Key Features
✅ Uses CNN for high-accuracy pet classification.
✅ Processes new images and predicts breed labels.
✅ Handles corrupted images gracefully.
✅ Model can be improved with data augmentation for better generalization.

🔧 Technologies Used
Python
TensorFlow / Keras
OpenCV & PIL (Image Processing)
NumPy & Matplotlib
