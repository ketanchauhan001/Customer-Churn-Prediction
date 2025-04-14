# Customer-Churn-Prediction

Developed a deep learning model using an Artificial Neural Network (ANN) to predict customer churn for a bank. The objective was to identify customers who are likely to leave the bank, enabling proactive retention strategies.

Key Features:
Data Preprocessing: Handled missing values and removed duplicates to ensure data quality. Applied LabelEncoder to encode categorical variables and StandardScaler to normalize feature values.
Feature Selection: Selected key features such as Credit Score, Age, Gender, Tenure, Balance, Number of Products, Credit Card status, Active Membership, and Estimated Salary.

Model Architecture:
Built a Sequential ANN with two hidden layers.
Used ReLU activation functions and Dropout layers to prevent overfitting.
Output layer used Sigmoid activation to handle binary classification.

Training & Optimization:
Used the Adam optimizer and binary_crossentropy loss function.
Implemented EarlyStopping to halt training when validation loss stopped improving.
Split data into training, validation, and testing sets using an 80/20 ratio.

Evaluation:
Achieved 85% accuracy on test data and ~YY% on training data (replace with your actual scores).
Model performance evaluated using accuracy metrics and prediction thresholding.
Interactive Prediction: Implemented real-time user input for prediction, allowing users to enter customer data and receive immediate churn likelihood results.
