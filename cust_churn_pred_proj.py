import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load and clean data
data = pd.read_csv(r"C:\Users\ketan\ketan_python\csv files\Churn_Modelling.csv")
data.drop_duplicates(inplace=True)

# Encode categorical column
lc = LabelEncoder()
data['Gender'] = lc.fit_transform(data['Gender'])

# Select features and label
x = data[['CreditScore','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']]
y = data['Exited']

# Standardize the features
sc = StandardScaler()
x = pd.DataFrame(sc.fit_transform(x), columns=x.columns)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build the model with Dropout layers
ann = Sequential()
ann.add(Dense(64, input_dim=9, activation='relu'))     # Increased neurons
ann.add(Dropout(0.3))                                  # Dropout to prevent overfitting
ann.add(Dense(32, activation='relu'))
ann.add(Dropout(0.3))
ann.add(Dense(1, activation='sigmoid'))

# Compile the model
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
ann.fit(x_train, y_train, batch_size=100, epochs=50, validation_split=0.2, callbacks=[early_stop])


print(data.head(10).to_string())

credit_score = float(input("Credit Score: "))
gender = input("Gender (Male/Female): ").strip()
gender_encoded = lc.transform([gender])[0]  # Encode gender using same LabelEncoder
age = int(input("Age: "))
tenure = int(input("Tenure: "))
balance = int(input("Balance: "))
num_products = int(input("Number of Products: "))
has_card = int(input("Has Credit Card (1=Yes, 0=No): "))
is_active = int(input("Is Active Member (1=Yes, 0=No): "))
salary = int(input("Estimated Salary: "))

user_input = [[credit_score, gender_encoded, age, tenure, balance,num_products, has_card, is_active, salary]]
user_input_scale=sc.transform(user_input)
prediction=ann.predict(user_input_scale)

for i in prediction:
    if(i[0]>0.5):
        print("Prediction: Customer is liked to exit. [1]")
    else:
        print("Prediction: Customer is not liked to exit. [0]")


# Predictions
y_pred = ann.predict(x_test)
x_pred = ann.predict(x_train)

# Convert probabilities to binary (0 or 1)
prd_data = [1 if i > 0.5 else 0 for i in y_pred]
prd_data2 = [1 if i > 0.5 else 0 for i in x_pred]

# Accuracy calculation
acc_test = accuracy_score(y_test, prd_data) * 100
print(f"Test Accuracy: {acc_test:.2f}%")

acc_train = accuracy_score(y_train, prd_data2) * 100
print(f"Train Accuracy: {acc_train:.2f}%")

