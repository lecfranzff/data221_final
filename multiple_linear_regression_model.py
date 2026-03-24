# Dataset cleaning

import pandas
from keras.src.losses import mean_absolute_error

dataframe = pandas.read_csv("car details v4.csv")

# Trim leading/trailing spaces from all text data
for column in dataframe.select_dtypes(include = "string"):         # Find all text columns in dataset
    dataframe[column] = dataframe[column].str.strip()

# Convert Price from Indian Rupees to CAD
indian_rupees_to_cad = 0.01525                  # Conversion rate as of Dec 31, 2025
dataframe["Price"] = dataframe["Price"] * indian_rupees_to_cad

# Remove unnecessary columns
dataframe = dataframe.drop(columns = [
    "Color",
    "Max Power",
    "Max Torque",
    "Length",
    "Width",
    "Height"
])

# Clean Engine column by convert values like "1197 cc" to a float value = 1197
clean_engine = []

for value in dataframe["Engine"]:
    if pandas.notna(value):                                 # Check for missing value
        value = str(value)                                  # Convert to string just in case
        value = value.replace("cc", "")         # Remove "cc"
        value = value.strip()                               # Remove extra spaces
        clean_engine.append(float(value))                   # Add number to the list
    else:
        clean_engine.append(value)                          # Adds missing values back into the list

dataframe["Engine"] = clean_engine

# Remove records with missing Drivetrain values
dataframe = dataframe.dropna(subset = ["Drivetrain"])

# Impute missing values with median
# Seating Capacity
median_seats = dataframe["Seating Capacity"].median()
dataframe["Seating Capacity"] = dataframe["Seating Capacity"].fillna(median_seats)      # Fills empty values with median

# Fuel Tank Capacity
median_fuel = dataframe["Fuel Tank Capacity"].median()
dataframe["Fuel Tank Capacity"] = dataframe["Fuel Tank Capacity"].fillna(median_fuel)   # Fills empty values with median

# Remove outliers using IQR
columns_with_outliers = [           # List of columns we want to remove outliers from
    "Price",
    "Kilometer",
    "Engine",
    "Seating Capacity",
    "Fuel Tank Capacity",
]

for column in columns_with_outliers:

    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataframe = dataframe[
        (dataframe[column] >= lower_bound) &
        (dataframe[column] <= upper_bound)
        ]

# Save cleaned dataset
dataframe.to_csv("cleaned_car_details_dataset.csv", index = False)

print("Cleaning complete. File saved.")



# Multiple Linear Regression Model

import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

dataset = pandas.read_csv("cleaned_car_details_dataset.csv")

dataset = dataset.drop(columns = ["Model"])         # Too many different models, overfitting

dataset = pandas.get_dummies(dataset, drop_first = True)        # Converts categorical variables to binary

# print(dataset.head())       # Testing clean dataset

X = dataset.drop("Price", axis = 1)
y = dataset["Price"] / 1000         # Scale price to thousands

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
mean_absolute_error = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mean_absolute_error)

print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")