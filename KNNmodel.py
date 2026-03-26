import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

#load data
data_frame = pandas.read_csv("car details v4.csv")

#convert indian rupees to cad
rupee_to_cad = 0.01525
data_frame["Price"] = data_frame["Price"] * rupee_to_cad

#remove columns that aren't needed
data_frame = data_frame.drop(columns = ["Color", "Length", "Width", "Height", "Max Power", "Max Torque"])

#remove "cc" from engine data
cleaned_engine = []
for cc in data_frame["Engine"]:
    if pandas.notna(cc):
        cc = str(cc)
        cc = cc.replace("cc", "")
        cc = cc.strip()
        cleaned_engine.append(cc)
    else:
        cleaned_engine.append(cc)
#new engine data
data_frame["Engine"] = cleaned_engine
#fill in missing values with the median
median_for_engine = data_frame["Engine"].median()
data_frame["Engine"] = data_frame["Engine"].fillna(median_for_engine)

#some values in the drivetrain are missing, this will remove them
data_frame = data_frame.dropna(subset = ["Drivetrain"])

#fill in median for missing values of seating capacity
median_for_seats = data_frame["Seating Capacity"].median()
data_frame["Seating Capacity"] = data_frame["Seating Capacity"].fillna(median_for_seats)

#fill in median for missing values of fuel tank capacity
median_for_fuel = data_frame["Fuel Tank Capacity"].median()
data_frame["Fuel Tank Capacity"] = data_frame["Fuel Tank Capacity"].fillna(median_for_fuel)

#if we want to see similar cars, keep a copy of the cleaned dataset for later
original_data = data_frame.copy()

#converts text into values
data_frame = pandas.get_dummies(data_frame, drop_first = True)



