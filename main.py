import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv("Indian_housing_Mumbai_data.csv")
print(df.head())

print("providing the  summary of the dataset")

print(df.info())#in order to get the summary of the dataset


print("providing the  statistical data  of the dataset")

print(df.describe())


print("providing the  shape of the dataset (rows, columns)")
print(df.shape)

#figuring out the no of  null values in the dataset
print("providing the  no of null values in the dataset")
print(df.isnull().sum())


#Handling the null values in the dataset

# as we are planned to replace the  null values median value of numbathrooms 
df['numBathrooms'].fillna(df['numBathrooms'].median(), inplace=True)

print(df["house_size"])


#now converting the house_size column to numeric values by removing the ' sq ft' and ',' characters and converting
# the resulting string to an integer
# "1200 sq ft" -> "1200" -> 1200
df['house_size'] = df['house_size'].str.replace(' sq ft', '')
df['house_size'] = df['house_size'].str.replace(',', '')
df['house_size'] = df['house_size'].astype(int)

print(df["house_size"])
print(df.info())

'''
#Now performing the EDA operations on the dataset
sns.histplot(df["price"], kde=True)
plt.show()

sns.scatterplot(x="house_size", y="price", data=df)
plt.xscale("log")

plt.show()


# to identify the sumary of the price distribution across different cities
sns.boxplot(x="city", y="price", data=df)
plt.show()


#using heatmap we are trying to find the relationship between the
#numerical features in the dataset, such as price, house_size, numBedrooms, and numBathrooms.

plt.figure(figsize=(10,8))
#it is used to set the size of the figure to 10 inches wide and 8 inches tall. 
#This is useful for improving the readability of the plot, especially
# when there are many data points or when the default size is too small.
sns.heatmap(df.corr(numeric_only=True), annot=True)#here we are indicating that we
#need to use only numeric cloumn fo calculating the correlation matrix,
# and annot=True is used to display the correlation coefficients on the heatmap.
plt.show()

#Here we are detecting the outlieries in price column
sns.boxplot(x=df["price"])
median = df["price"].median()
plt.text(median, 0,
         f"Median: {median:.0f}",
         ha='center')
plt.show()


#as the price column produce lot of oulteries it can affect
#the  training model performance, so we are 
# applying log transformation to the price column to reduce the
# skewness and make the distribution more normal.

df["price"] = np.log(df["price"])

median = df["price"].median()
ax=sns.boxplot(x=df["price"])
plt.text(median, 0,
         f"Median: {median:.0f}",
         ha='center')
plt.show()


#the  Relationship Between Bathrooms and Price
sns.scatterplot(x="numBathrooms", y="price", data=df)
plt.show()



# it is used to create a count plot of the "house_type"
# column in the dataset, which shows the frequency of each unique value in that column.
plt.figure(figsize=(10,8))

ax=sns.countplot(x="house_type", data=df)
plt.xticks(rotation=45)#the label will be rotated by 45 degree to make it more readable

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom')
plt.show()


#pair plot is used to visualize the pairwise relationships between the numerical features in 
# the dataset, such as price, house_size, and numBathrooms.
sns.pairplot(df[['house_size', 'numBathrooms', 'price','house_type']],hue="house_type")
plt.show()
'''
#here we  checking the skewness of the price
#if the skew value is less then it will work will  with reggression model 
print(df['price'].skew())



#here data cleaning  process


#here your removing the columns that are not relevant 
# for the analysis or modeling process.
df.drop([
    'priceSqFt',
    'numBalconies',
    'isNegotiable',
    'description',
    'currency',
    'verificationDate'
], axis=1, inplace=True)

print(df.head())


print(df.isnull().sum())

df = pd.get_dummies(df, drop_first=True)
print(df.head())

pd.set_option('display.max_columns', None)

print(df)



# now building the model is started


# step-1 seprating  the important feature data target variable
X = df.drop("price", axis=1)

y = df["price"]

# step-2 splitting the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,# 20% of the data for testing
    random_state=42
)


#print("X_train shape:", X_train.sh)
print("X_train Shape:",  X_train.shape)
print("X_test Shape:", X_test.shape)
print("Y_train Shape:", y_train.shape)
print("Y_test Shape:", y_test.shape)
print("X_train:", X_train)



#it is time to build the linear regression model and train it on the training data
model = LinearRegression()



model.fit(X_train, y_train)
#here we are making model to learn from the data and draw the best fit line by calculating 
#the  m and b using least square method


print("The model's m value", model.coef_)
print("The model's b value", model.intercept_)



predictions = model.predict(X_test)
#print("Predictions:", predictions)


# lets evaluate the model performance by comparing the predicted values with the actual values from the test set.
# MAE (MEAN ABSOLUTE ERROR)

mae = mean_absolute_error(y_test, predictions)

print("MEAN ABSOLUTE ERROR:",mae)

'''for actual, predicted in zip(y_test, predictions):
    print("Actual:", actual, "Predicted:", predicted)
'''

#predictions is predicted values
r2 = r2_score(y_test, predictions)
print("R2 Score:", r2)


#RMSE method is used to evalaute the model performance 
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("Root Mean Squared Error:", rmse)


print("Mean of y_test:", y_test.mean())



errors = abs(y_test - predictions)
print(errors)
largest_errors = errors.sort_values(ascending=False)

print(largest_errors.head(10))  # Display the top 10 largest errors
largest_idx = largest_errors.head(10).index
print(largest_idx)
#below i am  giving one dimensional array
#with labeled indexe
# why we are converting the predictions to a pandas Series 
# with the same index as y_test ?
# because we know that prediction=array([11800, 14800, 9800])
# now this array need to have the index  same as ytest 
#then only while fetching the element from predictions_series with ytest
#index it will give the correct predicted value
predictions_series = pd.Series(predictions, index=y_test.index)

a=pd.DataFrame({
    "Actual": y_test.loc[largest_idx],#label based indexing and fetching
    "Predicted": predictions_series.loc[largest_idx],
    "Error": abs(y_test.loc[largest_idx] - predictions_series.loc[largest_idx])
})
#here with labels of the array elements will be algined as df index


print(a) 


plt.scatter(y_test, predictions)
plt.plot(
    [y_test.min(), y_test.max()],#x coorditnates [0,1000, 20000]
    [y_test.min(), y_test.max()],#y coorditnates  [0,1000, 20000]
    color='red'
)
#(0,0) to (20000,20000) red line will be drawn
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()


for feature, coef in zip(X.columns, model.coef_):
    print(feature, coef)