#------------------------------------------------------------------------------------------------
#Step 1: Import Dataset
#------------------------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the iris dataset
iris = pd.read_csv("https://www.alvinang.sg/s/iris_dataset.csv")

# Create a label encoder
le = LabelEncoder()

# Fit the label encoder to the species column
le.fit(iris["species"])

# Transform the species column
iris["species"] = le.transform(iris["species"])

iris

#In the iris dataset,
#Rows 0 to 49 = Setosa = Class 0
#Rows 50 to 99 = Versicolor = Class 1
#Rows 100 to 149 = Vriginica = Class 2

#------------------------------------------------------------------------------------------------
#Step 2: Train Test Split
#------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

# Target
y = iris["species"]

# Features
X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

#Split the data into a training set and a testing set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#------------------------------------------------------------------------------------------------
#Step 3: Train the Decision Tree
#------------------------------------------------------------------------------------------------
from sklearn import tree

#Build the classifier
dtc = tree.DecisionTreeClassifier(criterion='entropy')

#Train the classifier
dtc.fit(X_train, y_train)

#------------------------------------------------------------------------------------------------
#Step 4: Test the Tree
#------------------------------------------------------------------------------------------------
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    "predicted_class": dtc.predict(X_test),
    "actual_class": y_test.tolist()
})


# Print the DataFrame
display(df)

#------------------------------------------------------------------------------------------------
#Step 5: Accuracy Score
#------------------------------------------------------------------------------------------------
from sklearn.metrics import accuracy_score

accuracy_score(y_test.tolist(), dtc.predict(X_test))

#very high accuracy!

#------------------------------------------------------------------------------------------------
#Step 6: Visualize the Tree
#------------------------------------------------------------------------------------------------
from graphviz import Source

class_names = [str(class_name) for class_name in pd.unique(iris['species'])]

graph = Source(tree.export_graphviz(dtc, out_file=None,
                      feature_names=X.columns,
                      class_names=class_names,
                       filled=True,
                      #rounded=True,
                      #node_ids= False,
                      #special_characters=False
               ))

display(graph)
#note that display only works for Google Colab
#u must change to print and install graphviz on your computer if you are using thonny


#------------------------------------------------------------------------------------------------
#Step 7: Prediction
#------------------------------------------------------------------------------------------------
import pandas as pd

# Create a DataFrame with a simulated row of data
# Replace the values below with whatever you'd like to test
simulated_data = pd.DataFrame({
    "sepal_length": [1.1],  # Example sepal length value
    "sepal_width": [1.5],   # Example sepal width value
    "petal_length": [6.4],  # Example petal length value
    "petal_width": [1.5]    # Example petal width value
})

# Use the trained decision tree classifier to predict the class of this simulated data
predicted_class = dtc.predict(simulated_data)

# Display the predicted class
print(f"Predicted class for the simulated data: {predicted_class[0]}")
