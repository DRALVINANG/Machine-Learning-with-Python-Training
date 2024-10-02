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
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)


#------------------------------------------------------------------------------------------------
#Step 3: Build and Train the RFC
#------------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier object.
rfc = RandomForestClassifier(n_estimators=100, criterion="entropy")

# Fit the model to the training data.
rfc.fit(train_X, train_y)


#------------------------------------------------------------------------------------------------
#Step 4: Prediction
#------------------------------------------------------------------------------------------------
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    "predicted_class": rfc.predict(test_X),
    "actual_class": test_y.tolist()
})


# Print the DataFrame
display(df)


#------------------------------------------------------------------------------------------------
#Step 5: Accuracy Score
#------------------------------------------------------------------------------------------------
from sklearn.metrics import accuracy_score

accuracy_score(test_y.tolist(), rfc.predict(test_X))

#very high accuracy!




#------------------------------------------------------------------------------------------------
#Step 6: Visualizing the Tree
#------------------------------------------------------------------------------------------------
import graphviz
from sklearn.tree import export_graphviz

# Extract individual decision trees from the Random Forest
estimators = rfc.estimators_

# Visualize each tree using Graphviz
for i in range(len(estimators)):
    dot_data = export_graphviz(estimators[i], out_file=None, feature_names=list(X.columns),
                               class_names=["Setosa", "Versicolor", "Virginica"], filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    display(graph)  # This will print the trees in your console

#------------------------------------------------------------------------------------------------
#Step 7: THE END
#------------------------------------------------------------------------------------------------


