#-----------------------------------------------------------
# Pip install Scikit Plot
#-----------------------------------------------------------
!pip install scikit-plot

#-----------------------------------------------------------
# Import Libraries
#-----------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
import matplotlib.pyplot as plt  # Import for plotting

#-----------------------------------------------------------
# Let X be features and y be target
#-----------------------------------------------------------
url = "https://www.alvinang.sg/s/iris_dataset.csv"
df = pd.read_csv(url)

X = df.iloc[:, :4]
y = df.iloc[:, -1]

#-----------------------------------------------------------
# Train Test Split
#-----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#-----------------------------------------------------------
# Train using Logistic Regression
#-----------------------------------------------------------
model = LogisticRegression(max_iter=200)  # Set max_iter to avoid convergence issues
model.fit(X_train, y_train)

#-----------------------------------------------------------
# Display the Probabilities of each class
#-----------------------------------------------------------
y_probs = model.predict_proba(X_test)

#-----------------------------------------------------------
# Plotting the ROC curve
#-----------------------------------------------------------
# Plot ROC curve for multi-class classification
skplt.metrics.plot_roc(y_test, y_probs)
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

#-----------------------------------------------------------
# THE END
#-----------------------------------------------------------
