
#----------------------------------------------------------------
# Import Libraries & Dataset
#----------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('https://www.alvinang.sg/s/results.csv')

#----------------------------------------------------------------
# Wrangle the Dataset
#----------------------------------------------------------------
df['Result'] = df.Result.map({'Fail': 0,
                              'Pass': 1})

df = df.drop('StudentId',
             axis = 1)

X = df.Hours
y = df.Result

#----------------------------------------------------------------
# METHOD 1: Using Seaborn (but usually fails...)
#----------------------------------------------------------------
sns.regplot(x = X,
            y = y,
            y_jitter = 0.03,
            data = df,
            logistic = True,
            ci = None)

plt.show()

#----------------------------------------------------------------
# METHOD 2: Self Created DEF
#troublesome but will surely work...
#----------------------------------------------------------------
# Train Test Split
#----------------------------------------------------------------
split = int(0.8*len(X))

X_train, X_test, y_train, y_test = X[:split],\
                                   X[split:],\
                                   y[:split],\
                                   y[split:]

X_train = X_train.values.reshape(-1,1)

#----------------------------------------------------------------
# Fit Logistic Regresion Curve & Get Coeeficient and Intercept
#----------------------------------------------------------------
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

a = model.coef_
b = model.intercept_

print(f'Model Coefficient is {a}')
print (f'Model Intercept is {b}')

#----------------------------------------------------------------
# Create DEF
#----------------------------------------------------------------
def logistic_func(X, a, b):
  return 1 / (1 + np.exp(-(a * X + b)))

#----------------------------------------------------------------
# Generate X_new and y_new
#----------------------------------------------------------------
# Generate x-axis values for a smooth curve
X_new = np.linspace(min(X), max(X), 100)

# Calculate y-values for the curve using the logistic function
y_new = logistic_func(X_new, a, b)
y_new = y_new.flatten()

#----------------------------------------------------------------
# Plotting & Cosmetics
#----------------------------------------------------------------
# Plot the data points and the logistic curve
plt.scatter(X, y)
plt.plot(X_new, y_new, color='red', label='Logistic Curve')

# Customize the plot for better visualization
plt.xlabel('X')
plt.ylabel('Probability')
plt.title('Logistic Regression Curve')
plt.legend()
plt.grid(True)  # Add gridlines for better readability (optional)
plt.show()

#----------------------------------------------------------------
# THE END
#----------------------------------------------------------------


view rawWays to Plot Logistic Regression Curve in Python.py hosted with ❤ by GitHub
