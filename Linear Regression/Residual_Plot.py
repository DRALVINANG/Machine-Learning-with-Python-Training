#----------------------------------------------------------
# Create a Linear Regression model
#----------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

#----------------------------------------------------------
# Predict the values for the test set
#----------------------------------------------------------
y_pred = model.predict(X_test)

#----------------------------------------------------------
# Plot the residuals
#----------------------------------------------------------
plt.figure(figsize=(12, 6))
sns.residplot(x=y_test, y=y_pred, lowess=True, color='g')

#----------------------------------------------------------
# Add labels and title
#----------------------------------------------------------
plt.xlabel('Actual MPG')
plt.ylabel('Residuals')
plt.title('Residual Plot: Actual MPG vs Predicted MPG')

#----------------------------------------------------------
# Display the plot
#----------------------------------------------------------
plt.show()
