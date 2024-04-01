#-------------------------------------------------------
#Using .Reshape
#-------------------------------------------------------
#used when
#'valueerror: expected 2D array, got 1D array instead

X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)


#-------------------------------------------------------
#Using pd.DataFrame
#-------------------------------------------------------
#used when
#'valueerror: expected 2D array, got 1D array instead

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

#-------------------------------------------------------
#Using .flatten()
#-------------------------------------------------------
#used when
#valueerror: x and y must have same first dimension, but
#have shapes (100,) and (1,100)

y_new = y_new.flatten()

#-------------------------------------------------------
# THE END
#-------------------------------------------------------


