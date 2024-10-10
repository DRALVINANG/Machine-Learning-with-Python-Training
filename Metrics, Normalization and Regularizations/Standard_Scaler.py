from sklearn.preprocessing import StandardScaler

#-------------------------------------------------------------
# Standard Scaling
#-------------------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#-------------------------------------------------------------
# Inverse Transform (to get back original)
#---------------------------------------------------
X_train_original = scaler.inverse_transform(X_train_scaled)

X_test_original = scaler.inverse_transform(X_test_scaled)

#---------------------------------------------------
# THE END
#---------------------------------------------------
