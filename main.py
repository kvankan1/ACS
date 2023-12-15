import numpy as np
from sklearn.cross_decomposition import CCA
from input_data import Data
import numpy as np

##=======================================================First We Define The Functions=======================================================##

def GetCorrelationMatrix(Data : dict):
    #Get the correlation matrix of a data dictionary

    # Assuming Data is a dictionary
    keys = list(Data.keys())

    # Extracting data from the dictionary using keys
    data = np.array([Data[key] for key in keys])

    # Check for zero variance in columns
    zero_variance_columns = np.where(np.std(data, axis=0) == 0)[0]

    # Remove columns with zero variance
    data = np.delete(data, zero_variance_columns, axis=1)

    # Get the remaining keys after removing columns with zero variance
    remaining_keys = [keys[i] for i in range(len(keys)) if i not in zero_variance_columns]

    # Transposing the array for the correct shape (samples x features)
    data = data.T

    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(data, rowvar=False)

    return correlation_matrix, remaining_keys


def PrintCorrelationMatrix(keys, correlation_matrix):
    #Print the Correlation Matrix

    # Set a fixed width for each column
    column_width = 50

    # Create a text file with dictionary keys above and on the side of the matrix
    with open("correlation_matrix.txt", 'w') as file:
        # Write the keys above the matrix
        file.write("\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t".join(f"{key:<{column_width}}" for key in remaining_keys) + "\n")

        # Write the matrix with keys on the side
        for i in range(len(remaining_keys)):
            file.write(f"{remaining_keys[i]:<{column_width}}\t" + "\t".join(f"{value:.3f}".ljust(column_width) for value in correlation_matrix[i]) + "\n")

def PrintEquation(cca, X_keys, y_keys):
    #Print the equation mapping the canonical coefficients to each output y

    # Get the canonical coefficients
    canonical_coefs = cca.coef_ # Access the first component

    # Get the intercept term from the CCA object
    intercept = cca.intercept_

    # Create the equation string

    for i, y in enumerate(y_keys):
        equation = f"{y} = "
        for j in range(n_components):
            equation += f"{canonical_coefs[i][j]:.3f} * C{j} + "
        equation += f"{intercept[i]:.3f}"
        print(equation)

def PrintXWeights(cca, X_keys, y_keys):
    #Print the weights of X that makes up each canonical variable

    # Set a fixed width for each column
    column_width = 50

    weights = cca.x_weights_

    # Create a text file with dictionary keys above and on the side of the matrix
    with open("Xweights.txt", 'w') as file:
        # Write the keys above the matrix
        file.write("\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t".join(f"C{i:<{column_width}}" for i in range(n_components)) + "\n")

        # Write the matrix with keys on the side
        for i in range(len(X_keys)):
            file.write(f"{X_keys[i]:<{column_width}}\t" + "\t".join(f"{value:.3f}".ljust(column_width) for value in weights[i]) + "\n")

def PrintYWeights(cca, X_keys, y_keys):
    #Print the weights of y that correspond to each canonical variable

    # Set a fixed width for each column
    column_width = 50

    weights = cca.y_weights_

    # Create a text file with dictionary keys above and on the side of the matrix
    with open("Yweights.txt", 'w') as file:
        # Write the keys above the matrix
        file.write("\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t".join(f"C{i:<{column_width}}" for i in range(n_components)) + "\n")

        # Write the matrix with keys on the side
        for i in range(len(y_keys)):
            file.write(f"{y_keys[i]:<{column_width}}\t" + "\t".join(f"{value:.3f}".ljust(column_width) for value in weights[i]) + "\n")
    
def PrintXLoadings(cca, X_keys, y_keys):
    #Print the loadings for each X variable and the canonical variable, this basically shows the correlation of each canonical variable to the X variable. Higher means more strongly correlated, negative means negatively correlated

    # Set a fixed width for each column
    column_width = 50

    x_loadings = cca.x_loadings_

    # Create a text file with dictionary keys above and on the side of the matrix
    with open("Xloadings.txt", 'w') as file:
        # Write the keys above the matrix
        file.write("\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t".join(f"C{i:<{column_width}}" for i in range(n_components)) + "\n")

        # Write the matrix with keys on the side
        for i in range(len(X_keys)):
            file.write(f"{X_keys[i]:<{column_width}}\t" + "\t".join(f"{value:.3f}".ljust(column_width) for value in x_loadings[i]) + "\n")


def PrintYLoadings(cca, X_keys, y_keys):
    #Print the loadings for each Y variable and the canonical variable, showing the correlation

    # Set a fixed width for each column
    column_width = 50

    y_loadings = cca.y_loadings_

    # Create a text file with dictionary keys above and on the side of the matrix
    with open("Yloadings.txt", 'w') as file:
        # Write the keys above the matrix
        file.write("\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t".join(f"C{i:<{column_width}}" for i in range(n_components)) + "\n")

        # Write the matrix with keys on the side
        for i in range(len(y_keys)):
            file.write(f"{y_keys[i]:<{column_width}}\t" + "\t".join(f"{value:.3f}".ljust(column_width) for value in y_loadings[i]) + "\n")


# Define the Optimize function
def Optimize(X_train, Y_train, X_predict, y_keys):
    n_components = len(y_keys)

    # Fit the CCA model
    cca = CCA(n_components = n_components)
    cca.fit(X_train, Y_train)

    # Predict using the CCA model for all candidates
    y_pred_candidates = cca.predict(X_predict)

    # Extract the predicted values for the constraints
    temp_in_turbo = y_pred_candidates[:, 1]  # Assuming temp_in_turbo is the second column
    in_cylinder_max_pressure = y_pred_candidates[:, 2]  # Assuming in_cylinder_max_pressure is the third column
    max_compressor_pressure = y_pred_candidates[:, 5]  # Assuming max_compressor_pressure is the sixth column
    knock_limited_mass = y_pred_candidates[:, 4]  # Assuming knock_limited_mass is the fifth column
    torque_limit = y_pred_candidates[:,0 ]

    # Check if constraints are met for each candidate
    constraints_met = (
        (temp_in_turbo <= 1223.15) &
        (in_cylinder_max_pressure <= 140.0) &
        (max_compressor_pressure <= 2.5) &
        (knock_limited_mass <= 164.0) &
        (torque_limit >= 100.0)
    )

    # Filter candidates that meet all constraints
    valid_candidates = y_pred_candidates[constraints_met]

    # If there are valid candidates, find the one with the lowest BSFC
    if len(valid_candidates) > 0:
        # Get indices of valid candidates
        optimal_index = np.argmin(valid_candidates[:,3])

        # Get the optimal solution
        optimal_solution = valid_candidates[optimal_index]
        optimal_parameters = X_optimize[optimal_index]
    else:
        # If there are no valid candidates, set optimal_solution to None
        optimal_solution = None
        optimal_parameters = None

    return optimal_solution, optimal_parameters


def PrintSol(opt_sol, opt_param):
    for i in range(len(y_keys_all)):
        print(f"Optimal {y_keys_all[i]} = {opt_sol[i]} \n")

    for j in range(len(X_keys)):
        print(f"Optimal {X_keys[j]} = {opt_param[j]} \n")



##=========================================================Now We Use the Functions with the Input Data============================================================#
    
#Get and print the correlation matrix
correlation_matrix, remaining_keys = GetCorrelationMatrix(Data)
PrintCorrelationMatrix(remaining_keys, correlation_matrix)


##===========================================================BASIC SET UP OF THE DATA===========================================================================#

#We define, y_keys, used to create a CCA object to study only some keys, and y_keys_all, to keep track of all the outputs, and to use later in the optimize function

# Extracting data from the dictionary using keys
X_keys = ['Stroke/Bore', 'Volumetric coefficient', 'Compression ratio', 'norm. TKE', 'SA', 'Water inj.', 'EIVC']    #Choose x_keys according to which inputs you want to include
y_keys_all = ["Torque [Nm]", "Temp in Turbo [K]", "In cylinder max Pressure [bar]", 'BSFC [g/kwH]',"Knock mass [mg]", "Max compressor pressure [bar]", "BMEP [bar]"]  #All outputs
y_keys = ['BSFC [g/kwH]', "Temp in Turbo [K]", "In cylinder max Pressure [bar]"]  #Choose y_keys according to which outputs you want to model


# Extracting data using keys, Data is taken from input_data.py
#In this case, y is created from all the y_keys, such that all output data are placed in y

X = np.array([Data[key] for key in X_keys])
y = np.array([Data[key] for key in y_keys_all])

# Transposing the arrays for the correct shape (samples x features)
X = X.T
y = y.T

# Initialize a CCA object
#n_components should match number of keys, right now all outputs are studied
n_components = 7
cca = CCA(n_components)

# Fit the CCA model
cca.fit(X, y)


#Print all equations, weights and loadings, functions are called from above
PrintEquation(cca, X_keys, y_keys_all)
PrintXWeights(cca, X_keys, y_keys_all)
PrintYWeights(cca, X_keys, y_keys_all)
PrintXLoadings(cca, X_keys, y_keys_all)
PrintYLoadings(cca, X_keys, y_keys_all)



#========================================================NOW THE OPTIMIZER=============================================================#
##Now the Optimizer, we first define the subspace of input values to be studied, this is done below, where minimum and maximum values
#of each input parameter are given, along with the amount of data points. This should be the same for all data. Then reshaped

n_data = 10

# Stroke to Bore Ratio
sbr = np.linspace(0.8, 1.3, n_data).reshape(-1, 1)

# Volumetric Coefficient
volumetric_coefficient = np.linspace(0.7, 2.2, n_data).reshape(-1, 1)

# Compression Ratio
compression_ratio = np.linspace(6, 14, n_data).reshape(-1, 1)

# Norm. TKE
norm_tke = np.linspace(0.8, 1.5, n_data).reshape(-1, 1)

# Spark Advance
spark_advance = np.linspace(0, 60, n_data).reshape(-1, 1)

# Water Injection
water_injection = np.linspace(0, 50, n_data).reshape(-1, 1)

# EIVC
eivc = np.linspace(165, 230, n_data).reshape(-1, 1)

# Stack all arrays vertically
#X_optimize = np.hstack((sbr, volumetric_coefficient, compression_ratio, norm_tke, spark_advance, water_injection, eivc))

# Create a grid of all possible combinations
X_optimize = np.array(np.meshgrid(sbr, volumetric_coefficient, compression_ratio, norm_tke, spark_advance, water_injection, eivc)).T.reshape(-1, 7)
print(X_optimize)

#Finally call the optimizer function and print any results

opt_sol, opt_param = Optimize(X_train = X, Y_train = y, X_predict = X_optimize, y_keys = y_keys_all)

PrintSol(opt_sol, opt_param)











