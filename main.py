import numpy as np
from sklearn.cross_decomposition import CCA
from input_data import Data
import numpy as np

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
    

correlation_matrix, remaining_keys = GetCorrelationMatrix(Data)
PrintCorrelationMatrix(remaining_keys, correlation_matrix)

# Assuming Data is a dictionary
#print(Data.keys())

# Extracting data from the dictionary using keys
X_keys = ['Stroke/Bore', 'Volumetric coefficient', 'Compression ratio', 'norm. TKE', 'SA', 'Water inj.', 'EIVC']
y_keys = ['BSFC [g/kwH]']

# Extracting data using keys
X = np.array([Data[key] for key in X_keys])
y = np.array([Data[key] for key in y_keys])

# Transposing the arrays for the correct shape (samples x features)
X = X.T
y = y.T

# Initialize a CCA object
n_components = 1
cca = CCA(n_components)

# Fit the CCA model
cca.fit(X, y)

# Get the loadings of the canonical variables
loadings = cca.x_weights_

PrintEquation(cca, X_keys, y_keys)
PrintXWeights(cca, X_keys, y_keys)
PrintYWeights(cca, X_keys, y_keys)
PrintXLoadings(cca, X_keys, y_keys)
PrintYLoadings(cca, X_keys, y_keys)


print("loadings =", loadings)


##Need to print the equation
##Then print what makes up the canonical variable
##Then print the loadings 
##Interpret everything then try to optimize with 2/3 most important variables



###Or use PLS Regression?

#https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html

