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


correlation_matrix, remaining_keys = GetCorrelationMatrix(Data)
PrintCorrelationMatrix(remaining_keys, correlation_matrix)

# Assuming Data is a dictionary
#print(Data.keys())

# Extracting data from the dictionary using keys
X_keys = ['Stroke/Bore', 'Volumetric coefficient', 'Compression ratio', 'norm. TKE', 'SA', 'Water inj.', 'EIVC']
y_keys = ['Torque [Nm]', 'Temp in Turbo [K]', 'In cylinder max Pressure [bar]', 'BSFC [g/kwH]', 'Knock mass [mg]', 'Max compressor pressure [bar]', 'BMEP [bar]']

# Extracting data using keys
X = np.array([Data[key] for key in X_keys])
y = np.array([Data[key] for key in y_keys])

# Transposing the arrays for the correct shape (samples x features)
X = X.T
y = y.T

# Initialize a CCA object
#cca = CCA(n_components=6)

# Fit the CCA model
#cca.fit(X, y)

# Get the loadings of the canonical variables
loadings = cca.x_weights_

#print("weights =", loadings)



###Or use PLS Regression?

#https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html

