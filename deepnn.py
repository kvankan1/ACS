import torch
import numpy as np
from input_data import Data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# Set the PyTorch and numpy random seeds for reproducibility:
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

def train_regressor_nn(n_features, n_outputs, n_hidden_neurons, learning_rate, n_epochs, X, Y):

    

    model = torch.nn.Sequential(
    torch.nn.Linear(n_features, n_hidden_neurons),
    torch.nn.ReLU(),
    torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
    torch.nn.ReLU(),
    torch.nn.Linear(n_hidden_neurons, n_outputs)  
)


    # MSE loss function:
    loss_fn = torch.nn.MSELoss()

    # optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the network:
    for t in range(n_epochs):
        # Forward pass
        y_pred = model(X)
    # Compute and print loss. We pass Tensors containing the predicted and
    # true values of y, and the loss function returns a Tensor containing
    # the loss.
        loss = loss_fn(y_pred, Y)
        # if t % 100 == 0:
        # print(t, loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # return the trained model
    return model


def Optimize(model, X_optimize, bmep_tolerance = 5):

    # Set the model to evaluation mode
    model.eval()

    # Forward pass to get predictions
    with torch.no_grad():
        y_pred_candidates = np.array(model(X_optimize))

    # Extract the predicted values for the constraints
    temp_in_turbo = y_pred_candidates[:, 1]  # Assuming temp_in_turbo is the second column
    in_cylinder_max_pressure = y_pred_candidates[:, 2]  # Assuming in_cylinder_max_pressure is the third column
    max_compressor_pressure = y_pred_candidates[:, 5]  # Assuming max_compressor_pressure is the sixth column
    knock_limited_mass = y_pred_candidates[:, 4]  # Assuming knock_limited_mass is the fifth column
    torque_limit = y_pred_candidates[:,0 ]  #Torque limit is first column
    bmep = y_pred_candidates[:, -1] #Bmep is the last column

    optimal_bmep = 28.0

    # Check if constraints are met for each candidate


    constraints_met = (
        (temp_in_turbo <= 1223.15) &
        (in_cylinder_max_pressure <= 140.0) &
        (max_compressor_pressure <= 3.0) &
        (knock_limited_mass <= 233.3) &
        (torque_limit >= 100.0) &
        (abs(bmep - optimal_bmep) < bmep_tolerance) 
    )

    # Filter candidates that meet all constraints
    valid_candidates = y_pred_candidates[constraints_met]

    # If there are valid candidates, find the one with the lowest BSFC
    if len(valid_candidates) > 0:
        # Get indices of valid candidates
        optimal_index = np.argmin(valid_candidates[:,3])

        # Get the optimal solution
        optimal_solution = valid_candidates[optimal_index]

    else:
        # If there are no valid candidates, set optimal_solution to None
        optimal_solution = None
    

    return optimal_solution




y_keys_all = ["Torque [Nm]", "Temp in Turbo [K]", "In cylinder max Pressure [bar]", 'BSFC [g/kwH]',"Knock mass [mg]", "Max compressor pressure [bar]", "BMEP [bar]"]  #All outputs
y_keys_all = ['BSFC [g/kwH]', "Temp in Turbo [K]", "In cylinder max Pressure [bar]"]  #Choose y_keys according to which outputs you want to model


X_keys = ['Stroke/Bore', 'Volumetric coefficient', 'Compression ratio', 'norm. TKE', 'SA', 'Water inj.', 'EIVC']    #Choose x_keys according to which inputs you want to include
y_keys = ['BSFC [g/kwH]', "Temp in Turbo [K]", "In cylinder max Pressure [bar]"]

keys_all = ["Torque [Nm]", "Temp in Turbo [K]", "In cylinder max Pressure [bar]", 'BSFC [g/kwH]',"Knock mass [mg]", "Max compressor pressure [bar]", "BMEP [bar]", 'Stroke/Bore', 'Volumetric coefficient', 'Compression ratio', 'norm. TKE', 'SA', 'Water inj.', 'EIVC']





#========================================================NOW THE OPTIMIZER=============================================================#
##Now the Optimizer, we first define the subspace of input values to be studied, this is done below, where minimum and maximum values
#of each input parameter are given, along with the amount of data points. This should be the same for all data. Then reshaped

n_data = 10

# Stroke to Bore Ratio
sbr = np.linspace(0.8, 1.3, n_data).reshape(-1, 1)

# Volumetric Coefficient
volumetric_coefficient = np.linspace(0.5, 2.0, n_data).reshape(-1, 1)

# Compression Ratio
compression_ratio = np.linspace(12, 14, n_data).reshape(-1, 1)

# Norm. TKE
norm_tke = np.linspace(0.8, 1.5, n_data).reshape(-1, 1)

# Spark Advance
spark_advance = np.linspace(0, 40, n_data).reshape(-1, 1)

# Water Injection
water_injection = np.linspace(0, 40, n_data).reshape(-1, 1)

# EIVC
eivc = np.linspace(0, 230, n_data).reshape(-1, 1)

# Stack all arrays vertically
#X_optimize = np.hstack((sbr, volumetric_coefficient, compression_ratio, norm_tke, spark_advance, water_injection, eivc))

# Create a grid of all possible combinations
X_optimize = X_optimize = torch.tensor(scaler.fit_transform(np.array(np.meshgrid(sbr, volumetric_coefficient, compression_ratio, norm_tke, spark_advance, water_injection, eivc)).T.reshape(-1, 7)), dtype=torch.float32)


#print(X_optimize)

#Finally call the optimizer function and print any results
if __name__ == "__main__":
     
    input_data = torch.tensor(scaler.fit_transform(np.array([Data[key] for key in X_keys])), dtype=torch.float32).T
    output_data = torch.tensor(np.array([Data[key] for key in keys_all]), dtype=torch.float32).T


    print(input_data.shape)
    print(output_data.shape)

    model = train_regressor_nn(len(X_keys), len(keys_all), 200, 0.25, 25, input_data, output_data)

    opt_sol = Optimize(model, X_optimize, bmep_tolerance=5)

    for i, key in enumerate(keys_all):
        print(f"{key} = {opt_sol[i]}")
