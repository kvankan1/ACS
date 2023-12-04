import pandas as pd

# Replace 'your_file.xlsx' with the actual path to your Excel file
file_path = 'For_Kasper__From_France.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path)

# Drop rows containing NaN values in any column
df = df.dropna()

# Create an empty dictionary to store the arrays
data_dict = {}

# Iterate through each column and store the data in a dictionary
for column_name in df.columns:
    data_dict[column_name] = df[column_name].tolist()

# Print the dictionary for verification
#print(data_dict)

Data = data_dict

print("Data Imported")