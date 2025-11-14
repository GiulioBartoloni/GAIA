import pandas as pd

# Read the Excel file
file_path = 'Syngenta_2023_Cash_Flow_Statement_Electric_Adjusted.xlsx'  # Replace with your actual filename
df = pd.read_excel(file_path)

# Find rows where GL_account is "Utility Expense - Energy" and divide by 3.5
# You need to specify which column to divide - I'll assume it's a column with numeric values
# Replace 'Amount' with your actual column name
mask = df['GL_Account'] == 'Utility Expense - Energy'
df.loc[mask, 'Amount_EUR'] = df.loc[mask, 'Amount_EUR'] / 3.5

# Save back to the same file
df.to_excel(file_path, index=False)

print(f"File '{file_path}' has been updated successfully!")