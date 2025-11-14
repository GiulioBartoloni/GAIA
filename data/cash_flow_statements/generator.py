import pandas as pd
import numpy as np

# --- Configuration ---
# The constant number provided by you
DIVISOR_NUM = 1461336289
# The GL Account to filter on
GL_ACCOUNT_NAME = "Sales of Products"
# Replace 'your_file.xlsx' with the actual path to your Excel document
INPUT_FILE_PATH = 'Syngenta_2023_Simulated_Cash_Flow_Statement.xlsx' 
# Name for the output file
OUTPUT_FILE_PATH = 'Syngenta_2023_Simulated_Cash_Flow_Statement.xlsx'

# --- Processing ---
try:
    # 1. Read the Excel file
    df = pd.read_excel(INPUT_FILE_PATH)
    print(f"Successfully loaded file: {INPUT_FILE_PATH}")

    # 2. Create the boolean mask to filter the rows
    filter_mask = df['GL_Account'] == GL_ACCOUNT_NAME
    
    # 3. Count the number of rows that match the filter (X)
    row_count = filter_mask.sum()
    
    if row_count == 0:
        print(f"⚠️ Warning: Found no rows for GL_Account='{GL_ACCOUNT_NAME}'. No changes made.")
    else:
        # 4. Calculate the value to be added (V = 1461336289 / X)
        value_to_add = DIVISOR_NUM / row_count
        print(f"Found {row_count} rows for '{GL_ACCOUNT_NAME}'.")
        print(f"Calculated value to add (V): {value_to_add:,.2f}")

        # 5. Identify all numerical columns in the DataFrame
        # This ensures we only add the value to columns that are numbers (int or float)
        numeric_cols = df.select_dtypes(include=np.number).columns
        
        # 6. Add V to all numerical entries in the filtered rows
        # .loc is used to safely update specific rows and columns in the original DataFrame
        df.loc[filter_mask, numeric_cols] = df.loc[filter_mask, numeric_cols] + value_to_add
        
        print(f"✅ Successfully added V to all numeric entries in the '{GL_ACCOUNT_NAME}' rows.")

    # 7. Save the updated DataFrame to a new Excel file
    df.to_excel(OUTPUT_FILE_PATH, index=False)
    print(f"✅ Updated data saved to: {OUTPUT_FILE_PATH}")

except FileNotFoundError:
    print(f"Error: File not found at {INPUT_FILE_PATH}. Please check the file path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")