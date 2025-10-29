![alt text](image.png)

# Cash Flow Statement
It's the hand-made cash flow statement we'll present in the (if we trust it enough) demo the product. It follows the example dataset that's on Teams.

*PROBLEM*: Not all descriptions contain the amount (kg,l,etc.) purchased. Therefore we need a way to calculate it.

# Amount per euro dataset
This is a dependency for the calculation of emissions. Based on country(?)
    - https://teseo.clal.it/en/?section=gasolio_agricolo
    - https://www.cargopedia.net/tools

# Parser
A python script that reads the cash flow statement and:
    - accepts different formats of the same unit of measurement (e.g. kg/kilogram/kilograms/ etc.)
    - reads amount from description if available.
    - standardises it to kg for weight, liters for volumes and kwh for energy.

# XGBoost (✨AI✨)
(https://www.geeksforgeeks.org/machine-learning/xgboost/)
Ensemble learning model that uses multiple decision trees (classificators in our case) to classify data:
- Uses TF-IDF (https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/) for string classification. 
- Results are in the notebooks presented. 
- Extremely high accuracy if data input makes sense. 
- Can easily use Synthetic data for input. 
- It's agnostic to languages. 
- Some hyperparameter tuning could be done, but not strictly necessary. 
- Can be tested on hand-made. 
 
# Complete data with all amounts and classes
This dataset is the same as the previous one, with the predicted classes. We only classify useful ones, the rest is classified as 'other' and ignored.

# Emission per class dataset
This dataset contains the amount of co2eq or whatever per class.

# Emission dataset
This final result calculates all emissions needed for the ESG indicators (water, electricity, co2eq, etc.).

# Report generator
Final component that uses the emission dataset and all indicators calculated to produce the final output.