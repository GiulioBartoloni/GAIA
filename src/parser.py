from pathlib import Path
import pickle
import pandas as pd
from pandas.api.types import is_numeric_dtype
import re
import json
from enum import Enum, auto
from typing import List


PROJECT_DIR = Path(__file__).parent.parent
INPUT_DATASET_PATH = PROJECT_DIR / 'data' / 'cash_flow_statements' / 'Syngenta_2023_Simulated_Cash_Flow_Statement.xlsx'
UNITS_OF_MEASUREMENT_DATASET_PATH = PROJECT_DIR / 'data' / 'parser' / 'units_of_measurement.csv'
CONVERSION_RATES_PATH = PROJECT_DIR / 'data'/ 'parser' / 'conversion_rates.csv'
UNITS_OF_MEASUREMENT_VARIATIONS_PATH = PROJECT_DIR / 'data'/ 'parser' / 'units_of_measurement_variations.json'
MODEL_PATH = PROJECT_DIR / 'models' / 'cash_flow_classifier.pkl'
ESG_CONVERTION_RATES_PATH = PROJECT_DIR / 'data' / 'parser' / 'ESG_indicators_conversion_rates.csv'


class NumberNotations(Enum):
    """
    This class is an enum for supported number format notations.
    """
    
    EU = auto()
    US = auto()


def predict_classes(texts:List)->pd.DataFrame:
    # Load the model
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)

    model = pipeline['model']
    vectorizer = pipeline['vectorizer']
    label_encoder = pipeline['label_encoder']
    confidence_threshold = pipeline['confidence_threshold']
    fallback_label = pipeline['fallback_label']
    
    # Transform text to features
    X = vectorizer.transform(texts)

    # Get predictions and probabilities
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Get confidence scores
    confidence_scores = probabilities.max(axis=1)

    # Decode labels
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Apply confidence threshold and fallback
    final_predictions = []
    for label, confidence in zip(predicted_labels, confidence_scores):
        if confidence < confidence_threshold:
            final_predictions.append(fallback_label)
        else:
            final_predictions.append(label)
    
    return final_predictions


def standardize_units(text:str, unit_variations:dict)->str:
    """Replace units of measurement variations in given text,
    
    Args: 
        text (str): input text where variations will be replaced.
        unit_variations (dict): dictionary with (variation:standard_unit).

    Returns: 
        str: string with unit of measurement variations replaced with its standard chosen unit.

    """
    for variation, standard_unit in unit_variations.items():
        if variation in text:
            text = text.replace(variation,standard_unit)

    return text


def parse_amount_field(dataset:pd.DataFrame, field:str, number_format:NumberNotations=NumberNotations.US)->pd.DataFrame:
    """Convert string amount field into numeric.
    
    Args:
        dataset(DataFrame): Dataset with the column to be corrected.
        field(str): column to be corrected.
        number_format(NumberNotation): US|EU notation.
    
        Returns:
            DataFrame: dataframe with chosen column converted to numeric.
    """
    if is_numeric_dtype(dataset[field]):
        return dataset

    if number_format == NumberNotations.US:
        dataset[field] = dataset[field].str.replace(',', '', regex=False)
    else:
        dataset[field] = dataset[field].str.replace('.', '', regex=False)
        dataset[field] = dataset[field].str.replace(',', '.', regex=False)
    
    dataset[field] = pd.to_numeric(dataset[field], errors='coerce')
    return dataset


def standardize_units_of_measurement(dataset:pd.DataFrame)->pd.DataFrame:
    """Convert all amounts to standard unit of measurment for volumes, energies and weights.
    
    Args:
        dataset(DataFrame): Dataset to standardize.
    
    Returns:
        Dataframe: dataframe with standardized amounts.
    """
    df = pd.read_csv(CONVERSION_RATES_PATH)
    df.columns = df.columns.str.lower()

    result = pd.merge(dataset,df, how='left')

    result['unit'] = result['target_unit']
    result['amount'] = result['amount'] * result['conversion_rate']

    result = result.drop(['target_unit', 'conversion_rate'], axis=1)

    return result


def main():
    # Read cash flow dataset
    cash_flow_dataset = pd.read_excel(INPUT_DATASET_PATH)
    cash_flow_dataset.columns = cash_flow_dataset.columns.str.lower()
    cash_flow_dataset = cash_flow_dataset.map(lambda s: s.lower() if type(s) == str else s)

    # Load units of measurement variations
    with open(UNITS_OF_MEASUREMENT_VARIATIONS_PATH) as json_reader:
        unit_variations = json.load(json_reader)

    cash_flow_dataset['description'] = cash_flow_dataset['description'].apply(
        lambda x: standardize_units(x, unit_variations)
    )


    # Predict classes using trained model
    cash_flow_dataset ['class'] = predict_classes(cash_flow_dataset['description'])


    # Import units of measurements and extract amounts if present in the description field
    units_of_measurment_dataset = pd.read_csv(UNITS_OF_MEASUREMENT_DATASET_PATH)
    units_of_measurment_dataset.columns = units_of_measurment_dataset.columns.str.lower()
    units_of_measurment_dataset = units_of_measurment_dataset.map(lambda s: s.lower() if type(s) == str else s)

    all_units='|'.join(list(units_of_measurment_dataset['unit']))
    pattern = rf'([\d.,]+)\s*({all_units})\b'

    cash_flow_dataset[['amount', 'unit']] = cash_flow_dataset['description'].str.extract(pattern, flags=re.IGNORECASE)


    # Parse amount fields and standardize its unit of measurement
    cash_flow_dataset = parse_amount_field(dataset=cash_flow_dataset, field='amount')
    cash_flow_dataset = parse_amount_field(dataset=cash_flow_dataset, field='amount_eur')
    cash_flow_dataset = standardize_units_of_measurement(cash_flow_dataset)
    
    # Only keep relevant columns
    cash_flow_dataset = cash_flow_dataset[['description','gl_account','amount','unit','amount_eur','class']]

    # Get ESG indicators conversion rates and add info to dataset
    ESG_conversion_rates = pd.read_csv(ESG_CONVERTION_RATES_PATH)
    ESG_conversion_rates.columns = ESG_conversion_rates.columns.str.lower()
    cash_flow_dataset = cash_flow_dataset.merge(ESG_conversion_rates, on='class')
    
    # Calculate amounts with cost of purchase if not available from cash flow statement
    cash_flow_dataset['amount'] = cash_flow_dataset['amount'].fillna(cash_flow_dataset['amount_eur'] * cash_flow_dataset['cost_of_purchase'])
    cash_flow_dataset['unit'] = cash_flow_dataset['unit'].fillna(cash_flow_dataset['standard_unit_of_measure'])
    
    # Calculate total revenue as sum of sales of products
    revenue = cash_flow_dataset.loc[cash_flow_dataset['gl_account'] == 'sales of products']['amount_eur'].sum()
        
    # Calculate total waste produced and waste intensity
    waste_disposal_dataframe = cash_flow_dataset.loc[cash_flow_dataset['class'] == 'Waste Disposal'].copy()
    waste_disposal_dataframe['amount'] = waste_disposal_dataframe['amount_eur'] / waste_disposal_dataframe['cost_of_purchase']
    total_waste_produced = waste_disposal_dataframe['amount'].abs().sum()
    waste_intensity = round((total_waste_produced*1000)/revenue, 2)
    
    # Calculate total water consumed and water intensity
    water_bills_dataframe = cash_flow_dataset.loc[cash_flow_dataset['class'] == 'Water'].copy()
    water_bills_dataframe['amount'] = water_bills_dataframe['amount_eur'] / water_bills_dataframe['cost_of_purchase']
    total_water_consumed = water_bills_dataframe['amount'].abs().sum()
    water_intensity = round(total_water_consumed/revenue, 2)
    
    # print all calculated ESG indicators
    print("="*80)
    print("CALCULATED ESG INDICATORS\n")
    print(F"WASTE INTENSITY: {waste_intensity}")
    print(F"WATER INTENSITY: {water_intensity}")
    print("="*80)


if __name__ == "__main__":
    main()
