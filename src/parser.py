"""parser.py

This module parses the cash flow statement.

It classifies all the entries and calculates ESG indicators.
"""

from pathlib import Path
import re
import json
from enum import Enum, auto
from typing import List

import pickle
import pandas as pd
from pandas.api.types import is_numeric_dtype


PROJECT_DIR = Path(__file__).parent.parent
INPUT_DATASET_PATH = PROJECT_DIR / 'data' / 'cash_flow_statements' / 'Syngenta_2023_Cash_Flow_Statement.xlsx'
UNITS_OF_MEASUREMENT_DATASET_PATH = PROJECT_DIR / 'data' / 'parser' / 'units_of_measurement.csv'
CONVERSION_RATES_PATH = PROJECT_DIR / 'data'/ 'parser' / 'conversion_rates.csv'
UNITS_OF_MEASUREMENT_VARIATIONS_PATH = PROJECT_DIR / 'data'/ 'parser' / 'units_of_measurement_variations.json'
MODEL_PATH = PROJECT_DIR / 'models' / 'cash_flow_classifier.pkl'
ESG_CONVERTION_RATES_PATH = PROJECT_DIR / 'data' / 'parser' / 'ESG_indicators_conversion_rates.csv'

SCOPE_3_PERCENTAGE=0.973


class NumberNotations(Enum):
    """
    This class is an enum for supported number format notations.
    """

    EU = auto()
    US = auto()


def predict_classes(texts:List)->List:
    """Predict classes from the description field.
    
    Args:
        texts: List containing descriptions.

    Returns:
        List containing predicted labels.
    """
    # Load the model and extract elements
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)

    model = pipeline['model']
    vectorizer = pipeline['vectorizer']
    label_encoder = pipeline['label_encoder']
    confidence_threshold = pipeline['confidence_threshold']
    fallback_label = pipeline['fallback_label']

    # Transform text to features
    x = vectorizer.transform(texts)

    # Get predictions and probabilities
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)

    # Get confidence scores
    confidence_scores = probabilities.max(axis=1)

    # Decode labels
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Apply confidence threshold and fallback
    final_predictions = []
    for label, confidence in zip(predicted_labels, confidence_scores):
        final_predictions.append(fallback_label if confidence < confidence_threshold else label)

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

def extract_amount_from_description(dataframe: pd.DataFrame)->pd.DataFrame:
    """Extract amount from description field from cash flow statement dataframe standardizing it.

    Args:
        dataframe: cash flow statement dataframe.
    
    Returns:
        pd.DataFrame: dataframe containing two new columns 'amount' and 'unit'.
    """

    # Import units of measurements and extract amounts if present in the description field
    units_of_measurement_dataset = pd.read_csv(UNITS_OF_MEASUREMENT_DATASET_PATH)
    units_of_measurement_dataset.columns = units_of_measurement_dataset.columns.str.lower()
    units_of_measurement_dataset = units_of_measurement_dataset.map(lambda s: s.lower() if isinstance(s, str) else s)

    all_units='|'.join(list(units_of_measurement_dataset['unit']))
    pattern = rf'([\d.,]+)\s*({all_units})\b'

    dataframe[['amount', 'unit']] = dataframe['description'].str.extract(pattern, flags=re.IGNORECASE)


    # Parse amount fields and standardize its unit of measurement
    dataframe = parse_amount_field(dataset=dataframe, field='amount')
    dataframe = parse_amount_field(dataset=dataframe, field='amount_eur')
    dataframe = standardize_units_of_measurement(dataframe)

    return dataframe


def add_amounts_from_cost(dataframe: pd.DataFrame)->pd.DataFrame:
    """Add amount where missing, calculating it from the cost ot purchase.

    Args:
        dataframe: DataFrame containing the cash flow statement.

    Returns:
        pd.DataFrame with correct amounts based on cost of purchase.
    """
    esg_conversion_rates = pd.read_csv(ESG_CONVERTION_RATES_PATH)
    esg_conversion_rates.columns = esg_conversion_rates.columns.str.lower()
    dataframe = dataframe.merge(esg_conversion_rates, on='class')

    # Calculate amounts with cost of purchase if not available from cash flow statement
    dataframe['amount'] = dataframe['amount'].fillna(dataframe['amount_eur'] * dataframe['cost_of_purchase'])
    dataframe['unit'] = dataframe['unit'].fillna(dataframe['standard_unit_of_measure'])

    return dataframe


def main():
    """
    The main function:
    1. reads cash flow statement.
    2. predicts classes using previously trained XGBoost model.
    3. tries extracting amounts from descriptions.
    4. calculates amounts with cost of purchase where applicable if not in description.
    5. calculates ESG indicators and returns them.
    """
    # Read cash flow dataset
    cash_flow_dataset = pd.read_excel(INPUT_DATASET_PATH)
    cash_flow_dataset.columns = cash_flow_dataset.columns.str.lower()
    cash_flow_dataset = cash_flow_dataset.map(lambda s: s.lower() if isinstance(s, str) else s)

    # Load units of measurement variations
    with open(UNITS_OF_MEASUREMENT_VARIATIONS_PATH, encoding='utf8') as json_reader:
        unit_variations = json.load(json_reader)

    cash_flow_dataset['description'] = cash_flow_dataset['description'].apply(
        lambda x: standardize_units(x, unit_variations)
    )


    # Extract amounts from description field if possible
    cash_flow_dataset = extract_amount_from_description(cash_flow_dataset)

    # Predict classes using trained model
    cash_flow_dataset ['class'] = predict_classes(cash_flow_dataset['description'])

    # Only keep relevant columns
    cash_flow_dataset = cash_flow_dataset[['description','gl_account','amount','unit','amount_eur','class']]

    # Add amounts if not already present
    cash_flow_dataset = add_amounts_from_cost(cash_flow_dataset)

    # Calculate total revenue as sum of sales of products
    revenue = cash_flow_dataset.loc[cash_flow_dataset['gl_account'] == 'sales of products']['amount_eur'].sum()


    # Calculate total waste produced and waste intensity
    waste_disposal_dataframe = cash_flow_dataset.loc[cash_flow_dataset['class'] == 'Waste Disposal'].copy()
    waste_disposal_dataframe['amount'] = waste_disposal_dataframe['amount_eur'] / waste_disposal_dataframe['cost_of_purchase']
    total_waste_produced = waste_disposal_dataframe['amount'].abs().sum()
    waste_intensity = round((total_waste_produced * 1000) / revenue, 2)


    # Calculate total water consumed and water intensity
    water_bills_dataframe = cash_flow_dataset.loc[cash_flow_dataset['class'] == 'Water'].copy()
    water_bills_dataframe['amount'] = water_bills_dataframe['amount_eur'] / water_bills_dataframe['cost_of_purchase']
    total_water_consumed = water_bills_dataframe['amount'].abs().sum()
    water_intensity = round(total_water_consumed/revenue, 2)


    # Calculate total energy absorbed and energy intensity
    electricity_bills_dataframe = cash_flow_dataset.loc[cash_flow_dataset['class'] == 'Electricity'].copy()
    electricity_bills_dataframe['amount'] = electricity_bills_dataframe['amount_eur'] / electricity_bills_dataframe['cost_of_purchase']
    total_energy_absorbed = electricity_bills_dataframe['amount'].abs().sum() * 0.0000036
    energy_intensity = round((total_energy_absorbed) / (revenue / 1000000),2)


    # Calculate SCOPE 2 emissions
    scope_2_dataframe = cash_flow_dataset.loc[cash_flow_dataset['class'] == 'Electricity'].copy()
    scope_2_dataframe['amount'] = scope_2_dataframe['amount_eur'] / scope_2_dataframe['cost_of_purchase']
    scope_2_dataframe['co2_eq_produced'] = scope_2_dataframe['amount'] * scope_2_dataframe['co2eq']
    scope_2_co2eq = round(scope_2_dataframe['co2_eq_produced'].abs().sum() / 1000000)


    # Calculate scope 1 and 3 emissions
    scope_1_3_dataframe = cash_flow_dataset.loc[cash_flow_dataset['class'].isin(['Other','Waste Disposal'])].copy()
    scope_1_3_dataframe['co2_eq_produced'] = scope_1_3_dataframe['amount_eur'] / scope_1_3_dataframe['co2eq'].max()
    scope_1_3_co2eq = scope_1_3_dataframe['co2_eq_produced'].abs().sum()

    # Divide previous calculation into scope 1 and scope 3
    scope_1_co2eq = round((scope_1_3_co2eq*(1-SCOPE_3_PERCENTAGE)) / 1000000)
    scope_3_co2eq = round((scope_1_3_co2eq*SCOPE_3_PERCENTAGE) / 1000000)
    total_ghg = scope_1_co2eq + scope_2_co2eq + scope_3_co2eq
    ghg_emission_intensity = round(total_ghg/(revenue / 100000),2)


    # print all calculated ESG indicators
    print("="*80)
    print("CALCULATED ESG INDICATORS\n")
    print(f"SCOPE 1 GHG EMISSIONS: {scope_1_co2eq} 000s of tonnes")
    print(f"SCOPE 2 GHG EMISSIONS: {scope_2_co2eq} 000s of tonnes")
    print(f"SCOPE 3 GHG EMISSIONS: {scope_3_co2eq} 000s of tonnes")
    print(f"TOTAL GHG EMISSIONS: {total_ghg} 000s of tonnes")
    print(f"GHG EMISSION INTENSITY: {ghg_emission_intensity} 000s of tonnes")
    print(f"TOTAL ENERGY ABSORBED: {round(total_energy_absorbed)} TJ")
    print(f"ENERGY INTENSITY: {energy_intensity} TJ/sales")
    print(f"WATER INTENSITY: {water_intensity} liters/sales")
    print(f"WASTE INTENSITY: {waste_intensity} g/sales")
    print("="*80)


if __name__ == "__main__":
    main()
