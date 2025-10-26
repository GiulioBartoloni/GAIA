"""Parse demo with streamlit"""


import json
from enum import Enum, auto
import re

import pandas as pd
import streamlit as st

INPUT_DATASET_PATH = "data/input_dataset_v.1.xlsx"
UNITS_OF_MEASUREMENT_DATASET_PATH = "data/units_of_measurement.csv"
CONVERSION_RATES_PATH = "data/conversion_rates.csv"
UNITS_OF_MEASUREMENT_VARIATIONS_PATH = "data/units_of_measurement_variations.json"


class NumberNotations(Enum):
    """Enum class for number notation options"""
    EU = auto()
    US = auto()


with open(UNITS_OF_MEASUREMENT_VARIATIONS_PATH, encoding='utf-8') as json_reader:
    unit_variations = json.load(json_reader)


def standardize_units(text:str)->str:
    for variation, standard_unit in unit_variations.items():
        if variation in text:
            text = text.replace(variation,standard_unit)

    return text


def parse_amount_field(dataset:pd.DataFrame, field:str, number_format:NumberNotations=NumberNotations.US)->pd.DataFrame:
    if number_format == NumberNotations.US:
        dataset[field] = dataset[field].str.replace(',', '', regex=False)
    else:
        dataset[field] = dataset[field].str.replace('.', '', regex=False)
        dataset[field] = dataset[field].str.replace(',', '.', regex=False)

    dataset[field] = pd.to_numeric(dataset[field], errors='coerce')
    return dataset


def standardize_units_of_measurement(dataset:pd.DataFrame)->pd.DataFrame:
    df=pd.read_csv(CONVERSION_RATES_PATH)
    result = pd.merge(dataset,df, how='left')

    result['Unit'] = result['TargetUnit']
    result['Amount'] = result['Amount'] * result['ConversionRate']

    result = result.drop(['TargetUnit', 'ConversionRate'], axis=1)

    return result


def extract_description(dataset:pd.DataFrame)->pd.DataFrame:
    filtered_dataset = dataset.loc[dataset['GL_Account (General Ledger)'].isin(['raw materials expense','utilities expense'])]

    return filtered_dataset


st.title("Cash flow statement parser")

uploaded_file = st.file_uploader(
    "Upload cash flow statement", accept_multiple_files=False, type="xlsx"
)

if uploaded_file is not None:
    # Read the Excel file
    cash_flow_dataset = pd.read_excel(uploaded_file)
    cash_flow_dataset = cash_flow_dataset.map(lambda s: s.lower() if isinstance(s,str) else s)

    units_of_measurment_dataset = pd.read_csv(UNITS_OF_MEASUREMENT_DATASET_PATH)
    units_of_measurment_dataset = units_of_measurment_dataset.map(lambda s: s.lower() if isinstance(s,str) else s)

    st.dataframe(cash_flow_dataset)

    cash_flow_dataset['Description'] = cash_flow_dataset['Description'].apply(standardize_units)

    all_units='|'.join(list(units_of_measurment_dataset['Unit']))
    pattern = rf'([\d.,]+)\s*({all_units})\b'

    cash_flow_dataset[['Amount', 'Unit']] = cash_flow_dataset['Description'].str.extract(pattern, flags=re.IGNORECASE)

    cash_flow_dataset = parse_amount_field(dataset=cash_flow_dataset, field='Amount')
    cash_flow_dataset = parse_amount_field(dataset=cash_flow_dataset, field='Amount_EUR')

    cash_flow_dataset = standardize_units_of_measurement(cash_flow_dataset)

    cash_flow_dataset = extract_description(cash_flow_dataset)

    cash_flow_dataset = cash_flow_dataset[['Description','Amount','Unit','Amount_EUR']]

    st.dataframe(cash_flow_dataset)