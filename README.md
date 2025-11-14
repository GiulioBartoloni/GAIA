![alt text](meta/GaiaLogo.png)

# Project folder structure
```
GAIA
├── architecture_documentation
│   ├── components.md
│   └── image.png
├── data
│   ├── cash_flow_statements
│   │   └── Syngenta_2023_Cash_Flow_Statement.xlsx
│   ├── parser
│   │   ├── conversion_rates.csv
│   │   ├── ESG_indicators_conversion_rates.csv
│   │   ├── units_of_measurement.csv
│   │   └── units_of_measurement_variations.json
│   └── xgboost
│       └── training_dataset.csv
├── meta
│   ├── architecture.drawio
│   ├── architecture.png
│   └── GaiaLogo.png
├── models
│   └── cash_flow_classifier.pkl
├── README.md
├── requirements.txt
└── src
    ├── classificator_training.py
    └── parser.py
```
The main files are:
- `architecture_documentation`: contains info on the possible system architecture.
- `data`: contains all datasets used.
- `meta`: architecture graph files.
- `requirements.txt`: lists all python libraries necessary to run all programs.
- `src`: contains all the `.py` code.


# Instructions to run this code
The main components are:
- `src/classificator_training.py`: this script trains the XGBoost classifier using the training data.
- `src/parser.py`: this script runs the parser to calculate ESG indicators from the given cash flow statement.


Before running anything, you need to setup a python virtual environment and install required dependencies:
```bash
cd GAIA
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```
or if you're using a package manager like `uv`:
```bash
cd GAIA
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```



## Running Python scripts
The first script you need to run in order to compute ESG indicators is `src/classificator_training`.
```
python src/classificator_training.py
```
This will create a `.pkl` file in the `models` folder. It will be then used for classification.

Then you can run `src/parser.py`.
```
python src/parser.py
```

You will get calculated ESG indicators as an output.

To get more information on any of the functions, check out the docstrings or use the `help` command in a python instance.
