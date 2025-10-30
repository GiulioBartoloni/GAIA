# GAIA

# Project folder structure
```
GAIA
├── README.md
├── architecture_documentation
│   ├── components.md
│   └── image.png
├── data
│   ├── cash_flow_statements
│   │   ├── input_dataset_v.1.xlsx
│   │   └── synthetic_dataset.xlsx
│   ├── conversion_rates.csv
│   ├── units_of_measurement.csv
│   ├── units_of_measurement_variations.json
│   └── xgboost_test_data.csv
├── meta
│   ├── architecture.drawio
│   └── architecture.png
├── requirements.txt
└── src
│   └── parser_demo.py
├── classificator_test.ipynb
└── parser.ipynb
```
The main files are:
- `architecture_documentation`: contains info on the possible system architecture.
- `data`: contains all datasets used.
- `meta`: architecture graph files.
- `requirements.txt`: lists all python libraries necessary to run all programs.
- `src`: contains all the `.py` code.


# Instructions to run this code
The main components are:
- `src/classificator_test.ipynb`: this notebook contains the test done with XGBoost and TF-IDF vectorization for string classification.
- `src/parser.ipynb`: this notebook contains the code for the _Cash Flow Statement_ parser.
- `src/parser_demo.py`: this code is a streamlit app that uses the same logic in `src/parser.ipynb`, but with a prettier interface.


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
Now we need to install the kernel to run notebooks:
```
python -m ipykernel install --user --name GAIA --display-name "GAIA venv"
```
(_Reloading VS code window or restarting it may be necessary_)


## Notebooks
Both notebooks `parser.ipynb` and `classificator_test.ipynb` define macros at the beginning with file paths that can be customized if needed.

Other than that, no additional info should be required.

## Demo file
This file cannot be run like a regular python file. The command required is:
```bash
streamlit run src/parser_demo.py
```
With this default command, the app will be hosted on `localhost:8501`. The port and hostname can be customized, but for demo purposes it should be fine.