# EnergyExpenditureEst

Energy Expenditure Estimation Models

## Overview

Builds and evaluates models to predict human energy expenditure (“cosmed” values) from physiological features. Includes data, report slides, and scripts.

## Features

- ARFF datasets (`0.arff` … `9.arff`)  
- Python notebook (`projectfinal.ipynb`) for EDA and modeling  
- Command‑line script (`projectfinal.py`) for batch runs  
- Presentation (`Energy ExpendIture Estimation-sunum.ppsx`)

## Prerequisites

- Python 3.8+  
- pandas, numpy, scikit-learn, matplotlib

## Installation

```bash
git clone https://github.com/bahademircioglu/EnergyExpenditureEst.git
cd EnergyExpenditureEst
pip install -r requirements.txt
```

*(requirements.txt: `pandas numpy scikit-learn matplotlib scipy arff`)*

## Usage

Run the notebook for exploratory analysis:

```bash
jupyter notebook projectfinal.ipynb
```

Or execute the script:

```bash
python projectfinal.py --input 0.arff --output results.csv
```

## Project Structure

```
EnergyExpenditureEst/
├── *.arff
├── projectfinal.ipynb
├── projectfinal.py
├── Energy ExpendIture Estimation-sunum.ppsx
├── LICENSE (GPL-3.0)
└── README.md
```

## Contributing

Feel free to open issues or PRs.

## License

GPL‑3.0 License.
