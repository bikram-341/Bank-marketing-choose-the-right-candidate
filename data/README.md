# Bank Marketing Dataset Instructions

Place the Bank Marketing dataset CSV file in this directory (`data/`).

### Supported Datasets:
1. **UCI Bank Marketing Dataset** (`bank-full.csv` or `bank.csv`):
   - Direct download URL: [UCI Machine Learning Repository: Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
   - Source paper: S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 2014.
   - Semicolon-delimited (`;`) or Comma-delimited (`,`) formats are automatically detected.

2. **Kaggle Bank Marketing Dataset**:
   - URL: [Kaggle Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)

### File Detection:
The system automatically looks for:
- `data/bank-full.csv` (~45,211 rows)
- `data/bank.csv` (~4,521 rows)
- `data/bank-additional-full.csv` (~41,188 rows)
- `data/bank_marketing.csv`
- Any custom CSV uploaded directly in the Streamlit application interface.

### Automatic Downloader:
You can also run the bundled download script from the project root:
```bash
python scripts/download_dataset.py
```
This fetches the official standard UCI dataset and extracts `bank-full.csv` and `bank.csv` into `data/`.
