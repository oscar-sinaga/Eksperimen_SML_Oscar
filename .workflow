name: Automate Data Preprocessing

on:
  push:
    paths:
      - 'predictive_maintenance_raw/**'
      - 'preprocessing/automate_*.py'
      - '.github/workflows/preprocessing.yml'
  workflow_dispatch: # Manual trigger

jobs:
  data-preprocessing:
    runs-on: ubuntu-latest

    steps:
      - name: 1. Checkout Repository
        uses: actions/checkout@v4

      - name: 2. Set up Python 3.12.7
        uses: actions/setup-python@v5
        with:
          python-version: '3.12.7'

      - name: 3. Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install ipykernel==7.2.0 kaggle==2.0.1 mlflow==2.19.0 seaborn==0.13.2 unzip==1.0.0

      - name: 4. Run Automate Preprocessing Script
        run: |
          # Ganti Nama-siswa sesuai dengan nama file Anda!
          python preprocessing/automate_Oscar.py

      - name: 5. Upload Processed Data as Artifact (BUKTI ADVANCE)
        uses: actions/upload-artifact@v4
        with:
          name: Dataset-Siap-Latih-dan-Scaler
          path: preprocessing/predictive_maintenance_preprocessing/
          retention-days: 7

      - name: 6. Commit and Push Changes to Repo
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add preprocessing/predictive_maintenance_preprocessing/predictive_maintenance_processed.csv
          git add preprocessing/predictive_maintenance_preprocessing/scaler.pkl
          # || echo... mencegah error jika tidak ada perubahan data
          git commit -m "Auto-update: Dataset preprocessing via GitHub Actions" || echo "No changes to commit"
          git push