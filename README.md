# Anomaly Detection on NSL-KDD Dataset using Isolation Forest 

This project uses Isolation Forest to detect anomalies in the NSL-KDD dataset, a widely used dataset in cybersecurity.

## How to Run

1. Install requirements:
2. Download the dataset and place `KDDTrain+.txt` in the `dataset/` folder. The dataset has 43 columns, if you downloaded onw that has 42 columns, remember to add a column (difficulty). 
3. Run the code:

## Installation
pip install -r requirements.txt

4. ## Output
- Confusion matrix
- Classification report
  
## Notes
- Anomalies are defined as non-normal traffic.
- This is a basic unsupervised anomaly detection approach.
