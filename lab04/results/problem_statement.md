# Production ETL Problem Formulation

**Lead Data Engineer:** Miray Büyüktepe (ID: 73681)  
**Dataset:** Adult Census Income (OpenML, v2)  

### Objective & Architecture
Develop an MLOps-ready ETL ingestion, validation, and transformation pipeline predicting whether individual income exceeds $50K.

### Constraints & Evaluation
- 14 raw predictors.
- F1-score evaluated as the principal benchmark due to the 76/24 majority class skew.
- Strict isolation of training transformations from evaluation data.
