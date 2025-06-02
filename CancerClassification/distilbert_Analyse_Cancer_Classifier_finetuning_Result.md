## Comparative Study: Before and After Fine-Tuning

This table summarizes the performance of a binary classification model (Cancer vs. Non-Cancer) *before* and *after* LLM fine-tuning. 

**Jupyter Notebook:** [distilbert_cancer_classification_finetuning.ipynb](https://github.com/Git-PratikVyas/Finetuning-LORA/blob/main/CancerClassification/distilbert_cancer_classification_finetuning.ipynb)


| Metric                     | Before Fine-Tuning | After Fine-Tuning | Change   | Notes                                                                 |
| :------------------------- | :-----------------: | :----------------: | :------- | :--------------------------------------------------------------------- |
| **Overall Performance**    |                    |                   |          |                                                                       |
| Accuracy                   | 0.495              | 0.565              | +0.070   | Accuracy Has Now improved, showing potential training.                        |
| **Class-Specific Accuracy** |                    |                   |          |                                                                       |
| Accuracy for Label 0      | 0.000              | 0.693              | +0.693   | Signfificant Improvement, model can distinguish negative                     |
| Accuracy for Label 1      | 1.000              | 0.434              | -0.566   | Bias is reducing, it has reduced memorization      |
| **Classification Report**  |                    |                   |          |                                                                       |
| Precision (Label 0)        | 0.00               | 0.56               | +0.56    |     Model  is more now accurate            |
| Recall (Label 0)           | 0.00               | 0.69              | +0.69    |     Now model identifies the true classes in testing dataset             |
| F1-Score (Label 0)         | 0.00               | 0.62               | +0.62    | Better F1 score.          |
| Precision (Label 1)        | 0.49               | 0.58               | +0.09     | Now has some slight precision           |
| Recall (Label 1)           | 1.00               | 0.43               | -0.57     | The metric is changed since there isn't full bias                        |
| F1-Score (Label 1)         | 0.66               | 0.50               | -0.16    | The f1 score is better now from dataset   




- **Next Step** 

  **Loss Function Balancing (Weighted Loss):**
  you can assign different weights to each class. Give a higher weight to the Non-Cancer class to penalize misclassifications more heavily. You want to have it well

  **Adjust Classification Threshold**
