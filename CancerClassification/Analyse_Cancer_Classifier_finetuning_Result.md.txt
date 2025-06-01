## Comparative Study: Before and After Fine-Tuning

This table summarizes the performance of a binary classification model (Cancer vs. Non-Cancer) *before* and *after* LLM fine-tuning. 


| Metric             | **Before Fine-Tuning**| **After Fine-Tuning** | Change          | Interpretation                                                                                                                                    |
| ------------------ | ------------------ | ----------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Accuracy           | 0.480              | 0.760             | +0.280 (28%)    | Overall correctness improved, suggesting the fine-tuning process did help the model. 
| Precision (Cancer) | 1.00               | 1.00              | 0.00            | Model correctly predicts that something has cancer than non-cancer
| Recall (Cancer)    | 0.48               | 0.76              | +0.28 (28%)     | Identifies the majority of actual cancer cases now, but there's still room for improvement
| F1-Score (Cancer)  | 0.65               | 0.86              | +0.21 (21%)     | Good indication of balanced precision and recall *for the Cancer class only*. As high precision and recall does point to that.
| Precision,Recall (Non-Cancer) | 0.00               | 0.00              | 0.00            | Cannot identify any Non-Cancer instances



- **Accuracy:** The accuracy improvement is the most positive outcome. The model is now much better at distinguishing between Cancer and Non-Cancer overall.

- **Need Improvement in Recall**

  **Precision and Recall Balance:** The precision/recall results for the "Non-Cancer" class (0) need to address. The model is struggling to  predicting Non-Cancer.


- **Next Step** 

  **Loss Function Balancing (Weighted Loss):**
  you can assign different weights to each class. Give a higher weight to the Non-Cancer class to penalize misclassifications more heavily. You want to have it well

  **Adjust Classification Threshold**
