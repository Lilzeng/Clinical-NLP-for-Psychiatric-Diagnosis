# 🧠 Clinical Diagnosis Classification with Distilled and Compressed Transformers

This project explores the use of transformer-based models for clinical diagnosis classification using discharge notes from the MIMIC-III dataset. The task is a binary classification problem: determine whether a patient was admitted for a psychiatric reason based on free-text clinical notes.

We start by fine-tuning a large pretrained language model (Bio_ClinicalBERT), then distill its knowledge into a much smaller BERT-Tiny model. We further explore compression through quantization and structured pruning. Model performance, size, and inference speed are all evaluated and compared.

--

### 🧪 Task Definition

**Goal:** Classify discharge summaries as *psychiatric admission* or *non-psychiatric*.  
**Labeling:** Admissions are labeled as psychiatric if any ICD-9 diagnosis code falls between `290–319`.  
**Dataset:** MIMIC-III `noteevents` and `diagnoses_icd` tables, filtered via BigQuery.

--

### 📁 Folder Structure and Execution Order

The Jupyter notebooks should be executed in the following order:

1. **`process_data.ipynb`**  
   Cleans and processes the MIMIC-III `noteevents` data extracted from Google BigQuery. Selects relevant note sections (e.g., *History of Present Illness*, *Past Medical History*) and removes noise such as placeholders for personal information.

2. **`train_teacher_model.ipynb`**  
   Fine-tunes the [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) model on the classification task using the processed discharge notes.

3. **`train_student_model_from_scratch.ipynb`**  
   Fine-tunes a [BERT-Tiny](https://huggingface.co/gaunernst/bert-tiny-uncased) model on the same dataset *without* guidance from the teacher model.

4. **`distill_student_model.ipynb`**  
   Distills BERT-Tiny using logits from the fine-tuned Bio_ClinicalBERT teacher model. This improves performance of the small model with little added computational cost.

5. **`prune_student_model.ipynb`**  
   Applies structured pruning to the distilled BERT-Tiny model by removing attention heads (1 out of 2 heads per layer). Retraining is done post-pruning.

6. **`quantize_student_model.ipynb`**  
   Performs post-training dynamic quantization on both the distilled BERT-Tiny and the distilled + pruned BERT-Tiny models to reduce size and speed up inference.

7. **`measure_models_size.ipynb`**  
   Measures and compares the on-disk size of each model variant.

8. **`measure_models_speed.ipynb`**  
   Measures average per-sample inference time for each model on CPU to ensure consistent comparison.

--

### 🧠 Model Comparison

| Model                        | Best Accuracy | Best F1 Score | Inference Speed (ms) | Size (MB) |
| ---------------------------- | -------- | -------- | ---------- | --------- |
| Bio-ClinicalBERT (Teacher)  | 0.7929   | 0.6740   | 1795.76 &plusmn; 9.40   | 413.25     |
| BERT-Tiny (Distilled)        | 0.7810   | 0.6719   | 18.00 &plusmn; 0.62    | 16.75     |
| BERT-Tiny (Trained Directly) | 0.7821   | 0.6523   | 17.57 &plusmn; 0.38    | 16.75      |
| BERT-Tiny (Quantized)        | 0.7812   | 0.6722   | 17.67 &plusmn; 0.34    | 15.59      |
| BERT-Tiny (Pruned)           | 0.7631   | 0.6281   | 13.04 &plusmn; 0.30    | 16.50     |
| BERT-Tiny (Pruned + Quantized)   | 0.7683   | 0.6279   | 11.77 &plusmn; 0.24    | 15.52      |

✅ **Best trade-off**: BERT-Tiny (quantized) — near identical performance to the teacher model at a fraction of the size and inference time.

--

### 🔍 Observations

- **Distillation is highly effective**: Distilled BERT-Tiny nearly matches the teacher’s F1 score while being 100x faster and 25x smaller.
- **Quantization yields modest gains**: Dynamic quantization slightly reduces model size and speeds up inference without hurting performance.
- **Pruning is fragile**: Structured pruning (head removal) significantly degrades performance, even after retraining, and offers little in size reduction.
- **Noise in labels and text**: The clinical notes often mention mental health conditions without being the primary diagnosis, causing label ambiguity. This is a key source of misclassification, not model error per se.
- **Dataset limitations**: The data comes from ICU admissions. Psychiatric diagnoses are often secondary or incidental, limiting the clarity of the classification task.

--

### 📌 Requirements

- MIMIC-III data access via Google BigQuery
- Python 
- PyTorch
- Hugging Face Transformers
- Datasets 4.0.0 (only this version will work)
- scikit-learn
- Pandas 
- NumPy
- tqdm
- Jupyter notebooks

--

### 👨‍🔬 Acknowledgments

- MIMIC-III (MIT Lab for Computational Physiology)
- [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- [BERT-Tiny](https://huggingface.co/gaunernst/bert-tiny-uncased)

