# RoBERTa for Sequence Classification on AG-NEWS dataset

## Pre-requisites:
1) Install torch from https://pytorch.org/
2) Install hugging-face transformers using `pip install transformers`
3) Install hugging-face datasets using `pip install datasets`
4) Install TQDM progress bar using `pip install tqdm`
5) Install matplotlib using `pip install matplotlib`
6) Install pandas using `pip install pandas`
7) Install scikit-learn using `pip install scikit-learn` (required for precision,recall and f1 score calculation)

# AG-NEWS dataset
The ‘ag_news’ dataset consists of totally 120000 train rows and 7600 test rows, and consist of 2 columns – ‘text’ and ‘label’. The ‘text’ column denotes the news and ‘label’ denotes the category of the news. The news is classified as 4 types:
| Label      | Integer Representation |
| ----------- | ----------- |
| World      | 0       |
| Sports   | 1        |
| Business   | 2        |
| Sci/Tech   | 3        |

We load this dataset using datasets.load_dataset and shuffle it.

# Training Dataset and model
- The input to the model consists of 2 values: input sequence and the attention mask.
- The tokenizer_helper function will return the tokenized output for a sentence. To tokenize the entire dataset, we use the map function of the dataset.
- We then create a TensorDataset using the encoded input sequence, attention masks, and labels. Finally, We create a Dataloader for train and test dataset using the above TensorDataset and RandomSampler.
- The default RobertaForSequenceClassification model is designed for binary classification. Thus, we change the final layer to output size 4.

# Training Procedure
- We finetune the model on the given down-stream task for 5 epochs, and batch size of 64.
- We use the Adam optimizer with learning rate of 2e-5
- To better finetune the model, we introduce an Exponential Learning Rate Scheduler.
- Exponential LR Scheduler decays the learning rate by ‘gamma’ for each parameter group, on every epoch.
- This helps the model approach the global minima better!
- In order to prevent gradient explosion, we also clip the gradients using clip_grad_norm_()
- The graphs plotted using macro_precision, macro_recalls, and macro_f1s show that the model performance peaks at 4 epochs.
- We finally achieve a validation accuracy 95.86%

# Insights and Observation
- Often at times, we may need a LR Scheduler to better navigate to the global minima.
- Micro Averaging for Precision/Recall/F1-Score is often influenced by the majority class, and thus can be a bad representative of the performance of the model.
- Macro average is better for this case since it give weight for each individual class instead
- At Epoch 5, validation loss has increased a little bit. This can also be seen in a the decrease on validation accuracy, precision, recall and f1 scores. This shows that the gradient descent has overshoot too much. This can be fixed by introducing a MultiStepLR with a step at epoch 4.

