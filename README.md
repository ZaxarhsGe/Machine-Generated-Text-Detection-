# 🧠 AI-Generated Text Detection with BERT  

This project was developed as part of my **Bachelor’s Thesis**, focusing on building a **machine learning model** capable of detecting whether a tweet was written by a **human** or **AI (bot)**.  

The implementation leverages **BERT (Bidirectional Encoder Representations from Transformers)** for text classification, trained on a dataset of labeled tweets.  

---

## 🚀 Key Features  
- **Text Classification:** Distinguishes between human-written and AI-generated tweets.  
- **BERT Fine-Tuning:** Uses `bert-base-uncased` for sequence classification.  
- **Data Cleaning & Preprocessing:** Handles text normalization, stopword removal, and tokenization.  
- **Evaluation Metrics:** Includes accuracy, precision, recall, and F1-score.  
- **Visualization:** Data distribution plots using Seaborn and Matplotlib.  
- **GPU Acceleration:** Automatic use of CUDA if available for faster training.  

---

## 🧰 Tech Stack  
- **Languages:** Python  
- **Libraries:** PyTorch, Transformers (Hugging Face), scikit-learn, NLTK, Pandas, NumPy, Seaborn, Matplotlib  
- **Model:** `bert-base-uncased` from Hugging Face Transformers  
- **Dataset:** Twitter dataset (`train.csv`) containing labels: *human* and *bot*  

---

## ⚙️ Data Preprocessing  

The preprocessing pipeline includes:
1. **Loading Data:**  
   ```python
   tweets = pd.read_csv("train.csv", encoding='latin', header=None, delimiter=",", quotechar='"')
   tweets['tweets'] = tweets[1]
   tweets['label'] = tweets[2]
   ```

2. **Label Mapping:**  
   ```python
   label_mapping = {'bot': 1, 'human': 0}
   tweets['generated'] = tweets['label'].map(label_mapping)
   ```

3. **Text Cleaning:**  
   ```python
   def preprocess(text):
       text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
       words = [w.lower() for w in text.split() if w.isalpha()]
       words = [w for w in words if w not in stop_words]
       text = ' '.join(words)
       text = re.sub(r'[^\w\s]', '', text)
       return text
   tweets['preprocess'] = tweets['tweets'].apply(preprocess)
   ```

4. **Data Visualization:**  
   ```python
   sns.countplot(data=tweets, x='generated')
   plt.show()
   ```

---

## 🧠 Model Training  

1. **Train-Test Split:**  
   ```python
   X_train_val, X_test, y_train_val, y_test = train_test_split(
       tweets['preprocess'], tweets['generated'], test_size=0.2, random_state=42
   )
   ```

2. **Tokenization:**  
   ```python
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
   encoded_train = tokenizer(X_train_val.tolist(), padding=True, truncation=True, return_tensors='pt')
   ```

3. **Dataset & DataLoader:**  
   ```python
   train_dataset = TensorDataset(encoded_train['input_ids'], encoded_train['attention_mask'], torch.tensor(y_train_val.values))
   train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
   ```

4. **Model Setup:**  
   ```python
   model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
   optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
   criterion = nn.CrossEntropyLoss()
   ```

5. **Training Loop:**  
   ```python
   for epoch in range(epochs):
       model.train()
       total_loss, correct_preds = 0, 0
       for batch in train_loader:
           input_ids, mask, labels = [b.to(device) for b in batch]
           optimizer.zero_grad()
           outputs = model(input_ids, attention_mask=mask)
           loss = criterion(outputs.logits, labels)
           loss.backward()
           optimizer.step()
   ```

---

## 📊 Evaluation  

After training, the model is evaluated on a validation split:  
```python
val_accuracy = accuracy_score(val_labels, val_preds)
val_precision = precision_score(val_labels, val_preds)
val_recall = recall_score(val_labels, val_preds)
val_f1 = f1_score(val_labels, val_preds)

print(f"Validation Accuracy: {val_accuracy:.2f}")
print(f"Validation Precision: {val_precision:.2f}")
print(f"Validation Recall: {val_recall:.2f}")
print(f"Validation F1-score: {val_f1:.2f}")
```

📈 **Results:**  
- Accuracy: **0.70**  
- Precision: **0.72**  
- Recall: **0.60**  
- F1-Score: **0.66**

---

## 🧪 Inference on Test Data  

Once trained, the model predicts the probability of a tweet being AI-generated:  
```python
model.eval()
predictions = []

with torch.no_grad():
    for i in range(0, len(X_test), 4):
        batch_data = X_test.iloc[i:i+4].tolist()
        tokenized = tokenizer(batch_data, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model(**tokenized)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
        predictions.extend(probs)
```

Results are saved as:  
```python
submission = pd.DataFrame({'generated': predictions})
submission.to_csv('submission.csv', index=False)
```

---

## 🧾 Summary  
This project demonstrates how **transformer-based models** can detect AI-generated content with strong performance metrics. It applies **data preprocessing, fine-tuning, and evaluation** using modern NLP techniques.  

---

## 📚 Acknowledgments  
- [Hugging Face Transformers](https://huggingface.co/transformers)  
- [PyTorch](https://pytorch.org)  
- [NLTK](https://www.nltk.org)  

---

## 👤 Author  
**George Zacharis**  
📩 `gzachrs@gmail.com`  
🎓 BSc(Hons) in Computer Science, Metropolitan College, Greece  
