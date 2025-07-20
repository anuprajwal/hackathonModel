from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

app = Flask(__name__)

model_dir = "../bert_epoch_4"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

label_map = {
    0: "pro",
    1: "anti",
    2: "info",
    3: "slang",
    4: "other"
}



def predict_intent(text):
    # Tokenize the input text
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    return label_map[predicted_class]


@app.route('/is-running')
def is_running():
    return "Flask server is running on port 9000!"

@app.route('/', methods=['GET', 'POST'])
def home():
    
    return render_template("question.html")

@app.route('/answer', methods=['GET', 'POST'])
def answer_it():
    answer = None
    if request.method == 'POST':
        question = request.form.get('question')
        print("Received question:", question)  # Placeholder: you can replace this later
        answer = predict_intent(question.lower())
    return render_template('question.html', answer=answer)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)

