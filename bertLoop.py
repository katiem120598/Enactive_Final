from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os

def perform_bert(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
    inputs = tokenizer(text, return_tensors="pt")
    labels = torch.tensor([0])
    outputs = model(**inputs, labels=labels)
    loss, logits = outputs[:2]
    return logits.softmax(dim=-1)[0].tolist()

def analyze_text(csv_file):
    results_df = pd.DataFrame(columns=['left', 'center', 'right', 'expected'])
    df = pd.read_csv(csv_file, skiprows=[1, 2])
    
    for index, row in df.iterrows():
        text = str(row.iloc[18])  # Ensure the text is a string
        if pd.isna(text) or text.strip() == "":
            continue  # Skip empty or NaN text entries
        results = perform_bert(text)
        expected_value = results[0] * 0 + results[1] * 1 + results[2] * 2
        new_row = pd.DataFrame({'left': [results[0]], 'center': [results[1]], 'right': [results[2]], 'expected': [expected_value]})
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    # Extract the base name of the CSV file and append '_results'
    base_name = os.path.splitext(csv_file)[0]
    results_file = f"{base_name}_results.csv"
    
    # Ensure the results directory exists
    os.makedirs('./results', exist_ok=True)
    
    # Save the results DataFrame to a new CSV file in the results directory
    results_df.to_csv(f"./results/{results_file}", index=False)
    
    return results_df

# Example usage
human_text = analyze_text('human_generated.csv')
ai_text = analyze_text('ai_generated.csv')
print(human_text)
print(ai_text)