from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os
from scipy.stats import ttest_ind, f_oneway

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

# Perform t-test on means
human_expected = human_text['expected']
ai_expected = ai_text['expected']

t_stat, p_value = ttest_ind(human_expected, ai_expected)

alpha = 0.05
print(f"T-statistic (means): {t_stat}, P-value (means): {p_value}")
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the means of the two groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the means of the two groups.")

# Perform F-test on variances
f_stat, p_value_var = f_oneway(human_expected, ai_expected)

print(f"F-statistic (variances): {f_stat}, P-value (variances): {p_value_var}")
if p_value_var < alpha:
    print("Reject the null hypothesis: There is a significant difference between the variances of the two groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the variances of the two groups.")


t_stat, p_value = ttest_ind(human_expected, ai_expected)

alpha = 0.05
print(f"T-statistic (means): {t_stat}, P-value (means): {p_value}")
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the means of the two groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the means of the two groups.")

# Perform F-test on variances
f_stat, p_value_var = f_oneway(human_expected, ai_expected)

print(f"F-statistic (variances): {f_stat}, P-value (variances): {p_value_var}")
if p_value_var < alpha:
    print("Reject the null hypothesis: There is a significant difference between the variances of the two groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the variances of the two groups.")

def perform_ttest(csv_file1,csv_file2,column_name):
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    column1 = df1[column_name]
    column2 = df2[column_name]
    t_stat, p_value = ttest_ind(column1, column2)
    f_stat, p_value_var = f_oneway(column1, column2)
    return t_stat, p_value, f_stat, p_value_var

t_stat1,_,_,p_value_var1 = perform_ttest('results/human_review_results.csv','results/ai_review_results.csv','Q62')
t_stat2,_,_,p_value_var2 = perform_ttest('results/human_review_results.csv','results/ai_review_results.csv','Q64')
t_stat3,_,_,p_value_var3 = perform_ttest('results/human_review_results.csv','results/ai_review_results.csv','Q66')

print(f"Q62: T-statistic (means): {t_stat1}, P-value (variances): {p_value_var1}")
print(f"Q64: T-statistic (means): {t_stat2}, P-value (variances): {p_value_var2}")
print(f"Q66: T-statistic (means): {t_stat3}, P-value (variances): {p_value_var3}")

