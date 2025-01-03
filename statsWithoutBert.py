from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os
from scipy.stats import ttest_ind, f_oneway
from scipy.stats import mannwhitneyu
from numpy import mean, std
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def perform_ttest(csv_file1, csv_file2, column_name):
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    if column_name not in df1.columns or column_name not in df2.columns:
        raise ValueError(f"Column {column_name} not found in one of the files")
    
    column1 = df1[column_name]
    column2 = df2[column_name]
    
    # Perform t-test and F-test
    t_stat, p_value = ttest_ind(column1, column2, nan_policy='omit')
    f_stat, p_value_var = f_oneway(column1, column2)
    
    print(f"Data for {column_name} in file 1: {column1.describe()}")
    print(f"Data for {column_name} in file 2: {column2.describe()}")
    
    return t_stat, p_value, f_stat, p_value_var


t_stat_1,p_val_1,f_stat1,p_value_var1 = perform_ttest('results/human_review_results.csv','results/ai_review_results.csv','Q62')
t_stat_2,p_val_2,f_stat2,p_value_var2 = perform_ttest('results/human_review_results.csv','results/ai_review_results.csv','Q64')
t_stat_3,p_val_3,f_stat3,p_value_var3 = perform_ttest('results/human_review_results.csv','results/ai_review_results.csv','Q66')

print(f"Q62: P-value (means): {p_val_1}, P-value (variances): {p_value_var1}")
print(f"Q64: P-value (means): {p_val_2}, P-value (variances): {p_value_var2}")
print(f"Q66: P-value (means): {p_val_3}, P-value (variances): {p_value_var3}")

#create a histogram of results for each of the questions for each of the groups

#create a histogram of column 1 from csv_file1 Q62 in groups of -2, -1, 0, 1, 2
def create_horizontal_bar_chart(csv_file1, csv_file2, column_name):
    # Load the data
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
    # Check if the column exists in both dataframes
    if column_name not in df1.columns or column_name not in df2.columns:
        raise ValueError(f"Column {column_name} not found in one of the files")
    
    # Get the column data
    column1 = df1[column_name]
    column2 = df2[column_name]
    
    # Define bins and calculate frequencies
    bins = [-2, -1.5, -0.5, 0.5, 1.5, 2]
    human_counts, _ = np.histogram(column1, bins=bins)
    ai_counts, _ = np.histogram(column2, bins=bins)
    
    # Define bar positions
    bin_labels = ['Very Liberal', 'Liberal','Neutral', 'Conservative', 'Very Conservative']
    x = np.arange(len(bin_labels))
    
    # Create a horizontal bar chart
    bar_width = 0.4  # Width of the bars
    plt.barh(x - bar_width/2, human_counts, height=bar_width, label='Human', color='blue', alpha=0.7)
    plt.barh(x + bar_width/2, ai_counts, height=bar_width, label='AI', color='orange', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Frequency')
    plt.ylabel('Score Ranges')
    plt.yticks(x, bin_labels)
    plt.title('Histogram of Political Bias Ratings')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
create_horizontal_bar_chart('results/human_review_results.csv', 'results/ai_review_results.csv', 'Q62')


def create_horizontal_bar_chart(csv_file1, csv_file2, column_name):
    # Load the data
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
    # Check if the column exists in both dataframes
    if column_name not in df1.columns or column_name not in df2.columns:
        raise ValueError(f"Column {column_name} not found in one of the files")
    
    # Get the column data
    column1 = df1[column_name]
    column2 = df2[column_name]
    
    # Define bins and calculate frequencies
    bins = [-0.5, 0.5, 1.5]
    human_counts, _ = np.histogram(column1, bins=bins)
    ai_counts, _ = np.histogram(column2, bins=bins)
    
    # Define bar positions
    bin_labels = ['Believe to be AI', 'Believed to be Human']
    x = np.arange(len(bin_labels))
    
    # Create a horizontal bar chart
    bar_width = 0.4  # Width of the bars
    plt.barh(x - bar_width/2, human_counts, height=bar_width, label='Human', color='blue', alpha=0.7)
    plt.barh(x + bar_width/2, ai_counts, height=bar_width, label='AI', color='orange', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Frequency')
    plt.ylabel('Score Ranges')
    plt.yticks(x, bin_labels)
    plt.title('Histogram of Political Bias Ratings')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
create_horizontal_bar_chart('results/human_review_results.csv', 'results/ai_review_results.csv', 'Q64')

def create_horizontal_bar_chart(csv_file1, csv_file2, column_name):
    # Load the data
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
    # Check if the column exists in both dataframes
    if column_name not in df1.columns or column_name not in df2.columns:
        raise ValueError(f"Column {column_name} not found in one of the files")
    
    # Get the column data
    column1 = df1[column_name]
    column2 = df2[column_name]
    
    # Define bins and calculate frequencies
    bins = [-2, -1.5, -0.5, 0.5, 1.5, 2]
    human_counts, _ = np.histogram(column1, bins=bins)
    ai_counts, _ = np.histogram(column2, bins=bins)
    
    # Define bar positions
    bin_labels = ['Very Unconfident','Unconfident','Neutral','Confident','Very Confident']
    x = np.arange(len(bin_labels))
    
    # Create a horizontal bar chart
    bar_width = 0.4  # Width of the bars
    plt.barh(x - bar_width/2, human_counts, height=bar_width, label='Human', color='blue', alpha=0.7)
    plt.barh(x + bar_width/2, ai_counts, height=bar_width, label='AI', color='orange', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Frequency')
    plt.ylabel('Score Ranges')
    plt.yticks(x, bin_labels)
    plt.title('Histogram of Political Bias Ratings')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
create_horizontal_bar_chart('results/human_review_results.csv', 'results/ai_review_results.csv', 'Q66')


def create_scatterplot(csv_file, column_name, group_label):
    # Load the data
    df = pd.read_csv(csv_file)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in {csv_file}")
    
    # Generate scatterplot
    x_values = range(len(df[column_name]))
    y_values = df[column_name]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, alpha=0.7, label=group_label)
    plt.title(f"Scatterplot of {column_name}")
    plt.xlabel("Index")
    plt.ylabel(column_name)
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
# Replace 'Q62', 'Q64', and 'Q66' with your column names
questions = ['Q62', 'Q64', 'Q66']
csv_files = [
    ('results/human_review_results.csv', 'Human'),
    ('results/ai_review_results.csv', 'AI')
]

# Generate scatterplots for each question and group
for csv_file, group_label in csv_files:
    for question in questions:
        create_scatterplot(csv_file, question, group_label)
def create_stacked_boxplots(human_file, ai_file, questions):
    """
    Creates stacked boxplots for human and AI responses for each question.

    Args:
        human_file (str): Path to the human responses CSV file.
        ai_file (str): Path to the AI responses CSV file.
        questions (list): List of question column names.

    Returns:
        None: Displays the stacked boxplots.
    """
    # Load the data
    human_df = pd.read_csv(human_file)
    ai_df = pd.read_csv(ai_file)

    # Create a combined DataFrame for easier plotting
    data = []
    for question in questions:
        if question not in human_df.columns or question not in ai_df.columns:
            raise ValueError(f"Column {question} not found in one of the files.")
        
        for value in human_df[question].dropna():
            data.append({'Question': question, 'Group': 'Human', 'Value': value})
        for value in ai_df[question].dropna():
            data.append({'Question': question, 'Group': 'AI', 'Value': value})
    
    combined_df = pd.DataFrame(data)

    # Create the boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=combined_df,
        x='Question',
        y='Value',
        hue='Group',
        palette={'Human': 'blue', 'AI': 'orange'},
        showmeans=True,
        meanline=True
    )

    # Add labels and title
    plt.title('Boxplots for Human and AI Responses Across Questions')
    plt.xlabel('Question')
    plt.ylabel('Scores')
    plt.legend(title='Group')
    plt.tight_layout()
    plt.show()

# Example usage
questions = ['Q62']
create_stacked_boxplots('results/human_review_results.csv', 'results/ai_review_results.csv', questions)

questions = ['Q64']
create_stacked_boxplots('results/human_review_results.csv', 'results/ai_review_results.csv', questions)

questions = ['Q66']
create_stacked_boxplots('results/human_review_results.csv', 'results/ai_review_results.csv', questions)

#
#create scatterplot with trendline and rsquared of accuracy vs Q66 for human and ai
def plot_confidence_vs_accuracy(human_file, ai_file):
    # Load the data
    human_df = pd.read_csv(human_file)
    ai_df = pd.read_csv(ai_file)
    
    # Calculate accuracy based on Q66
    human_df['accuracy'] = human_df['Q66'].apply(lambda x: 1 if x >= 0.5 else 0)
    ai_df['accuracy'] = ai_df['Q66'].apply(lambda x: 1 if x <= 0.5 else 0)
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    
    # Scatter plot for Human responses
    plt.scatter(human_df['Q66'], human_df['accuracy'], label='Human Responses', color='blue', alpha=0.7)
    
    # Scatter plot for AI responses
    plt.scatter(ai_df['Q66'], ai_df['accuracy'], label='AI Responses', color='orange', alpha=0.7)
    
    # Add trend lines
    human_slope, human_intercept = np.polyfit(human_df['Q66'], human_df['accuracy'], 1)
    ai_slope, ai_intercept = np.polyfit(ai_df['Q66'], ai_df['accuracy'], 1)
    
    plt.plot(human_df['Q66'], human_slope * human_df['Q66'] +
                human_intercept, color='blue', linestyle='dashed', label='Human Trend Line')
    plt.plot(ai_df['Q66'], ai_slope * ai_df['Q66'] +
                ai_intercept, color='orange', linestyle='--', label='AI Trend Line')
    #reduce lineweight
    plt.setp(plt.gca().get_lines(), lw=1)
    #make trendlines more sparsely dashed
    #make trendlines more sparsely dashed
    # Add labels and title
    plt.title('Confidence vs. Accuracy for AI and Human Responses')
    plt.xlabel('Confidence Level')
    #only have labels for 0 and 1 on the y axis
    plt.yticks([0,1],['Incorrect','Correct'])
    plt.ylabel('Accuracy (0 = Incorrect, 1 = Correct)')
    plt.legend()
    plt.tight_layout()

    plt.show()
    #calculate and return average correctness for each group
    human_correct = human_df['accuracy'].mean()
    ai_correct = ai_df['accuracy'].mean()
    #also return rsquared values for correlation for each
    human_residuals = human_df['accuracy'] - (human_slope * human_df['Q66'] + human_intercept)
    human_residuals_squared = human_residuals ** 2
    human_ss_res = human_residuals_squared.sum()
    human_ss_tot = ((human_df['accuracy'] - human_df['accuracy'].mean()) ** 2).sum()
    human_r_squared = 1 - (human_ss_res / human_ss_tot)
    #do the same for ai
    ai_residuals = ai_df['accuracy'] - (ai_slope * ai_df['Q66'] + ai_intercept)
    ai_residuals_squared = ai_residuals ** 2
    ai_ss_res = ai_residuals_squared.sum()
    ai_ss_tot = ((ai_df['accuracy'] - ai_df['accuracy'].mean()) ** 2).sum()
    ai_r_squared = 1 - (ai_ss_res / ai_ss_tot)
    return human_correct, ai_correct, human_r_squared, ai_r_squared


human_correct,ai_correct,human_r_squared,ai_r_squared=plot_confidence_vs_accuracy('results/human_review_results.csv', 'results/ai_review_results.csv')
print(f"Human Correctness: {human_correct:.2f}")
print(f"AI Correctness: {ai_correct:.2f}")
print(f"Human R^2: {human_r_squared:.2f}")
print(f"AI R^2: {ai_r_squared:.2f}")






