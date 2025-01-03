import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
from numpy import mean, std

# Define colors
colors = {'Human': 'cornflowerblue', 'AI': 'indianred'}

# Function: Perform t-tests and F-tests
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

# Example usage for statistical tests
questions = ['Q62', 'Q64', 'Q66']
for question in questions:
    t_stat, p_val, f_stat, p_val_var = perform_ttest(
        'results/human_review_results.csv', 
        'results/ai_review_results.csv', 
        question
    )
    print(f"{question}: T-test p-value = {p_val}, F-test p-value = {p_val_var}")

# Function: Create horizontal bar charts
def create_histogram(csv_file1, csv_file2, column_name, bin_labels, bins, title):
    """
    Creates a horizontal histogram comparing two datasets.

    Args:
        csv_file1 (str): Path to the first CSV file (Human responses).
        csv_file2 (str): Path to the second CSV file (AI responses).
        column_name (str): Column name to generate the histogram for.
        bin_labels (list): Labels for the bins.
        bins (list): Bin edges.
        title (str): Title for the histogram.

    Returns:
        None: Displays the histogram.
    """
    # Load the data
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)

    # Check if the column exists in both dataframes
    if column_name not in df1.columns or column_name not in df2.columns:
        raise ValueError(f"Column {column_name} not found in one of the files.")

    # Extract the column data
    column1 = df1[column_name]
    column2 = df2[column_name]

    # Calculate frequencies
    human_counts, _ = np.histogram(column1, bins=bins)
    ai_counts, _ = np.histogram(column2, bins=bins)

    # Define positions for the bars
    x = np.arange(len(bin_labels))

    # Plot horizontal bar chart
    bar_width = 0.4  # Width of the bars
    plt.barh(x + bar_width / 2, ai_counts, height=bar_width, label='AI', color='indianred', alpha=0.7, edgecolor='black')
    plt.barh(x - bar_width / 2, human_counts, height=bar_width, label='Human', color='cornflowerblue', alpha=0.7, edgecolor='black')
    

    # Add labels and title
    plt.xlabel('Frequency')
    plt.ylabel('Categories')
    plt.yticks(x, bin_labels)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Histogram for Q62
create_histogram(
    csv_file1='results/human_review_results.csv',
    csv_file2='results/ai_review_results.csv',
    column_name='Q62',
    bin_labels=['Very Liberal', 'Liberal', 'Neutral', 'Conservative', 'Very Conservative'],
    bins=[-2, -1.5, -0.5, 0.5, 1.5, 2],
    title='Histogram for Q62: Political Bias Ratings'
)

# Histogram for Q64
create_histogram(
    csv_file1='results/human_review_results.csv',
    csv_file2='results/ai_review_results.csv',
    column_name='Q64',
    bin_labels=['AI Generated', 'Human Generated'],
    bins=[-0.5, 0.5, 1.5],
    title='Histogram for Q64: Belief About Origin'
)

# Histogram for Q66
create_histogram(
    csv_file1='results/human_review_results.csv',
    csv_file2='results/ai_review_results.csv',
    column_name='Q66',
    bin_labels=['Very Unconfident', 'Unconfident', 'Neutral', 'Confident', 'Very Confident'],
    bins=[-2, -1.5, -0.5, 0.5, 1.5, 2],
    title='Histogram for Q66: Confidence Levels'
)

def create_separate_boxplots(human_file, ai_file, questions):
    """
    Creates separate boxplot graphs for human and AI responses for each question.

    Args:
        human_file (str): Path to the human responses CSV file.
        ai_file (str): Path to the AI responses CSV file.
        questions (list): List of question column names.

    Returns:
        None: Displays the boxplots.
    """
    # Load the data
    human_df = pd.read_csv(human_file)
    ai_df = pd.read_csv(ai_file)

    for question in questions:
        # Check if question exists in both files
        if question not in human_df.columns or question not in ai_df.columns:
            raise ValueError(f"Column {question} not found in one of the files.")

        # Combine data into a single DataFrame
        data = []
        for value in human_df[question].dropna():
            data.append({'Group': 'Human', 'Value': value})
        for value in ai_df[question].dropna():
            data.append({'Group': 'AI', 'Value': value})

        combined_df = pd.DataFrame(data)

        # Create the boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            data=combined_df,
            x='Group',
            y='Value',
            palette={'Human': 'cornflowerblue', 'AI': 'indianred'},
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 7
            },
            width=0.6
        )

        # Add title and labels
        plt.title(f'Boxplot for {question}')
        plt.xlabel('Group')
        plt.ylabel('Scores')

        # Customize spacing between box plots
        plt.gca().set_xticks([0, 1])
        plt.gca().set_xticklabels(['Human', 'AI'], ha='center', fontsize=12)
        plt.tight_layout()

        # Show the plot
        plt.show()

# Example usage
questions = ['Q62', 'Q64', 'Q66']
create_separate_boxplots('results/human_review_results.csv', 'results/ai_review_results.csv', questions)

# Function: Scatterplot with trendlines
def plot_confidence_vs_accuracy(human_file, ai_file):
    human_df = pd.read_csv(human_file)
    ai_df = pd.read_csv(ai_file)

    # Calculate accuracy
    human_df['accuracy'] = human_df['Q66'].apply(lambda x: 1 if x >= 0.5 else 0)
    ai_df['accuracy'] = ai_df['Q66'].apply(lambda x: 1 if x <= 0.5 else 0)

    plt.figure(figsize=(10, 6))

    sns.scatterplot(x='Q66', y='accuracy', data=human_df, label='Human', color=colors['Human'], alpha=0.7)
    sns.scatterplot(x='Q66', y='accuracy', data=ai_df, label='AI', color=colors['AI'], alpha=0.7)

    sns.regplot(x='Q66', y='accuracy', data=human_df, scatter=False, color=colors['Human'], label='Human Trend Line', line_kws={"linestyle": "dashed"})
    sns.regplot(x='Q66', y='accuracy', data=ai_df, scatter=False, color=colors['AI'], label='AI Trend Line', line_kws={"linestyle": "dashed"})

    plt.xlabel('Confidence Level')
    plt.ylabel('Accuracy (0 = Incorrect, 1 = Correct)')
    plt.yticks([0, 1], ['Incorrect', 'Correct'])
    plt.title('Confidence vs. Accuracy for AI and Human Responses')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage for scatterplot
plot_confidence_vs_accuracy(
    'results/human_review_results.csv', 
    'results/ai_review_results.csv'
)
