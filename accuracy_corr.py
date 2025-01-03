import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define mappings
confidence_mapping = {'Very unconfident': -2, 'Somewhat unconfident': -1, 'Neutral': 0, 'Somewhat confident': 1, 'Very confident': 2}
ai_human_mapping = {'AI-generated': 0, 'Human-generated': 1}

def preprocess_data(file_path):
    """
    Preprocess the CSV file by applying mappings and deriving the true label from the file name.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Processed dataframe with true labels and response type.
    """
    df = pd.read_csv(file_path)

    # Derive the true label from the file name
    true_label = file_path.split('/')[-1].split('_')[0].capitalize() + "-generated"
    true_label = true_label.replace("Ai-generated", "AI-generated")  # Correct case if mismatched

    # Print raw values for debugging
    print("Raw Q64 values:", df['Q64'].unique())
    print("Raw Q66 values:", df['Q66'].unique())

    # Normalize case and trim whitespace
    df['Q64'] = df['Q64'].str.strip().str.title()
    df['Q66'] = df['Q66'].str.strip().str.title()

    # Debug normalized values
    print("Normalized Q64 values:", df['Q64'].unique())
    print("Normalized Q66 values:", df['Q66'].unique())

    # Apply mappings
    df['Q64_mapped'] = df['Q64'].map(ai_human_mapping)
    df['Q66_mapped'] = df['Q66'].map(confidence_mapping)

    # Debug mapped values
    print(df[['Q64', 'Q64_mapped', 'Q66', 'Q66_mapped']].head())

    # Add true label and response type columns
    df['true_label'] = ai_human_mapping[true_label]  # Map true label to numeric value
    df['response_type'] = true_label.split('-')[0]  # AI or Human label for plotting
    return df


def plot_confidence_vs_accuracy(ai_file, human_file):
    """
    Plot scatter plot of confidence vs. accuracy with trend lines for AI and Human responses.

    Args:
        ai_file (str): Path to the AI responses CSV file.
        human_file (str): Path to the Human responses CSV file.

    Returns:
        None: Displays the plot.
    """
    # Preprocess data
    ai_df = preprocess_data(ai_file)
    human_df = preprocess_data(human_file)

    # Combine the dataframes
    combined_df = pd.concat([ai_df, human_df], ignore_index=True)

    # Create accuracy column (1 if response type matches true label, 0 otherwise)
    combined_df['accuracy'] = (combined_df['Q64_mapped'] == combined_df['true_label']).astype(int)

    # Debugging: Check processed data
    print(combined_df[['Q64_mapped', 'Q66_mapped', 'true_label', 'accuracy']].head())

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Scatter plot for AI responses
    sns.scatterplot(
        data=combined_df[combined_df['response_type'] == 'AI'],
        x='Q66_mapped',
        y='accuracy',
        label='AI Responses',
        color='orange',
        alpha=0.7,
    )

    # Scatter plot for Human responses
    sns.scatterplot(
        data=combined_df[combined_df['response_type'] == 'Human'],
        x='Q66_mapped',
        y='accuracy',
        label='Human Responses',
        color='blue',
        alpha=0.7,
    )

    # Add trend lines
    sns.regplot(
        data=combined_df[combined_df['response_type'] == 'AI'],
        x='Q66_mapped',
        y='accuracy',
        scatter=False,
        label='AI Trend Line',
        color='orange',
        line_kws={"linestyle": "dashed"},
    )
    sns.regplot(
        data=combined_df[combined_df['response_type'] == 'Human'],
        x='Q66_mapped',
        y='accuracy',
        scatter=False,
        label='Human Trend Line',
        color='blue',
        line_kws={"linestyle": "dashed"},
    )

    # Customize plot
    plt.title('Confidence vs. Accuracy for AI and Human Responses')
    plt.xlabel('Confidence Level')
    plt.ylabel('Accuracy (0 = Incorrect, 1 = Correct)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
plot_confidence_vs_accuracy(
    ai_file='results/ai_review_results.csv',
    human_file='results/human_review_results.csv'
)
