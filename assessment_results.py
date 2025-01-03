import pandas as pd
import os

# Read the dataset (assuming it's a CSV file)
def calc_stats(csv_file,q1,q2,q3):
    file_path = csv_file  # Replace with your file path
    df = pd.read_csv(file_path, header=0)  # Skip the first two rows and make the third row the header

    # Remove the second and third row
    df = df.drop([0, 1])

    # Map text values to numeric scores for the questions
    rating_mapping = {'Very liberal': -2, 'Liberal': -1, 'Neutral': 0, 'Conservative': 1, 'Very conservative': 2}
    ai_human_mapping = {'AI-generated': 0, 'Human-generated': 1}
    confidence_mapping = {'Very unconfident': -2, 'Somewhat unconfident': -1, 'Neutral': 0, 'Somewhat confident': 1, 'Very confident': 2}

    # Apply the mappings
    df['Q62'] = df['Q62'].map(rating_mapping)
    df['Q64'] = df['Q64'].map(ai_human_mapping)
    df['Q66'] = df['Q66'].map(confidence_mapping)

    # Select only numeric columns for the groupby operation
    numeric_columns = ['Q62', 'Q64', 'Q66']

    # Calculate the average score for each question for each rating block and store in a df
    average_scores = df.groupby('RatingBlock_DO')[numeric_columns].mean()

    base_name = os.path.splitext(csv_file)[0]
    results_file = f"{base_name}_results.csv"
    
    # Ensure the results directory exists
    os.makedirs('./results', exist_ok=True)
    
    # Save the results DataFrame to a new CSV file in the results directory
    average_scores.to_csv(f"./results/{results_file}", index=False)

    return average_scores

ai_results = calc_stats('ai_review.csv','Q62','Q64','Q66')
human_results = calc_stats('human_review.csv','Q62','Q64','Q66')


print(ai_results)
print(human_results)
