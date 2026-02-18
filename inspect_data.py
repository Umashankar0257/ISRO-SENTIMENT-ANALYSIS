import pandas as pd

try:
    df = pd.read_csv('chandrayaan3_50k_realistic_global_opinion_dataset.csv')
    print("Columns:", df.columns.tolist())
    print("Unique Phases:", df['phase'].unique())
    print("Date Range:", df['date'].min(), "to", df['date'].max())
    print("Sentiment distribution:\n", df['sentiment'].value_counts())
    print("\nPhase Date Ranges:")
    print(df.groupby('phase')['date'].agg(['min', 'max', 'count']))
except Exception as e:
    print(e)
