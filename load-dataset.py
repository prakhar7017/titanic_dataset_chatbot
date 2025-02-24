import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def load_titanic_data():
    df = pd.read_csv(DATA_URL)
    return df

if __name__ == "__main__":
    df = load_titanic_data()
    print("First 5 rows of the dataset:")
    print(df.head())  # Print the first 5 rows to verify data
