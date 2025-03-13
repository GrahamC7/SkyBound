import pandas as pd

def preprocess_data():
    """
    Loads, cleans, and filters the space missions dataset to include only
    launches from Cape Canaveral. Converts launch status to 'Go' or 'No-Go'.
    Saves the processed dataset as 'processed_launch_data.csv'.
    """

    print("📂 Loading dataset...")

    # Load the dataset
    file_path = "../data/Space_Corrected.csv"
    df = pd.read_csv(file_path)

    # Display available columns
    print("📊 Columns in dataset:", df.columns)

    # Drop unnecessary columns
    df = df[['Datum', 'Location', 'Status Mission']]

    # Filter only launches from Cape Canaveral
    df = df[df['Location'].str.contains("Cape Canaveral", case=False, na=False)]

    # Convert 'Status Mission' to binary (1 = Go, 0 = No-Go)
    df['launch_status'] = df['Status Mission'].apply(lambda x: 1 if "Success" in x else 0)

    # Convert 'Datum' column to standard date format
    df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')

    # Remove rows where date conversion failed
    df = df.dropna(subset=['Datum'])

    # Rename columns for clarity
    df.rename(columns={'Datum': 'date', 'Location': 'launch_location'}, inplace=True)

    # Save the processed dataset
    processed_file_path = "../data/processed_launch_data.csv"
    df.to_csv(processed_file_path, index=False)

    print(f"✅ Data preprocessed and saved as '{processed_file_path}'")
    print(df.head())  # Display first few rows

if __name__ == "__main__":
    preprocess_data()

