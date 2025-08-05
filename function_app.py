#Will later change for a storage blob 

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# Define the path to the CSV
file_path = '/content/drive/My Drive/sportsanalytics/nhldraft.csv'

# Load the CSV into a DataFrame
hockey_df = pd.read_csv(file_path)

# Display the first few rows
hockey_df.head()
