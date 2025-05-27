import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(db_name="data/Book3.csv", test_size=0.2):
    # Load the dataset from a CSV file
    dataFrame = pd.read_csv(db_name)

    # Create a lightweight version of the DataFrame for display purposes
    df_display = dataFrame[['Product_Name', 'Brand', 'Category', 'Price_USD', 'Rating']].copy()

    # Save category labels separately for future use (e.g., during recommendations)
    index_to_category = df_display['Category'].values

    # Convert 'Product_Size' from string like "100ml" to integer
    dataFrame['Size_ml'] = dataFrame['Product_Size'].str.replace('ml', '').astype(int)

    # Drop the original 'Product_Size' column
    dataFrame.drop(columns=['Product_Size'], inplace=True)

    # Map usage frequency from categorical to numeric values
    freq_map = {'Occasional': 1, 'Monthly': 2, 'Weekly': 3, 'Daily': 4}
    dataFrame['Usage_Num'] = dataFrame['Usage_Frequency'].map(freq_map)

    # Drop the original 'Usage_Frequency' column
    dataFrame.drop(columns=['Usage_Frequency'], inplace=True)

    # List of categorical features to be one-hot encoded
    categories = [
        'Brand',
        'Category',
        'Skin_Type',
        'Gender_Target',
        'Packaging_Type',
        'Main_Ingredient',
        'Country_of_Origin'
    ]

    # Apply one-hot encoding to categorical columns, drop first level to avoid multicollinearity
    dataFrame = pd.get_dummies(dataFrame, columns=categories, drop_first=True)

    # Ensure 'Cruelty_Free' is of integer type (0 or 1)
    dataFrame['Cruelty_Free'] = dataFrame['Cruelty_Free'].astype(int)

    # Drop unused columns and extract the feature matrix X (without 'Product_Name' and 'Rating')
    X = dataFrame.drop(columns=['Product_Name', 'Rating']).values

    # Normalize all features using standard scaling (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Return both versions of the data: for display and for modeling
    return df_display, dataFrame, index_to_category, X_scaled, scaler

