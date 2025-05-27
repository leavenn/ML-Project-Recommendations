import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(db_name="data/TestBase.csv", test_size=0.2):
    dataFrame = pd.read_csv(db_name)
    df_display = dataFrame[['Product_Name', 'Brand', 'Category', 'Price_USD', 'Rating']].copy()
    index_to_category = df_display['Category'].values

    dataFrame['Size_ml'] = dataFrame['Product_Size'].str.replace('ml', '').astype(int)
    dataFrame.drop(columns=['Product_Size'], inplace=True)
    freq_map = {'Occasional':1, 'Monthly':2, 'Weekly':3, 'Daily':4}
    dataFrame['Usage_Num'] = dataFrame['Usage_Frequency'].map(freq_map)
    dataFrame.drop(columns=['Usage_Frequency'], inplace=True)

    categories = ['Brand','Category','Skin_Type','Gender_Target','Packaging_Type','Main_Ingredient','Country_of_Origin']
    dataFrame = pd.get_dummies(dataFrame, columns=categories, drop_first=True)
    dataFrame['Cruelty_Free'] = dataFrame['Cruelty_Free'].astype(int)

    X = dataFrame.drop(columns=['Product_Name', 'Rating']).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df_display, dataFrame, index_to_category, X_scaled, scaler
