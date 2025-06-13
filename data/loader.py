import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(data_paths, data_params):

    # Jeżeli wszystkie pliki istnieją, wczytaj i zwróć je
    processed_df_path = data_paths["preprocessors_paths"]["processed_dataframe"]
    cache_is_valid = False
    if os.path.exists(processed_df_path):
        try:
            # Check if cached columns match the expected columns from config
            cached_cols = pd.read_csv(processed_df_path, nrows=0).columns
            expected_cols = data_params["columns_to_load"].values()
            if set(expected_cols).issubset(set(cached_cols)):
                cache_is_valid = True
        except Exception:
            pass # If cache is invalid for any reason, proceed to reprocess

    if cache_is_valid and all(os.path.exists(p) for p in data_paths["preprocessors_paths"].values() if p):
        dataFrame = pd.read_csv(processed_df_path, index_col='ID')
        if 'ID' not in dataFrame.columns:
            dataFrame['ID'] = dataFrame.index
        X = joblib.load(data_paths["preprocessors_paths"]["processed_X"])
        scaler = joblib.load(data_paths["preprocessors_paths"]["processed_scaler"])
        return dataFrame, X, scaler

    # W przeciwnym razie – przetwórz dane od zera
    # Wczytaj dane tylko z wybranych kolumn
    columns_dict = data_params["columns_to_load"]
    selected_cols = list(columns_dict.values())

    dataFrame = pd.read_csv(
        data_paths['database_csv'],
        usecols=selected_cols,
        nrows=data_params.get("nrows_to_load", None)  # może być int lub None
    )

    # Zakoduj kolumnę Brand
    # Dane zamienia się z tekstowych na numeryczne, żeby model mógł je zrozumieć
    brand_encoder = LabelEncoder()
    encoded_brand_col = f'{data_params["columns_to_load"]["brand_col"]}_encoded'
    dataFrame[encoded_brand_col] = brand_encoder.fit_transform(dataFrame[data_params["columns_to_load"]["brand_col"]])

    # Zakoduj kolumnę Category
    # Dane zamienia się z tekstowych na numeryczne, żeby model mógł je zrozumieć
    category_encoder = LabelEncoder()
    encoded_category_col = f'{data_params["columns_to_load"]["category_col"]}_encoded'
    dataFrame[encoded_category_col] = category_encoder.fit_transform(dataFrame[data_params["columns_to_load"]["category_col"]])

    # Skaluj kolumny Price i Rating
    scaler = StandardScaler()
    scaled_cols = [f'{data_params["columns_to_load"]["price_col"]}_scaled', f'{data_params["columns_to_load"]["rating_col"]}_scaled']
    dataFrame[scaled_cols] = scaler.fit_transform(
        dataFrame[[data_params["columns_to_load"]["price_col"], data_params["columns_to_load"]["rating_col"]]]
    )

    # Zbuduj macierz X
    X = dataFrame[[encoded_brand_col, encoded_category_col] + scaled_cols].values

    dataFrame = dataFrame.set_index('ID')

    # Zapisz przetworzone dane
    dataFrame.to_csv(data_paths["preprocessors_paths"]["processed_dataframe"], index=True)
    joblib.dump(X, data_paths["preprocessors_paths"]["processed_X"])
    joblib.dump(scaler, data_paths["preprocessors_paths"]["processed_scaler"])

    # dataFrame to bazadanych po dodaniu i odjęciu dodatkowych kolumn
    # X to macież przetworzonych danych które są uśrednione około 1 [1, 0.5, 0,25, 1] [-1, 05, 0.25, -1]
    # scaler to pamięć tego jak są przeskalowane dane odchylenia, używa się go do nowych danych żeby były przeskalowane na np. liczby z przedziału 0-1 jak powyżej
    if 'ID' not in dataFrame.columns:
        dataFrame['ID'] = dataFrame.index
        
    return dataFrame, X, scaler
