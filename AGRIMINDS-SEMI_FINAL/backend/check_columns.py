import pandas as pd

print("Checking Crop_recommendation.csv columns:")
try:
    df_crop = pd.read_csv('Crop_recommendation.csv')
    print(f"Columns: {list(df_crop.columns)}")
    print(f"Shape: {df_crop.shape}")
    print(f"\nFirst few rows:")
    print(df_crop.head())
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*70 + "\n")

print("Checking Smart_Farming_Crop_Yield_2024.csv columns:")
try:
    df_yield = pd.read_csv('Smart_Farming_Crop_Yield_2024.csv')
    print(f"Columns: {list(df_yield.columns)}")
    print(f"Shape: {df_yield.shape}")
    print(f"\nFirst few rows:")
    print(df_yield.head())
except Exception as e:
    print(f"Error: {e}")
