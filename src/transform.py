import pandas as pd
import numpy as np
import os

def clean_dataframe(df):
    """
    Clean a DataFrame by processing name and email fields.
    
    Args:
        df: pandas DataFrame with 'name' and 'email' columns
        
    Returns:
        Cleaned pandas DataFrame
    """
    # Create a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Trim and lowercase name
    cleaned_df['name'] = cleaned_df['name'].str.strip().str.lower()
    
    # Trim and lowercase email
    cleaned_df['email'] = cleaned_df['email'].str.strip().str.lower()
    
    # Remove rows where email is null or empty string
    cleaned_df = cleaned_df[cleaned_df['email'].notna()]
    cleaned_df = cleaned_df[cleaned_df['email'] != '']
    
    # Remove duplicates based on email
    cleaned_df = cleaned_df.drop_duplicates(subset=['email'], keep='first')
    
    # Reset index
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df


# File paths
input_file = r'C:\Users\Navneet\Desktop\Project\customer-data-ci\raw_data\messy_sample_data.xlsx'
output_file = r'C:\Users\Navneet\Desktop\Project\customer-data-ci\raw_data\cleaned_sample_data.xlsx'

# Read the messy Excel file
print("Reading messy data from:")
print(f"  {input_file}")
print()

messy_df = pd.read_excel(input_file)

print("="*60)
print("ORIGINAL MESSY DATA:")
print("="*60)
print(messy_df)
print(f"\nTotal rows: {len(messy_df)}")
print(f"Null emails: {messy_df['email'].isna().sum()}")

# Clean the data
print("\n" + "="*60)
print("CLEANING DATA...")
print("="*60)
cleaned_df = clean_dataframe(messy_df)

print("\n" + "="*60)
print("CLEANED DATA:")
print("="*60)
print(cleaned_df)
print(f"\nTotal rows: {len(cleaned_df)}")
print(f"\nRows removed: {len(messy_df) - len(cleaned_df)}")

# Show what was removed
print("\n" + "="*60)
print("SUMMARY OF CHANGES:")
print("="*60)
print(f"• Removed {messy_df['email'].isna().sum()} rows with null emails")
print(f"• Removed {(messy_df['email'] == '').sum()} rows with empty emails")

# Find duplicates in original data
original_emails_cleaned = messy_df['email'].str.strip().str.lower()
duplicates_count = original_emails_cleaned.duplicated().sum()
print(f"• Removed {duplicates_count} duplicate emails")
print(f"• All names and emails trimmed and lowercased")

# Save cleaned data to new Excel file
cleaned_df.to_excel(output_file, index=False, sheet_name='Cleaned Data')
print(f"\n✓ Cleaned data saved to:")
print(f"  {output_file}")