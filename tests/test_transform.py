import pandas as pd
import numpy as np
import unittest

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
    
    # Handle empty DataFrame
    if len(cleaned_df) == 0:
        return cleaned_df
    
    # Trim and lowercase name (only if there are non-null values)
    if cleaned_df['name'].notna().any():
        cleaned_df['name'] = cleaned_df['name'].str.strip().str.lower()
    
    # Trim and lowercase email (only if there are non-null values)
    if cleaned_df['email'].notna().any():
        cleaned_df['email'] = cleaned_df['email'].str.strip().str.lower()
    
    # Remove rows where email is null or empty string
    cleaned_df = cleaned_df[cleaned_df['email'].notna()]
    cleaned_df = cleaned_df[cleaned_df['email'] != '']
    
    # Remove duplicates based on email
    cleaned_df = cleaned_df.drop_duplicates(subset=['email'], keep='first')
    
    # Reset index
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df


class TestCleanDataFrame(unittest.TestCase):
    """Test cases for the clean_dataframe function"""
    
    def test_trim_name(self):
        """Test that names with leading/trailing spaces are trimmed"""
        df = pd.DataFrame({
            'name': ['  John Doe  ', '  Jane Smith  '],
            'email': ['john@example.com', 'jane@example.com']
        })
        
        result = clean_dataframe(df)
        
        self.assertEqual(result['name'].iloc[0], 'john doe')
        self.assertEqual(result['name'].iloc[1], 'jane smith')
    
    def test_lowercase_name(self):
        """Test that names are converted to lowercase"""
        df = pd.DataFrame({
            'name': ['JOHN DOE', 'Jane Smith', 'bob JOHNSON'],
            'email': ['john@example.com', 'jane@example.com', 'bob@example.com']
        })
        
        result = clean_dataframe(df)
        
        self.assertEqual(result['name'].iloc[0], 'john doe')
        self.assertEqual(result['name'].iloc[1], 'jane smith')
        self.assertEqual(result['name'].iloc[2], 'bob johnson')
    
    def test_trim_email(self):
        """Test that emails with leading/trailing spaces are trimmed"""
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith'],
            'email': ['  john@example.com  ', 'jane@example.com  ']
        })
        
        result = clean_dataframe(df)
        
        self.assertEqual(result['email'].iloc[0], 'john@example.com')
        self.assertEqual(result['email'].iloc[1], 'jane@example.com')
    
    def test_lowercase_email(self):
        """Test that emails are converted to lowercase"""
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith'],
            'email': ['JOHN@EXAMPLE.COM', 'Jane@Example.COM']
        })
        
        result = clean_dataframe(df)
        
        self.assertEqual(result['email'].iloc[0], 'john@example.com')
        self.assertEqual(result['email'].iloc[1], 'jane@example.com')
    
    def test_remove_null_emails(self):
        """Test that rows with null emails are removed"""
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'email': ['john@example.com', None, 'bob@example.com']
        })
        
        result = clean_dataframe(df)
        
        self.assertEqual(len(result), 2)
        self.assertNotIn('Jane Smith', result['name'].values)
    
    def test_remove_nan_emails(self):
        """Test that rows with NaN emails are removed"""
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'email': ['john@example.com', np.nan, 'bob@example.com']
        })
        
        result = clean_dataframe(df)
        
        self.assertEqual(len(result), 2)
        self.assertIn('john doe', result['name'].values)
        self.assertIn('bob johnson', result['name'].values)
    
    def test_remove_empty_string_emails(self):
        """Test that rows with empty string emails are removed"""
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'email': ['john@example.com', '', 'bob@example.com']
        })
        
        result = clean_dataframe(df)
        
        self.assertEqual(len(result), 2)
        self.assertNotIn('jane smith', result['name'].values)
    
    def test_remove_duplicates_keep_first(self):
        """Test that duplicate emails are removed, keeping the first occurrence"""
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'John Duplicate'],
            'email': ['john@example.com', 'jane@example.com', 'john@example.com']
        })
        
        result = clean_dataframe(df)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result['name'].iloc[0], 'john doe')
        self.assertNotIn('john duplicate', result['name'].values)
    
    def test_remove_duplicates_case_insensitive(self):
        """Test that duplicate emails are detected case-insensitively"""
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'John Duplicate'],
            'email': ['john@example.com', 'jane@example.com', 'JOHN@EXAMPLE.COM']
        })
        
        result = clean_dataframe(df)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result['email'].iloc[0], 'john@example.com')
    
    def test_remove_duplicates_with_spaces(self):
        """Test that duplicates are detected even with different spacing"""
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'John Duplicate'],
            'email': ['john@example.com', 'jane@example.com', '  JOHN@EXAMPLE.COM  ']
        })
        
        result = clean_dataframe(df)
        
        self.assertEqual(len(result), 2)
    
    def test_reset_index(self):
        """Test that the index is reset after cleaning"""
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Williams'],
            'email': ['john@example.com', None, 'john@example.com', 'alice@example.com']
        })
        
        result = clean_dataframe(df)
        
        # Check that index starts at 0 and is sequential
        self.assertEqual(result.index.tolist(), list(range(len(result))))
    
    def test_original_dataframe_unchanged(self):
        """Test that the original DataFrame is not modified"""
        df = pd.DataFrame({
            'name': ['  JOHN DOE  '],
            'email': ['  JOHN@EXAMPLE.COM  ']
        })
        
        original_name = df['name'].iloc[0]
        original_email = df['email'].iloc[0]
        
        result = clean_dataframe(df)
        
        # Original should be unchanged
        self.assertEqual(df['name'].iloc[0], original_name)
        self.assertEqual(df['email'].iloc[0], original_email)
        
        # Result should be cleaned
        self.assertEqual(result['name'].iloc[0], 'john doe')
        self.assertEqual(result['email'].iloc[0], 'john@example.com')
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame({
            'name': [],
            'email': []
        })
        
        result = clean_dataframe(df)
        
        self.assertEqual(len(result), 0)
    
    def test_all_nulls(self):
        """Test DataFrame with all null emails"""
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith'],
            'email': [None, np.nan]
        })
        
        result = clean_dataframe(df)
        
        self.assertEqual(len(result), 0)
    
    def test_complex_scenario(self):
        """Test a complex scenario with multiple issues"""
        df = pd.DataFrame({
            'name': [
                '  John Doe  ',      # Spaces
                'JANE SMITH',        # Uppercase
                '  Bob Johnson  ',   # Spaces
                'Alice Williams',    # Normal
                'John Duplicate',    # Duplicate email
                'Grace Taylor',      # Null email
                '  SARAH MOORE  '    # Spaces + uppercase
            ],
            'email': [
                '  john@example.com  ',
                'JANE@EXAMPLE.COM',
                'bob@example.com  ',
                '  alice@example.com',
                '  JOHN@EXAMPLE.COM',  # Duplicate
                None,
                '  sarah@EXAMPLE.com  '
            ]
        })
        
        result = clean_dataframe(df)
        
        # Should have 5 rows (removed 1 null, 1 duplicate)
        self.assertEqual(len(result), 5)
        
        # All names should be lowercase and trimmed
        for name in result['name']:
            self.assertEqual(name, name.strip().lower())
        
        # All emails should be lowercase and trimmed
        for email in result['email']:
            self.assertEqual(email, email.strip().lower())
        
        # No duplicates
        self.assertEqual(len(result), len(result['email'].unique()))
        
        # No nulls
        self.assertEqual(result['email'].isna().sum(), 0)


if __name__ == '__main__':
    # Run all tests
    print("=" * 70)
    print("RUNNING TESTS FOR clean_dataframe()")
    print("=" * 70)
    print()
    
    unittest.main(verbosity=2)
