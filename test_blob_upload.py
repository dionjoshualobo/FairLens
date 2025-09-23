#!/usr/bin/env python3
"""
Test Azure Blob Storage upload functionality
"""

import sys
sys.path.append('.')

import pandas as pd
from azure_storage_helper import azure_storage

def test_blob_upload():
    print("🔄 Testing Azure Blob Storage upload...")
    
    # Load the test dataset
    df = pd.read_csv('test_dataset.csv')
    print(f"📁 Loaded test dataset: {df.shape}")
    
    # Test upload to Azure Blob Storage
    success, message = azure_storage.upload_dataframe_to_blob(df, 'test_dataset_from_local.csv')
    
    if success:
        print(f"✅ Upload successful: {message}")
        
        # Test listing datasets
        datasets = azure_storage.list_datasets()
        print(f"📂 Found {len(datasets)} datasets in Azure storage:")
        for dataset in datasets:
            print(f"   - {dataset['name']} ({dataset['size']} bytes)")
            
        # Test download
        print("🔄 Testing download...")
        downloaded_df, download_message = azure_storage.download_blob_to_dataframe('test_dataset_from_local.csv')
        
        if downloaded_df is not None:
            print(f"✅ Download successful: {download_message}")
            print(f"📊 Downloaded data shape: {downloaded_df.shape}")
            print("First few rows:")
            print(downloaded_df.head())
        else:
            print(f"❌ Download failed: {download_message}")
            
    else:
        print(f"❌ Upload failed: {message}")

if __name__ == "__main__":
    test_blob_upload()
