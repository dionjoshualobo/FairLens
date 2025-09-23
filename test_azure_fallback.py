#!/usr/bin/env python3
"""
Test script to verify Azure fallback functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from azure_config import azure_config
from azure_storage_helper import azure_storage

def test_azure_fallback():
    """Test Azure functionality with fallback"""
    
    print("🧪 Testing Azure Fallback Functionality")
    print("=" * 50)
    
    # Test 1: Azure Configuration
    print("\n1. Testing Azure Configuration...")
    print(f"   Storage Account: {azure_config.STORAGE_ACCOUNT_NAME}")
    print(f"   Container Name: {azure_config.STORAGE_CONTAINER_NAME}")
    print(f"   Credential Available: {azure_config.credential is not None}")
    
    # Test 2: Blob Service Client
    print("\n2. Testing Blob Service Client...")
    blob_client = azure_config.get_blob_service_client()
    print(f"   Blob Client Available: {blob_client is not None}")
    
    # Test 3: Storage Helper
    print("\n3. Testing Storage Helper...")
    print(f"   Storage Helper Initialized: {azure_storage.blob_service_client is not None}")
    
    # Test 4: List Datasets (should return empty list gracefully)
    print("\n4. Testing List Datasets...")
    datasets = azure_storage.list_datasets()
    print(f"   Datasets Found: {len(datasets)}")
    print(f"   Function Returned: {type(datasets)}")
    
    # Test 5: ML Client
    print("\n5. Testing ML Client...")
    ml_client = azure_config.get_ml_client()
    print(f"   ML Client Available: {ml_client is not None}")
    
    print("\n✅ Azure Fallback Test Complete!")
    print("🎯 The application should run in demo mode when Azure services are unavailable.")
    
if __name__ == "__main__":
    test_azure_fallback()
