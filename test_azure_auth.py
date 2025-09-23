#!/usr/bin/env python3
"""
Test Azure authentication and services
"""

import sys
sys.path.append('.')

from azure_config import azure_config
from azure_storage_helper import azure_storage

def test_azure_services():
    print("🔐 Testing Azure Authentication...")
    
    # Test credential
    if azure_config.credential:
        print("✅ Azure credentials loaded successfully")
        print(f"   - Tenant ID: {azure_config.AZURE_TENANT_ID[:8]}...")
        print(f"   - Client ID: {azure_config.AZURE_CLIENT_ID[:8]}...")
    else:
        print("❌ No Azure credentials available")
        return False
    
    # Test Blob Storage
    print("\n☁️ Testing Blob Storage...")
    try:
        blob_client = azure_config.get_blob_service_client()
        if blob_client:
            print("✅ Blob Service Client created successfully")
            
            # Test container access
            container_client = blob_client.get_container_client(azure_config.STORAGE_CONTAINER_NAME)
            if container_client.exists():
                print(f"✅ Container '{azure_config.STORAGE_CONTAINER_NAME}' exists and accessible")
            else:
                print(f"⚠️ Container '{azure_config.STORAGE_CONTAINER_NAME}' does not exist, creating...")
                container_client.create_container()
                print(f"✅ Container '{azure_config.STORAGE_CONTAINER_NAME}' created successfully")
                
            # List blobs
            datasets = azure_storage.list_datasets()
            print(f"📁 Found {len(datasets)} datasets in storage")
            
        else:
            print("❌ Failed to create Blob Service Client")
            
    except Exception as e:
        print(f"❌ Blob Storage test failed: {e}")
    
    # Test ML Workspace
    print("\n🧠 Testing ML Workspace...")
    try:
        ml_client = azure_config.get_ml_client()
        if ml_client:
            print("✅ ML Client created successfully")
            # Try to get workspace info
            workspace = ml_client.workspaces.get(azure_config.WORKSPACE_NAME)
            print(f"✅ ML Workspace '{workspace.name}' accessible")
            print(f"   - Location: {workspace.location}")
            print(f"   - Resource Group: {workspace.resource_group}")
        else:
            print("❌ Failed to create ML Client")
            
    except Exception as e:
        print(f"❌ ML Workspace test failed: {e}")
    
    print("\n🎯 Azure services test completed!")

if __name__ == "__main__":
    test_azure_services()
