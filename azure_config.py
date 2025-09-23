"""
Azure Configuration for FairLens
Contains all Azure service configurations and connection strings
"""

import os
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential, ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from azure.ai.ml import MLClient
import streamlit as st

class AzureConfig:
    def __init__(self):
        # Azure Storage Configuration
        self.STORAGE_ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT_NAME', 'fairlensmlwork3450594214')
        self.STORAGE_CONTAINER_NAME = os.getenv('AZURE_STORAGE_CONTAINER_NAME', 'datasets')
        
        # Azure ML Configuration
        self.SUBSCRIPTION_ID = os.getenv('AZURE_SUBSCRIPTION_ID', 'de944214-96b1-4af1-8059-53168fd5d1fd')
        self.RESOURCE_GROUP_NAME = os.getenv('AZURE_RESOURCE_GROUP_NAME', 'fairlens-rg')
        self.WORKSPACE_NAME = os.getenv('AZURE_ML_WORKSPACE_NAME', 'fairlens-ml-workspace')
        
        # Service Principal Configuration (from environment variables only)
        self.AZURE_CLIENT_ID = os.getenv('AZURE_CLIENT_ID')
        self.AZURE_CLIENT_SECRET = os.getenv('AZURE_CLIENT_SECRET')
        self.AZURE_TENANT_ID = os.getenv('AZURE_TENANT_ID')
        
        # Initialize Azure credentials with fallback
        self.credential = self._get_credential()
        
    def _get_credential(self):
        """Get Azure credential with service principal authentication"""
        try:
            # Try Service Principal first (production authentication)
            if self.AZURE_CLIENT_ID and self.AZURE_CLIENT_SECRET and self.AZURE_TENANT_ID:
                print("Using Service Principal authentication")
                return ClientSecretCredential(
                    tenant_id=self.AZURE_TENANT_ID,
                    client_id=self.AZURE_CLIENT_ID,
                    client_secret=self.AZURE_CLIENT_SECRET
                )
        except Exception as e:
            print(f"Service Principal authentication failed: {e}")
        
        try:
            # Try Managed Identity (for Azure environments)
            return ManagedIdentityCredential()
        except Exception as e:
            print(f"Managed Identity authentication failed: {e}")
            
        try:
            # Fallback to DefaultAzureCredential
            return DefaultAzureCredential()
        except Exception as e:
            print(f"Default Azure Credential failed: {e}")
            
        print("All authentication methods failed. Running in demo mode.")
        return None
        
    def get_blob_service_client(self):
        """Get Azure Blob Service client"""
        if self.credential is None:
            return None
            
        try:
            return BlobServiceClient(
                account_url=f"https://{self.STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
                credential=self.credential
            )
        except Exception as e:
            print(f"Failed to create blob service client: {e}")
            return None
    
    def get_ml_client(self):
        """Get Azure ML client"""
        if self.credential is None:
            return None
            
        try:
            return MLClient(
                credential=self.credential,
                subscription_id=self.SUBSCRIPTION_ID,
                resource_group_name=self.RESOURCE_GROUP_NAME,
                workspace_name=self.WORKSPACE_NAME
            )
        except Exception as e:
            print(f"Failed to create ML client: {e}")
            return None
    
    def create_container_if_not_exists(self):
        """Create blob container for datasets if it doesn't exist"""
        try:
            blob_service_client = self.get_blob_service_client()
            if blob_service_client:
                container_client = blob_service_client.get_container_client(self.STORAGE_CONTAINER_NAME)
                if not container_client.exists():
                    container_client.create_container()
                    print(f"Created container: {self.STORAGE_CONTAINER_NAME}")
                return True
        except Exception as e:
            print(f"Error creating container: {e}")
            return False

# Global configuration instance
azure_config = AzureConfig()
