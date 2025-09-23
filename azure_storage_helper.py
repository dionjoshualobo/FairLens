"""
Azure Blob Storage Helper for FairLens
Handles dataset uploads, downloads, and management in Azure Blob Storage
"""

import streamlit as st
import pandas as pd
import io
from azure_config import azure_config

class AzureStorageHelper:
    def __init__(self):
        self.blob_service_client = azure_config.get_blob_service_client()
        self.container_name = azure_config.STORAGE_CONTAINER_NAME
        
    def upload_dataframe_to_blob(self, dataframe, filename):
        """Upload pandas DataFrame to Azure Blob Storage as CSV"""
        try:
            if not self.blob_service_client:
                return False, "Azure Blob Storage not connected"
                
            # Convert DataFrame to CSV bytes
            csv_buffer = io.StringIO()
            dataframe.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode('utf-8')
            
            # Upload to blob
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=filename
            )
            
            blob_client.upload_blob(csv_bytes, overwrite=True)
            return True, f"Successfully uploaded {filename} to Azure Storage"
            
        except Exception as e:
            return False, f"Error uploading to Azure Storage: {str(e)}"
    
    def download_blob_to_dataframe(self, filename):
        """Download CSV from Azure Blob Storage as pandas DataFrame"""
        try:
            if not self.blob_service_client:
                return None, "Azure Blob Storage not connected"
                
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=filename
            )
            
            # Download blob data
            blob_data = blob_client.download_blob().readall()
            csv_string = blob_data.decode('utf-8')
            
            # Convert to DataFrame
            df = pd.read_csv(io.StringIO(csv_string))
            return df, "Successfully downloaded from Azure Storage"
            
        except Exception as e:
            return None, f"Error downloading from Azure Storage: {str(e)}"
    
    def list_datasets(self):
        """List all datasets in Azure Blob Storage"""
        try:
            if not self.blob_service_client:
                return []
                
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            
            blobs = []
            for blob in container_client.list_blobs():
                if blob.name.endswith('.csv'):
                    blobs.append({
                        'name': blob.name,
                        'size': blob.size,
                        'last_modified': blob.last_modified
                    })
            
            return blobs
            
        except Exception as e:
            st.error(f"Error listing datasets: {str(e)}")
            return []
    
    def upload_file_to_blob(self, uploaded_file):
        """Upload Streamlit uploaded file to Azure Blob Storage"""
        try:
            if not self.blob_service_client:
                return False, "Azure Blob Storage not connected"
            
            # Create blob name with timestamp to avoid conflicts
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"uploaded_{timestamp}_{uploaded_file.name}"
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=filename
            )
            
            # Upload the file
            uploaded_file.seek(0)  # Reset file pointer
            blob_client.upload_blob(uploaded_file.read(), overwrite=True)
            
            return True, f"Successfully uploaded {filename} to Azure Storage"
            
        except Exception as e:
            return False, f"Error uploading file: {str(e)}"

# Global storage helper instance
azure_storage = AzureStorageHelper()
