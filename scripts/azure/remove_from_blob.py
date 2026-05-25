from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# 1. Setup the client
# Use the 'account_name' found in your URI (workspaceblobstore)
account_url = "https://tleiogroupws3247560297.blob.core.windows.net"
blob_service_client = BlobServiceClient(account_url, credential=DefaultAzureCredential())

# 2. Point to the container
# Tip: Check 'Datastores' in Azure ML Studio to confirm the exact container name
container_name = "azureml-blobstore-208da79e-7c0b-4d5b-8e49-d93f3f3477a4"
container_client = blob_service_client.get_container_client(container_name)

# 3. Delete the specific prefix (the path from your URI)
target_path = "tartanair/carwelding"
blobs_to_delete = container_client.list_blobs(name_starts_with=target_path)

for blob in blobs_to_delete:
    container_client.delete_blob(blob.name)
    print(f"Deleted: {blob.name}")

print("Cleanup complete.")
