from azure.ai.ml import command, Input, Output
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="12136d29-060f-4b5f-98e2-4cc8b6cadc5c",
    resource_group_name="tleio",
    workspace_name="tleio-group-ws"
)

ds = ml_client.datastores.get("workspaceblobstore")

job = command(
    code="./",
    # We use ${{outputs.processed_data}} to tell Azure to insert the 
    # correct mount path directly into the CLI string.
    command="python scripts/download/download_eds.py ${{outputs.processed_data}}/eds --seq 6 --remove-images",
    outputs={
        "processed_data": Output(
            type="uri_folder",
            path=f"azureml://datastores/{ds.name}/paths",
            mode="rw_mount" # Ensures the compute node can read and write to the blob
        )
    },
    environment="tleio-env@latest",
    compute="cpu-cluster",
    display_name="EDS-Download-and-Process"
)

# Submit the job
ml_client.jobs.create_or_update(job)
