# run from root dir

from azure.ai.ml.entities import Environment
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="12136d29-060f-4b5f-98e2-4cc8b6cadc5c",
    resource_group_name="tleio",
    workspace_name="tleio-group-ws"
)

# Re-register your environment using the curated base
tleio_env = Environment(
    name="tleio-env",
    image="mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cuda11.8-gpu-inference:latest",
    conda_file="./environment.yaml", # This will install your specific 'tartanair' etc.
    description="VIO project environment based on curated ACPT"
)
ml_client.environments.create_or_update(tleio_env)
