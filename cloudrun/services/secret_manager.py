import os
from typing import Optional

from google.cloud import secretmanager


def get_secret_value(secret_name: str, project_id: Optional[str] = None) -> str:
    if not secret_name:
        raise ValueError("secret_name is required")
    project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set")

    client = secretmanager.SecretManagerServiceClient()
    secret_path = f"projects/{project}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": secret_path})
    return response.payload.data.decode("utf-8")
