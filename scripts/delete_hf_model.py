"""
huggingface-cli delete-repo username/model_name --type model
"""

from huggingface_hub import delete_repo
from huggingface_hub import list_models

my_models = list_models(author="aliangdw")

for model in my_models:
    print(model.modelId)

    model_id = model.modelId

    match = ["relative", "2"]
    
    if any(pattern in model_id for pattern in match):
        delete_repo(
            repo_id=model_id,
            repo_type="model"  
        )