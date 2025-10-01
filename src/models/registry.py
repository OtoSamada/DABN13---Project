from typing import Dict, Any

class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, Any] = {}

    def register_model(self, model_name: str, model: Any) -> None:
        """Register a new model."""
        if model_name in self.models:
            raise ValueError(f"Model '{model_name}' is already registered.")
        self.models[model_name] = model

    def get_model(self, model_name: str) -> Any:
        """Retrieve a registered model by name."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not registered.")
        return self.models[model_name]

    def list_models(self) -> Dict[str, Any]:
        """List all registered models."""
        return self.models

    def unregister_model(self, model_name: str) -> None:
        """Unregister a model by name."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not registered.")
        del self.models[model_name]