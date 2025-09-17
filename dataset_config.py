"""
Dataset Configuration Singleton

Centralized configuration for dataset-specific settings that need to be accessible
throughout the application without parameter passing.

This singleton is configured via gin files and provides global access to 
dataset-specific descriptions and naming conventions.
"""

import gin


@gin.configurable
class DatasetConfig:
    """
    Global dataset configuration accessible from anywhere in the application.
    
    This singleton stores dataset-specific settings like class descriptions and
    instance naming conventions. It's configured via gin files and eliminates
    the need to pass these parameters through multiple layers.
    
    Attributes:
        class_0_description: Human-readable description for class 0 (e.g., "under $50K", "no diabetes")
        class_1_description: Human-readable description for class 1 (e.g., "over $50K", "diabetes")  
        instance_type_naming: Type of instance being explained (e.g., "Person", "Patient", "Passenger")
        target_variable_name: Name of the target variable (e.g., "Income", "Y", "Survived")
    """
    
    _instance = None
    _initialized = False
    
    def __init__(self, 
                 class_0_description: str = "class 0",
                 class_1_description: str = "class 1", 
                 instance_type_naming: str = "instance", 
                 target_variable_name: str = "target"):
        """
        Initialize dataset configuration.
        
        Args:
            class_0_description: Human-readable description for class 0
            class_1_description: Human-readable description for class 1
            instance_type_naming: Type of instance being explained
            target_variable_name: Name of the target variable
        """
        # Prevent re-initialization of singleton
        if DatasetConfig._initialized:
            return
            
        self.class_0_description = class_0_description
        self.class_1_description = class_1_description
        self.instance_type_naming = instance_type_naming
        self.target_variable_name = target_variable_name
        
        DatasetConfig._initialized = True
    
    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of DatasetConfig.
        
        Returns:
            DatasetConfig: The singleton instance
            
        Raises:
            RuntimeError: If called before gin configuration is loaded
        """
        if cls._instance is None:
            # Try to create instance - gin should have configured this
            try:
                cls._instance = cls()
            except Exception as e:
                raise RuntimeError(
                    "DatasetConfig not properly initialized. "
                    "Make sure gin configuration is loaded before accessing DatasetConfig."
                ) from e
        return cls._instance
    
    @classmethod 
    def reset_instance(cls):
        """Reset the singleton instance. Useful for testing."""
        cls._instance = None
        cls._initialized = False
    
    def __repr__(self):
        return (f"DatasetConfig("
                f"class_0_description='{self.class_0_description}', "
                f"class_1_description='{self.class_1_description}', "
                f"instance_type_naming='{self.instance_type_naming}', "
                f"target_variable_name='{self.target_variable_name}')")