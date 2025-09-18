"""XAI cache manager that persists heavy explanation artefacts for reuse."""
import os
import pickle as pkl
import threading
import time
from typing import Dict, Optional, Any
import logging

import gin
from explain.action import compute_explanation_report


@gin.configurable
class XAICacheManager:
    """Manage precomputed XAI reports and visual explanations."""

    def __init__(self,
                 dataset_name: str = "adult",
                 cache_location: str = "./cache/{dataset}-xai-reports.pkl"):
        """
        Initialize XAI cache manager.
        
        Args:
            dataset_name: Name of the dataset (for cache file naming)
            cache_location: Path to cache file (with {dataset} placeholder)
        """
        self.dataset_name = dataset_name
        self.cache_location = cache_location.format(dataset=dataset_name)
        self.cache = self.load_cache()
        self.cache_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def load_cache(self) -> Dict:
        """Load existing cache from disk."""
        if os.path.exists(self.cache_location):
            try:
                with open(self.cache_location, 'rb') as file:
                    return pkl.load(file)
            except (pkl.PickleError, EOFError) as e:
                self.logger.warning(f"Failed to load XAI cache: {e}. Starting with empty cache.")
                return {}
        return {}
    
    def save_cache(self):
        """Save cache to disk."""
        with self.cache_lock:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.cache_location), exist_ok=True)
                with open(self.cache_location, 'wb') as file:
                    pkl.dump(self.cache, file)
            except Exception as e:
                self.logger.error(f"Failed to save XAI cache: {e}")
    
    def get_cached_xai_report(self, instance_id: int) -> Optional[Dict[str, Any]]:
        """
        Get precomputed XAI report for instance if available.
        
        Args:
            instance_id: ID of the instance
            
        Returns:
            Cached XAI data if available, None otherwise
        """
        with self.cache_lock:
            if instance_id in self.cache:
                cached_data = self.cache[instance_id]
                if cached_data.get("is_valid", False):
                    self.logger.info(f"Cache hit for instance {instance_id}")
                    return cached_data
                else:
                    self.logger.info(f"Cache entry for instance {instance_id} is invalid")
            else:
                self.logger.info(f"Cache miss for instance {instance_id}")
        return None
    
    def compute_and_cache_xai_report(self, conversation, instance_id: int, 
                                   feature_display_name_mapping=None) -> Dict[str, Any]:
        """
        Compute XAI report and visual explanations for an instance and cache them.
        
        Args:
            conversation: The conversation object
            instance_id: ID of the instance to compute for
            feature_display_name_mapping: Optional feature name mapping
            
        Returns:
            Computed XAI data
        """
        try:
            self.logger.info(f"Computing XAI report for instance {instance_id}")
            
            # Compute XAI report
            xai_report = compute_explanation_report(
                conversation=conversation,
                instance_id=instance_id,
                build_temp_dataset=True,
                instance_type_naming="instance",
                feature_display_name_mapping=feature_display_name_mapping,
                as_text=True
            )
            
            # Cache the results (visual explanations will be computed in logic.py when needed)
            cached_data = {
                "xai_report": xai_report,
                "visual_explanations": {},  # Will be populated when needed in logic.py
                "agent_initialization_data": {
                    "xai_report": xai_report,
                    "instance_id": instance_id,
                    "computation_time": time.time()
                },
                "computation_timestamp": time.time(),
                "is_valid": True
            }
            
            with self.cache_lock:
                self.cache[instance_id] = cached_data
            
            # Save to disk in background
            threading.Thread(target=self.save_cache, daemon=True).start()
            
            self.logger.info(f"Successfully cached XAI report for instance {instance_id}")
            return cached_data
            
        except Exception as e:
            self.logger.error(f"Failed to compute XAI report for instance {instance_id}: {e}")
            # Return empty data structure to prevent crashes
            return {
                "xai_report": {},
                "visual_explanations": {},
                "agent_initialization_data": {},
                "computation_timestamp": time.time(),
                "is_valid": False
            }
    
    def clear_cache(self):
        """Clear all cached data."""
        with self.cache_lock:
            self.cache.clear()
        self.save_cache()
        self.logger.info("XAI cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            total_entries = len(self.cache)
            valid_entries = sum(1 for data in self.cache.values() if data.get("is_valid", False))
            return {
                "total_entries": total_entries,
                "valid_entries": valid_entries,
                "cache_size_mb": os.path.getsize(self.cache_location) / (1024 * 1024) if os.path.exists(self.cache_location) else 0
            }
