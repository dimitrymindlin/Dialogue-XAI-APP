"""
XAI Cache Manager for precomputing and storing heavy XAI reports and visualizations.
Extends the existing cache system to support background computation and instant retrieval.
"""
import os
import pickle as pkl
import threading
import time
from typing import Dict, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor

import gin
from explain.action import compute_explanation_report


@gin.configurable
class XAICacheManager:
    """Manages precomputed XAI reports and visual explanations with background computation."""
    
    def __init__(self, 
                 dataset_name: str = "adult",
                 cache_location: str = "./cache/{dataset}-xai-reports.pkl",
                 max_background_workers: int = 2,
                 precompute_ahead: int = 2):
        """
        Initialize XAI cache manager.
        
        Args:
            dataset_name: Name of the dataset (for cache file naming)
            cache_location: Path to cache file (with {dataset} placeholder)
            max_background_workers: Maximum number of background computation threads
            precompute_ahead: Number of instances to precompute ahead of current position
        """
        self.dataset_name = dataset_name
        self.cache_location = cache_location.format(dataset=dataset_name)
        self.max_background_workers = max_background_workers
        self.precompute_ahead = precompute_ahead
        self.cache = self.load_cache()
        self.cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_background_workers)
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
    
    def precompute_instances_background(self, conversation, instance_ids: list[int], 
                                      feature_display_name_mapping=None):
        """
        Precompute XAI reports for multiple instances in background.
        
        Args:
            conversation: The conversation object
            instance_ids: List of instance IDs to precompute
            feature_display_name_mapping: Optional feature name mapping
        """
        def compute_worker(instance_id):
            if self.get_cached_xai_report(instance_id) is None:
                self.compute_and_cache_xai_report(conversation, instance_id, feature_display_name_mapping)
        
        # Submit background tasks
        for instance_id in instance_ids:
            self.executor.submit(compute_worker, instance_id)
        
        self.logger.info(f"Submitted {len(instance_ids)} instances for background computation")
    
    def precompute_ahead(self, conversation, current_datapoint_count: int, 
                        total_train_instances: int, feature_display_name_mapping=None):
        """
        Precompute XAI reports for upcoming instances.
        
        Args:
            conversation: The conversation object
            current_datapoint_count: Current position in training instances
            total_train_instances: Total number of training instances
            feature_display_name_mapping: Optional feature name mapping
        """
        # Calculate which instances to precompute
        start_idx = current_datapoint_count + 1
        end_idx = min(start_idx + self.precompute_ahead, total_train_instances)
        
        if start_idx < total_train_instances:
            # Get instance IDs for the range (this depends on your instance management)
            # You'll need to integrate this with your ExperimentHelper
            upcoming_instance_ids = list(range(start_idx, end_idx))
            
            self.precompute_instances_background(
                conversation, upcoming_instance_ids, feature_display_name_mapping
            )
    
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
    
    def shutdown(self):
        """Shutdown the cache manager and background workers."""
        self.executor.shutdown(wait=True)
        self.save_cache()
