#!/usr/bin/env python3
"""
Demonstration script showing clustering behavior and round-robin ordering.
This script loads pre-computed diverse instances from the cache folder and shows 
which cluster each datapoint belongs to and how the round-robin selection works.
"""

import sys
import os
import pickle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Only import what we need for the demonstration
try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, synthetic data demonstrations will be skipped")


def load_cached_diverse_instances():
    """Load the cached diverse instances from the adult dataset."""
    
    print("=== LOADING CACHED DIVERSE INSTANCES DEMONSTRATION ===\n")
    
    # Load the cached adult diverse instances (which has clustering format)
    cache_file = "cache/adult-diverse-instances.pkl"
    
    print(f"Loading diverse instances from: {cache_file}")
    
    with open(cache_file, 'rb') as f:
        result = pickle.load(f)
    
    print(f"Loaded diverse instances successfully!")
    print(f"Type: {type(result)}")
    
    if isinstance(result, dict):
        print(f"Number of clusters: {len(result)}")
        total_instances = 0
        
        for cluster_id, instance_indices in result.items():
            print(f"\nCluster {cluster_id}:")
            print(f"  - Contains {len(instance_indices)} instances")
            print(f"  - Instance indices: {instance_indices}")
            total_instances += len(instance_indices)
        
        print(f"\nTotal instances across all clusters: {total_instances}")
        
        return result
    else:
        print("Warning: Loaded data is not in expected dictionary format")
        return None


def simulate_diverse_instances_clustering():
    """Load and demonstrate the cached clustering results."""
    
    # Load the cached diverse instances
    result = load_cached_diverse_instances()
    
    print(f"\n=== CLUSTERING RESULTS ===")
    print(f"Type of result: {type(result)}")
    
    if isinstance(result, dict):
        print(f"Number of clusters found: {len(result)}")
        total_instances = 0
        
        for cluster_id, instance_indices in result.items():
            print(f"\nCluster {cluster_id}:")
            print(f"  - Selected {len(instance_indices)} instances")
            print(f"  - Instance indices: {instance_indices}")
            total_instances += len(instance_indices)
        
        print(f"\nTotal instances selected: {total_instances}")
        instances_per_cluster = len(next(iter(result.values()))) if result else 0
        print(f"Instances per cluster: {instances_per_cluster}")
        
        # Simulate round-robin ordering
        print(f"\n=== ROUND-ROBIN ORDERING SIMULATION ===")
        round_robin_order = create_round_robin_order(result)
        print(f"Round-robin order: {round_robin_order}")
        
        # Show which cluster each position belongs to
        print(f"\nPosition -> Cluster mapping:")
        for pos, instance_id in enumerate(round_robin_order):
            for cluster_id, instances in result.items():
                if instance_id in instances:
                    cluster_pos = instances.index(instance_id)
                    print(f"  Position {pos+1}: Instance {instance_id} from Cluster {cluster_id} "
                          f"(position {cluster_pos+1} within cluster)")
                    break
        
        # Answer the original question about datapoints 1,2,3,4,5
        print(f"\n=== DATAPOINT CLUSTER ASSIGNMENT ===")
        for datapoint_idx in range(5):  # Check datapoints 1,2,3,4,5
            if datapoint_idx < len(round_robin_order):
                instance_id = round_robin_order[datapoint_idx]
                for cluster_id, instances in result.items():
                    if instance_id in instances:
                        cluster_pos = instances.index(instance_id)
                        print(f"Datapoint {datapoint_idx+1}: Instance {instance_id} belongs to Cluster {cluster_id} "
                              f"(position {cluster_pos+1} within cluster)")
                        break
            else:
                print(f"Datapoint {datapoint_idx+1}: Not available (only {len(round_robin_order)} instances total)")
    
    return result


def create_round_robin_order(cluster_data):
    """
    Simulate the round-robin ordering logic from ExperimentHelper.
    """
    if not cluster_data:
        return []
        
    # Get all cluster instance lists
    cluster_lists = list(cluster_data.values())
    max_cluster_size = max(len(cluster_list) for cluster_list in cluster_lists)
    
    round_robin_instances = []
    
    # Round-robin through clusters
    for i in range(max_cluster_size):
        for cluster_list in cluster_lists:
            if i < len(cluster_list):  # If this cluster still has instances at this position
                round_robin_instances.append(cluster_list[i])
    
    return round_robin_instances


def demonstrate_cached_configurations():
    """Show different cached diverse instances files."""
    
    print(f"\n\n=== TESTING DIFFERENT CACHED DATASETS ===\n")
    
    datasets = ["adult", "titanic"]
    
    for dataset in datasets:
        cache_file = f"cache/{dataset}-diverse-instances.pkl"
        if os.path.exists(cache_file):
            print(f"--- {dataset.upper()} Dataset ---")
            
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
            
            print(f"Type: {type(result)}")
            
            if isinstance(result, dict):
                # Clustering format
                total_instances = sum(len(instances) for instances in result.values())
                print(f"Result: {len(result)} clusters, {total_instances} total instances")
                
                round_robin_order = create_round_robin_order(result)
                print(f"Round-robin order (first 10): {round_robin_order[:10]}")
                
                # Show cluster distribution in round-robin order
                cluster_positions = {}
                for idx, instance_id in enumerate(round_robin_order):
                    for cluster_id, cluster_instances in result.items():
                        if instance_id in cluster_instances:
                            if cluster_id not in cluster_positions:
                                cluster_positions[cluster_id] = []
                            cluster_positions[cluster_id].append(idx + 1)  # 1-indexed for display
                            break
                
                print(f"Cluster positions in round-robin: {cluster_positions}")
            elif isinstance(result, list):
                # Legacy format
                print(f"Legacy format: {len(result)} total instances")
                instance_ids = [item['id'] if isinstance(item, dict) else item for item in result]
                print(f"Instance IDs (first 10): {instance_ids[:10]}")
            print()
        else:
            print(f"Cache file not found: {cache_file}")


def demonstrate_different_configurations():
    """Show how different configurations affect the results using synthetic data."""
    
    if not SKLEARN_AVAILABLE:
        print(f"\n\n=== SKIPPING SYNTHETIC DATA DEMO (sklearn not available) ===\n")
        return
    
    print(f"\n\n=== TESTING DIFFERENT CONFIGURATIONS WITH SYNTHETIC DATA ===\n")
    
    from explain.explanations.diverse_instances import DiverseInstances
    
    # Generate the same data for consistency
    np.random.seed(42)
    X, true_labels = make_blobs(n_samples=15, centers=3, n_features=2, 
                               random_state=42, cluster_std=1.0)
    
    configurations = [
        {"instance_amount": 2, "n_clusters": 3},
        {"instance_amount": 3, "n_clusters": 3},
        {"instance_amount": 1, "n_clusters": 5},
    ]
    
    for config in configurations:
        print(f"--- Configuration: {config['instance_amount']} instances per cluster, {config['n_clusters']} clusters ---")
        
        diverse_instances = DiverseInstances(
            instance_amount=config["instance_amount"],
            use_clustering=True,
            n_clusters=config["n_clusters"]
        )
        
        diverse_instances.data_array = X
        diverse_instances.data_indices = list(range(len(X)))
        
        # Use the correct method to trigger clustering
        result = diverse_instances.get_instance_ids_to_show()
        
        if isinstance(result, dict):
            total_instances = sum(len(instances) for instances in result.values())
            print(f"Result: {len(result)} clusters, {total_instances} total instances")
            
            round_robin_order = create_round_robin_order(result)
            print(f"Round-robin order (first 10): {round_robin_order[:10]}")
            
            # Show cluster distribution in round-robin order
            cluster_positions = {}
            for idx, instance_id in enumerate(round_robin_order):
                for cluster_id, cluster_instances in result.items():
                    if instance_id in cluster_instances:
                        if cluster_id not in cluster_positions:
                            cluster_positions[cluster_id] = []
                        cluster_positions[cluster_id].append(idx + 1)  # 1-indexed for display
                        break
            
            print(f"Cluster positions in round-robin: {cluster_positions}")
        print()


if __name__ == "__main__":
    # Run the main demonstration
    result = simulate_diverse_instances_clustering()
    
    # Test different cached configurations
    demonstrate_cached_configurations()
