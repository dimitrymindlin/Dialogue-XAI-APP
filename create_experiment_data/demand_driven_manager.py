"""
Demand-Driven Test Instance Manager

This module provides the intelligent manager that dynamically requests instances
to achieve class balance, replacing the fixed-batch approach.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

from .demand_driven_generator import DemandDrivenGeneratorInterface

logger = logging.getLogger(__name__)


class DemandDrivenTestInstanceManager:
    """
    Manager that dynamically requests instances to achieve class balance.
    
    This manager uses an iterative approach:
    1. Determine current class imbalance for each phase
    2. Request instances targeting the underrepresented class
    3. Validate and assign instances
    4. Repeat until balance is achieved or all instances are assigned
    
    Note: intro_test uses the same instance as final_test (design constraint)
    """
    
    # Define phases to generate instances for (DRY principle)
    GENERATION_PHASES = ['test', 'final_test']
    ALL_PHASES = ['test', 'intro_test', 'final_test']  # All phases in output format
    
    def __init__(self, model, generator: DemandDrivenGeneratorInterface, data: pd.DataFrame):
        self.model = model
        self.generator = generator
        self.data = data
        self.assignments = {}  # train_id -> {phase -> {instance, predicted_class, type}}
        
    def generate_balanced_instances(self, training_instances: List[int],
                                  target_balance: float = 0.5,
                                  tolerance: float = 0.1) -> Dict[str, Any]:
        """
        Generate instances for all phases with target class balance.
        
        Args:
            training_instances: List of training instance IDs
            target_balance: Target ratio for class 1 (0.5 = 50/50)
            tolerance: Acceptable deviation from target balance
            
        Returns:
            Dictionary with phase assignments and balance report in TestInstances format
        """
        logger.info(f"Starting demand-driven generation for {len(training_instances)} instances")
        logger.info(f"Target balance: {target_balance:.1%} ± {tolerance:.1%}")
        
        results = {}
        
        # Generate balanced instances for test and final_test phases
        for phase in self.GENERATION_PHASES:
            phase_assignments = self._balance_single_phase(
                training_instances, phase, target_balance, tolerance
            )
            results[phase] = phase_assignments
            
        # Convert to TestInstances expected format
        test_instances = self._convert_to_test_instances_format(results, training_instances)

        # Turn to InstanceDatapoints format

        
        # Generate balance report
        balance_report = self._generate_balance_report(results)
        self._print_balance_summary(balance_report)
        
        return test_instances
    
    def _balance_single_phase(self, training_instances: List[int], phase: str,
                            target_balance: float, tolerance: float) -> Dict:
        """Balance instances for a single phase using demand-driven approach."""
        assignments = {}
        class_counts = {0: 0, 1: 0}
        total_instances = len(training_instances)
        target_class_1_count = int(total_instances * target_balance)
        
        # Track generation statistics
        generation_stats = {
            'similar_attempts': 0,
            'counterfactual_attempts': 0,
            'successful_generations': 0,
            'class_targeted_requests': 0,
            'fallback_requests': 0
        }
        
        logger.info(f"Balancing {phase} phase:")
        logger.info(f"  Target: {target_class_1_count} class 1, {total_instances - target_class_1_count} class 0")
        
        for i, train_id in enumerate(training_instances):
            # Log progress at 25%, 50%, 75% intervals
            progress_percent = (i + 1) / total_instances
            if i == 0 or progress_percent in [0.25, 0.50, 0.75] or i == total_instances - 1:
                logger.info(f"  Processing {phase} instances... {progress_percent:.0%} complete ({i+1}/{total_instances})")
            
            original_instance = self.data.loc[train_id:train_id]
            original_class = self.model.predict(original_instance)[0]
            
            # Determine what class we need more of
            current_class_1_count = class_counts[1]
            remaining_instances = total_instances - i
            remaining_class_1_needed = target_class_1_count - current_class_1_count
            
            logger.debug(f"  ID {train_id}: balance {current_class_1_count}/{total_instances-i} class 1, need {remaining_class_1_needed} more")
            
            # Decide on target class based on remaining needs
            if remaining_class_1_needed > remaining_instances:
                # We need class 1 for all remaining instances - impossible
                target_class = 1
                request_type = "urgent_targeted"
            elif remaining_class_1_needed <= 0:
                # We have enough class 1, need class 0
                target_class = 0
                request_type = "targeted"
            elif remaining_class_1_needed == remaining_instances:
                # We need exactly class 1 for all remaining
                target_class = 1
                request_type = "targeted"
            else:
                # We're in balance range, no specific target needed
                target_class = None
                request_type = "flexible"
            
            # Generate appropriate instance
            instance, instance_info = self._generate_balanced_instance(
                original_instance, original_class, target_class, generation_stats
            )
            
            if instance is not None:
                predicted_class = self.model.predict(instance)[0]
                class_counts[predicted_class] += 1
                generation_stats['successful_generations'] += 1
                
                assignments[train_id] = {
                    'instance': instance,
                    'predicted_class': predicted_class,
                    'original_class': original_class,
                    'target_class': target_class,
                    'request_type': request_type,
                    'generation_method': instance_info['method'],
                    'attempts_needed': instance_info['attempts']
                }
                
                logger.debug(f"    ID {train_id}: {instance_info['method']} -> class {predicted_class}")
            else:
                logger.warning(f"    ID {train_id}: Generation FAILED")
                # For failed generation, we need to handle this gracefully
                # We'll create a fallback assignment to maintain the expected structure
                assignments[train_id] = {
                    'instance': None,
                    'predicted_class': None,
                    'original_class': original_class,
                    'target_class': target_class,
                    'request_type': request_type,
                    'generation_method': 'failed',
                    'attempts_needed': 0
                }
        
        # Calculate final balance
        total_assigned = sum(class_counts.values())
        final_balance = class_counts[1] / total_assigned if total_assigned > 0 else 0
        balance_achieved = abs(final_balance - target_balance) <= tolerance
        
        logger.info(f"  {phase.upper()}: Final balance {class_counts[1]}/{total_assigned} = {final_balance:.3f} {'✓' if balance_achieved else '✗'}")
        
        return {
            'assignments': assignments,
            'class_counts': class_counts,
            'balance': final_balance,
            'balance_achieved': balance_achieved,
            'generation_stats': generation_stats
        }
    
    def _generate_balanced_instance(self, original_instance: pd.DataFrame,
                                  original_class: int, target_class: Optional[int],
                                  stats: Dict) -> Tuple[Optional[pd.DataFrame], Dict]:
        """Generate a single instance with optional class targeting."""
        
        # Strategy 1: Try similar instance first if no class constraint or target matches original
        if target_class is None or target_class == original_class:
            stats['similar_attempts'] += 1
            instance = self.generator.generate_similar_instance(
                original_instance, target_class
            )
            if instance is not None:
                return instance, {'method': 'similar', 'attempts': 1}
        
        # Strategy 2: Try counterfactual if we need the opposite class
        if target_class is not None and target_class != original_class:
            stats['counterfactual_attempts'] += 1
            stats['class_targeted_requests'] += 1
            instance = self.generator.generate_counterfactual_instance(
                original_instance, target_class
            )
            if instance is not None:
                return instance, {'method': 'counterfactual', 'attempts': 1}
        
        # Strategy 3: Fallback - try both methods without class constraint
        stats['fallback_requests'] += 1
        
        # Try similar without constraint
        instance = self.generator.generate_similar_instance(original_instance)
        if instance is not None:
            return instance, {'method': 'similar_fallback', 'attempts': 1}
        
        # Try counterfactual without constraint
        instance = self.generator.generate_counterfactual_instance(original_instance)
        if instance is not None:
            return instance, {'method': 'counterfactual_fallback', 'attempts': 1}
        
        return None, {'method': 'failed', 'attempts': 0}
    
    def _convert_to_test_instances_format(self, results: Dict, training_instances: List[int]) -> Dict[str, Any]:
        """
        Convert demand-driven results to the format expected by TestInstances.
        
        The original format expects:
        {instance_id: {phase: instance_dataframe}}
        """
        test_instances = {}
        ids_to_remove = []
        
        for train_id in training_instances:
            test_instances[train_id] = {}
            
            # Check if we have valid instances for all generation phases
            valid_phases = []
            for phase in self.GENERATION_PHASES:
                has_valid_instance = (train_id in results[phase]['assignments'] and 
                                    results[phase]['assignments'][train_id]['instance'] is not None)
                if has_valid_instance:
                    valid_phases.append(phase)
            
            if len(valid_phases) != len(self.GENERATION_PHASES):
                # Mark for removal if we don't have valid instances for all generation phases
                missing_phases = set(self.GENERATION_PHASES) - set(valid_phases)
                logger.debug(f"Train ID {train_id} missing phases: {missing_phases}")
                ids_to_remove.append(train_id)
                continue
            
            # Add instances for each generation phase
            for phase in self.GENERATION_PHASES:
                assignment = results[phase]['assignments'][train_id]
                if phase == 'final_test':
                    # Store under the key that ExplainBot expects: "final-test" 
                    test_instances[train_id]['final-test'] = assignment['instance']
                    # intro_test uses the same instance as final_test (design constraint)
                    test_instances[train_id]['intro_test'] = assignment['instance']
                else:
                    test_instances[train_id][phase] = assignment['instance']
        
        logger.info(f"Generated instances for {len(test_instances)} training instances")
        if ids_to_remove:
            logger.warning(f"Removing {len(ids_to_remove)} instances due to generation failures: {ids_to_remove}")
        
        return test_instances
    
    def _get_current_balance(self, class_counts: Dict) -> float:
        """Calculate the current balance ratio."""
        total = sum(class_counts.values())
        return class_counts[1] / total if total > 0 else 0
    
    def _generate_balance_report(self, results: Dict) -> Dict:
        """Generate comprehensive balance report."""
        report = {
            'phases': {},
            'overall_summary': {
                'total_phases': len(results),
                'balanced_phases': 0,
                'total_instances': 0,
                'total_successful': 0
            }
        }
        
        for phase, phase_data in results.items():
            assignments = phase_data['assignments']
            class_counts = phase_data['class_counts']
            total = sum(class_counts.values())
            
            report['phases'][phase] = {
                'class_distribution': class_counts,
                'total_instances': total,
                'balance_ratio': class_counts[1] / total if total > 0 else 0,
                'balance_achieved': phase_data['balance_achieved'],
                'generation_stats': phase_data['generation_stats']
            }
            
            if phase_data['balance_achieved']:
                report['overall_summary']['balanced_phases'] += 1
            
            report['overall_summary']['total_instances'] += len(assignments)
            report['overall_summary']['total_successful'] += total
        
        return report
    
    def _print_balance_summary(self, balance_report: Dict):
        """Print a summary of the balance results."""
        overall = balance_report['overall_summary']
        
        logger.info("="*50)
        logger.info("DEMAND-DRIVEN GENERATION SUMMARY")
        logger.info("="*50)
        
        for phase, phase_data in balance_report['phases'].items():
            class_dist = phase_data['class_distribution']
            total = phase_data['total_instances']
            balance = phase_data['balance_ratio']
            achieved = phase_data['balance_achieved']
            
            logger.info(f"{phase.upper()}: {class_dist[0]}/{class_dist[1]} " +
                       f"(balance: {balance:.3f}, {'✓' if achieved else '✗'})")
        
        logger.info(f"Overall: {overall['balanced_phases']}/{overall['total_phases']} phases balanced")
        logger.info(f"Success rate: {overall['total_successful']}/{overall['total_instances']} instances")
        logger.info("="*50)
