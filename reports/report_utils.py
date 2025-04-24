"""
Utility functions for Two-Tower model reports.

This module provides helper functions for finding experiment files,
resolving W&B entities, and loading dataset genealogy information.
"""

import os
import glob
import json
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

import wandb

# Get logger for this module
logger = logging.getLogger('two_tower.report_utils')

def find_experiment_files(run_id: Optional[str] = None, number: int = 5) -> List[Path]:
    """
    Find experiment files in the workspace.
    If run_id is provided, will look for files related to that specific run.
    Otherwise returns the most recent experiment files.
    
    Args:
        run_id: W&B run ID to filter files
        number: Number of files to return if run_id not specified
        
    Returns:
        List of experiment file paths sorted by recency
    """
    try:
        # Get all experiment files
        experiment_files = glob.glob("experiments/*.json")
        
        # Convert to Path objects
        experiment_paths = [Path(f) for f in experiment_files]
        
        # Sort by modification time (most recent first)
        experiment_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        
        if not experiment_paths:
            logger.warning("No experiment files found in the experiments/ directory")
            return []
            
        if run_id:
            # Filter files that contain the run_id
            matching_files = []
            for file_path in experiment_paths:
                try:
                    data = json.loads(file_path.read_text())
                    if 'wandb_run_id' in data and data['wandb_run_id'] == run_id:
                        matching_files.append(file_path)
                except Exception as e:
                    logger.warning(f"Error reading experiment file {file_path}: {str(e)}")
                    continue
                    
            if matching_files:
                logger.info(f"Found {len(matching_files)} experiment files matching run_id {run_id}")
                return matching_files
            else:
                logger.warning(f"No experiment files found with run_id {run_id}")
                return []
        else:
            # Return the most recent files up to the specified number
            logger.info(f"Returning {min(number, len(experiment_paths))} most recent experiment files")
            return experiment_paths[:number]
            
    except Exception as e:
        logger.error(f"Error finding experiment files: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def resolve_entity(entity_cfg: Optional[str]) -> str:
    """
    Resolve the W&B entity to use with multiple fallback options.
    
    Args:
        entity_cfg: Configured entity from arguments
        
    Returns:
        Resolved entity name
    """
    if entity_cfg:
        return entity_cfg
    
    # First try: current wandb run if available
    try:
        if wandb.run is not None:
            entity = wandb.run.entity
            logger.info(f"Using entity from current wandb.run: {entity}")
            return entity
    except Exception as e:
        logger.warning(f"Error getting entity from current run: {str(e)}")
    
    # Second try: default entity from API
    try:
        api = wandb.Api()
        entity = api.default_entity
        if entity:
            logger.info(f"Using default entity from wandb.Api(): {entity}")
            return entity
    except Exception as e:
        logger.warning(f"Error getting default entity from API: {str(e)}")
    
    # Third try: environment variable
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        logger.info(f"Using entity from WANDB_ENTITY environment variable: {entity}")
        return entity
    
    # Fourth try: try to get from config
    try:
        # First try the YAML config
        from twotower.utils import load_config
        try:
            default_config = load_config("configs/default_config.yml")
            entity = default_config.get("wandb", {}).get("entity")
            if entity:
                logger.info(f"Using entity from default_config.yml: {entity}")
                return entity
        except (FileNotFoundError, KeyError) as e:
            logger.warning(f"Could not load WANDB_ENTITY from default_config.yml: {str(e)}")

        # Fall back to old config if needed
        from config import WANDB_ENTITY
        if WANDB_ENTITY:
            logger.info(f"Using entity from config.py: {WANDB_ENTITY}")
            return WANDB_ENTITY
    except (ImportError, AttributeError):
        logger.warning("Could not import WANDB_ENTITY from config")
    
    logger.warning("Could not resolve W&B entity, will use None")
    return ""

def resolve_run_id(project: str, entity: str, run_id: Optional[str]) -> Optional[str]:
    """
    Resolve the run ID to use with fallback options.
    
    Args:
        project: W&B project name
        entity: W&B entity name
        run_id: Run ID from arguments
        
    Returns:
        Resolved run ID
    """
    if run_id:
        return run_id
    
    # First try: current wandb run
    try:
        if wandb.run is not None:
            resolved_run_id = wandb.run.id
            logger.info(f"Using current wandb.run id: {resolved_run_id}")
            return resolved_run_id
    except Exception as e:
        logger.warning(f"Error getting run ID from current run: {str(e)}")
    
    # Second try: get the most recent run from the project
    logger.info("Attempting to find most recent run from API")
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}", per_page=1)
        if len(runs) > 0:
            resolved_run_id = runs[0].id
            logger.info(f"Using most recent run from project: {resolved_run_id}")
            return resolved_run_id
        else:
            logger.error(f"No runs found in project {entity}/{project}")
            return None
    except Exception as e:
        logger.error(f"Error fetching runs from API: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def load_genealogy(file_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """
    Load dataset genealogy from file.
    
    Args:
        file_path: Path to genealogy file
        
    Returns:
        Dictionary with genealogy data or None if file cannot be loaded
    """
    if not file_path:
        return None
    
    try:
        genealogy = json.loads(file_path.read_text())
        logger.info(f"Loaded dataset genealogy from {file_path}")
        return genealogy
    except Exception as e:
        logger.warning(f"Could not load genealogy {file_path}: {e}")
        return None

def format_genealogy_markdown(genealogy: Dict[str, Any]) -> str:
    """
    Format dataset genealogy as Markdown.
    
    Args:
        genealogy: Dataset genealogy dictionary
        
    Returns:
        Markdown-formatted string
    """
    markdown = ""
    
    # Basic experiment info
    if 'experiment_id' in genealogy:
        markdown += f"**Experiment ID:** {genealogy.get('experiment_id', 'N/A')}\n\n"
    
    if 'ms_marco_split' in genealogy:
        markdown += f"**MS MARCO Split:** {genealogy.get('ms_marco_split', 'N/A')}\n\n"
    
    if 'preset_file' in genealogy:
        markdown += f"**Preset File:** {genealogy.get('preset_file', 'N/A')}\n\n"
    
    # More details
    if 'random_seed' in genealogy:
        markdown += f"**Random Seed:** {genealogy.get('random_seed', 'N/A')}\n\n"
    
    if 'sample_size' in genealogy:
        markdown += f"**Sample Size:** {genealogy.get('sample_size', 'All')} samples\n\n"
    
    # Preset config
    if 'preset_config' in genealogy:
        markdown += "**Preset Configuration:**\n```json\n"
        markdown += json.dumps(genealogy["preset_config"], indent=2)
        markdown += "\n```\n\n"
    
    # Preprocessing steps
    if 'preprocessing_steps' in genealogy:
        markdown += "**Preprocessing Steps:**\n"
        for i, step in enumerate(genealogy["preprocessing_steps"]):
            step_time = step.get("timestamp", "N/A").split("T")[1].split(".")[0] if "timestamp" in step else "N/A"
            markdown += f"{i+1}. **{step.get('step', 'unknown')}** ({step_time})\n"
            for k, v in step.items():
                if k not in ["step", "timestamp"]:
                    markdown += f"   - {k}: {v}\n"
    
    # Dataset info
    if 'triplets_info' in genealogy:
        markdown += "\n**Triplets Dataset:**\n"
        for k, v in genealogy["triplets_info"].items():
            markdown += f"- {k}: {v}\n"
    
    if 'sampled_dataset_info' in genealogy:
        markdown += "\n**Sampled Dataset:**\n"
        for k, v in genealogy["sampled_dataset_info"].items():
            markdown += f"- {k}: {v}\n"
    
    return markdown

def create_mermaid_flowchart(genealogy: Dict[str, Any]) -> str:
    """
    Create a Mermaid.js flowchart from dataset genealogy.
    
    Args:
        genealogy: Dataset genealogy dictionary
        
    Returns:
        Mermaid.js flowchart in markdown
    """
    mermaid = "```mermaid\nflowchart TD\n"
    
    # Dataset source
    mermaid += "A[MS MARCO Dataset] --> B[Convert to Parquet]\n"
    
    # Triplets creation
    triplets_count = genealogy.get('triplets_info', {}).get('row_count', '?')
    mermaid += f"B --> C[Create Triplets Format\\n{triplets_count} triplets]\n"
    
    # Sampling if applicable
    sample_size = genealogy.get('sample_size')
    if sample_size:
        mermaid += f"C --> D[Sample Dataset\\n{sample_size} samples]\n"
    else:
        mermaid += "C --> D[Use Full Dataset]\n"
    
    # Training
    mermaid += "D --> E[Training Data]\n```"
    
    return mermaid

def create_experiment_timeline(experiment_files: List[Path]) -> str:
    """
    Create a Mermaid.js timeline from experiment files.
    
    Args:
        experiment_files: List of experiment file paths
        
    Returns:
        Mermaid.js timeline in markdown
    """
    if not experiment_files:
        return "```mermaid\ntimeline\n    title Experiment Timeline\n    section Experiments\n    ✅ Current Run : Active\n```"
    
    try:
        # Load experiment summaries
        experiment_summaries = []
        for file_path in experiment_files:
            try:
                data = json.loads(file_path.read_text())
                experiment_summaries.append(data)
            except Exception as e:
                logger.warning(f"Error loading experiment summary {file_path}: {str(e)}")
        
        # Sort by timestamp
        experiment_summaries.sort(key=lambda x: x.get("timestamp", ""))
        
        # Create timeline markdown
        timeline_markdown = "```mermaid\ntimeline\n    title Experiment Timeline\n"
        
        # Group by day
        current_date = None
        for summary in experiment_summaries:
            timestamp = summary.get("timestamp", "")
            if timestamp:
                date = timestamp.split("T")[0]
                if date != current_date:
                    current_date = date
                    timeline_markdown += f"    title {date}\n"
                
                # Add experiment to timeline
                experiment_id = summary.get("experiment_id", "unknown")
                config_file = Path(summary.get("config_file", "unknown")).stem
                success = summary.get("success", False)
                training_time = summary.get("total_training_time", 0)
                
                # Format time string
                time_str = timestamp.split("T")[1].split(".")[0] if "T" in timestamp else ""
                
                success_icon = "✅" if success else "❌"
                timeline_markdown += f"    section {time_str}\n"
                timeline_markdown += f"    {success_icon} Exp {experiment_id[-8:]}: {config_file} ({training_time:.1f}s)\n"
        
        timeline_markdown += "```"
        return timeline_markdown
    
    except Exception as e:
        logger.warning(f"Error creating experiment timeline: {str(e)}")
        return "```mermaid\ntimeline\n    title Experiment Timeline\n    section Experiments\n    ✅ Current Run : Active\n```" 