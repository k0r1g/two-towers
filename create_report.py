#!/usr/bin/env python
"""
Create a Weights & Biases report for two-tower model experiments.
This script generates a comprehensive report that visualizes model performance,
training dynamics, and retrieval results.
"""

import os
import argparse
import wandb
import wandb_workspaces.reports.v2 as wr
import logging
# Import config settings
from config import WANDB_PROJECT, WANDB_ENTITY

# Get the logger from the main script
logger = logging.getLogger('two_tower')

def create_two_tower_report(project_name=None, entity=None, title=None, description=None, run_id=None):
    """
    Create a W&B report for a two-tower retrieval model project.
    
    Args:
        project_name: Name of the W&B project (defaults to config.WANDB_PROJECT)
        entity: W&B username or team name (defaults to config.WANDB_ENTITY)
        title: Custom title for the report (optional)
        description: Description for the report (optional)
        run_id: The ID of the current run to highlight in the report (optional)
    
    Returns:
        URL to the created report
    """
    try:
        # Use config defaults if not provided
        project_name = project_name or WANDB_PROJECT
        
        # Initialize the report
        report_title = title or f"Two-Tower Model Performance Report"
        report_description = description or (
            "This report analyzes the performance of two-tower models for retrieval. "
            "It includes training metrics, similarity distributions, and retrieval results."
        )
        
        # Use the current run if run_id is not provided
        if run_id is None and wandb.run is not None:
            run_id = wandb.run.id
        
        # Get entity from current run, config, or default to the currently logged in user
        if entity is None:
            if wandb.run is not None:
                entity = wandb.run.entity
            elif WANDB_ENTITY is not None:
                entity = WANDB_ENTITY
            else:
                # Get the default entity (current user) from W&B
                api = wandb.Api()
                entity = api.default_entity or "user"
                logger.info(f"No entity provided, using default entity: {entity}")
            
        logger.info(f"Creating W&B report for project {project_name} with entity {entity} and run_id {run_id}")
        
        # Create the report
        report = wr.Report(
            project=project_name,
            entity=entity,
            title=report_title, 
            description=report_description
        )
        
        # Define the run set for data visualization
        # Note: Class name is Runset (lowercase 's')
        runset = wr.Runset(
            project=project_name,
            entity=entity,
        )
        
        # Define panels for our report
        panels = [
            # Training loss panel
            wr.LinePlot(
                title="Training Loss",
                x="batch",
                y=["train/batch_loss"],
            ),
            
            # Similarity metrics panel
            wr.LinePlot(
                title="Query-Document Similarities",
                x="batch",
                y=["train/pos_similarity", "train/neg_similarity"],
            ),
            
            # Similarity difference panel
            wr.LinePlot(
                title="Similarity Gap",
                x="batch",
                y=["train/similarity_diff"],
            ),
        ]
        
        # Create a panel grid containing our panels and runset
        panel_grid = wr.PanelGrid(
            panels=panels,
            runsets=[runset]
        )
        
        # Create report structure with markdown and panel grid
        report.blocks = [
            # Title and introduction
            wr.H1(text="Two-Tower Model Report"),
            wr.MarkdownBlock(text="This report shows the training metrics from the two-tower retrieval model."),
            panel_grid,
        ]
        
        # Save the report
        try:
            logger.info(f"Attempting to save report to W&B project: {project_name}, entity: {entity}")
            report.save()
            logger.info(f"Report created and saved successfully: {report.url}")
            
            return report.url
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            logger.error(f"Report details - project: {project_name}, entity: {entity}, run_id: {run_id}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    except Exception as e:
        logger.error(f"Error creating report: {str(e)}")
        logger.error(f"Report details - project: {project_name}, entity: {entity}, run_id: {run_id}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

# Add a basic configure_logging function for when this script is run directly
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    global logger
    logger = logging.getLogger('two_tower_report')

if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Create a W&B report for two-tower model experiments")
    parser.add_argument("--project", type=str, default=WANDB_PROJECT, help="W&B project name")
    parser.add_argument("--entity", type=str, default=WANDB_ENTITY, help="W&B username or team name")
    parser.add_argument("--title", type=str, default=None, help="Custom title for the report")
    parser.add_argument("--description", type=str, default=None, help="Description for the report")
    parser.add_argument("--run-id", type=str, default=None, help="W&B run ID to focus on in the report")
    
    args = parser.parse_args()
    
    # Login to W&B if running directly
    if not wandb.api.api_key:
        logger.info("No W&B API key found. Please log in.")
        wandb.login()
    
    logger.info(f"Creating report for project {args.project}")
    report_url = create_two_tower_report(
        project_name=args.project,
        entity=args.entity,
        title=args.title,
        description=args.description,
        run_id=args.run_id
    )
    
    if report_url:
        print("\nTo view your report, visit:")
        print(report_url)
        print("\nYou can also find it in the Reports tab of your W&B project.")
    else:
        print("\nFailed to create report. Check the logs for details.") 