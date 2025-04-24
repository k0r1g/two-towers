"""
Comparison report generator for Two-Tower models.

This module provides functionality to create W&B reports that compare
multiple Two-Tower model runs.
"""

import logging
import traceback
from typing import List, Optional

import wandb
import wandb_workspaces.reports.v2 as wr

from .report_utils import resolve_entity
from .blocks import (
    toc, 
    training_dynamics_panels, 
    similarity_panels,
    performance_panels,
    gradient_panels,
    training_config_panels,
    comparison_config_panel
)

# Get logger for this module
logger = logging.getLogger('two_tower.compare_report')

def create_comparison_report(
    project_name: str,
    entity: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    run_ids: Optional[List[str]] = None
) -> Optional[str]:
    """
    Create a W&B report comparing multiple two-tower model runs
    
    Args:
        project_name: W&B project name
        entity: W&B entity (username or team)
        title: Report title
        description: Report description
        run_ids: List of W&B run IDs to compare. If None, will try to find recent runs.
        
    Returns:
        URL to the created report or None if failed
    """
    try:
        # Resolve entity
        entity = resolve_entity(entity)
        
        # Handle missing run_ids
        if run_ids is None or len(run_ids) == 0:
            logger.warning("No run_ids provided for comparison")
            
            # Try to get the most recent runs from the project
            logger.info("Attempting to find recent runs from API")
            try:
                api = wandb.Api()
                runs = api.runs(f"{entity}/{project_name}", per_page=5)
                if len(runs) > 0:
                    run_ids = [run.id for run in runs]
                    logger.info(f"Using {len(run_ids)} most recent runs for comparison")
                else:
                    logger.error(f"No runs found in project {entity}/{project_name}")
                    return None
            except Exception as e:
                logger.error(f"Error fetching runs from API: {str(e)}")
                logger.error(traceback.format_exc())
                return None
                
        # Validate we have at least 2 runs to compare
        if len(run_ids) < 2:
            logger.error(f"Need at least 2 runs for comparison, only found {len(run_ids)}")
            return None
        
        # Log diagnostic information
        logger.info(f"Creating comparison report for runs: {run_ids}")
        logger.info(f"W&B config - project: {project_name}, entity: {entity}")
        
        # Get run details from W&B API
        try:
            api = wandb.Api()
            runs = []
            for run_id in run_ids:
                run = api.run(f"{entity}/{project_name}/{run_id}")
                runs.append(run)
            logger.info(f"Retrieved {len(runs)} runs from API")
            
            # Set default title and description if not provided
            if title is None:
                title = f"Two-Tower Model Comparison: {len(runs)} Runs"
            if description is None:
                description = f"Comparative analysis of {len(runs)} different two-tower model runs"
                
        except Exception as e:
            logger.error(f"Error retrieving runs from W&B API: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
        # Create the report
        logger.info("Creating W&B comparison report...")
        
        report = wr.Report(
            project=project_name,
            entity=entity,
            title=title,
            description=description,
            width='fluid'  # Make the report full width for better visualization
        )
        
        # Create runset for runs
        try:
            query = " OR ".join([f"id={run_id}" for run_id in run_ids])
            runset = wr.Runset(
                entity=entity,
                project=project_name,
                name="All Compared Runs",
                query=query,
                order=[wr.OrderBy(name='CreatedTimestamp', ascending=False)]
            )
            logger.info(f"Created runset with {len(run_ids)} runs")
        except Exception as e:
            logger.error(f"Error creating runset: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
        # Build the report structure
        try:
            # Create the report structure
            report.blocks = [
                # Table of contents
                toc(),
                
                # Introduction
                wr.H1("Two-Tower Model Comparison"),
                wr.P(text="""
                This report compares multiple runs of the two-tower model, analyzing their performance,
                training dynamics, and embedding characteristics to identify the most effective configurations.
                """),
                
                # Run Information
                wr.H2("Run Information"),
                wr.P(text="Detailed information about the run configuration and parameters:"),
                comparison_config_panel(runset),
                
                # Performance Metrics section
                wr.H2("Performance Metrics"),
                wr.P(text="Comparison of key metrics across runs to identify performance patterns."),
                wr.PanelGrid(
                    runsets=[runset],
                    panels=[
                        wr.LinePlot(
                            title="Training Loss",
                            x="Step",
                            y=["train/batch_loss", "train/epoch_loss"], 
                            smoothing_factor=0.2,
                            layout=wr.Layout(x=0, y=0, w=8, h=6)
                        ),
                        wr.LinePlot(
                            title="Positive & Negative Similarity",
                            x="Step",
                            y=["train/pos_similarity", "train/neg_similarity"], 
                            smoothing_factor=0.2,
                            layout=wr.Layout(x=8, y=0, w=8, h=6)
                        ),
                        wr.LinePlot(
                            title="Similarity Gap",
                            x="Step",
                            y=["train/similarity_diff"], 
                            smoothing_factor=0.2,
                            layout=wr.Layout(x=16, y=0, w=8, h=6)
                        ),
                        wr.LinePlot(
                            title="Performance Metrics",
                            x="Step",
                            y=["performance/samples_per_second"],
                            smoothing_factor=0.2,
                            layout=wr.Layout(x=0, y=6, w=8, h=6)
                        ),
                    ]
                ),
                
                # Training Configuration section
                wr.H2("Training Configuration"),
                wr.P(text="Examining differences in learning rate, batch size, and other training parameters."),
                wr.PanelGrid(
                    runsets=[runset],
                    panels=[
                        wr.LinePlot(
                            title="Learning Rate",
                            x="Step",
                            y=["train/learning_rate"],
                            smoothing_factor=0.2,
                            layout=wr.Layout(x=0, y=0, w=8, h=6)
                        ),
                        wr.ScalarChart(
                            title="Batch Size",
                            metric="train/batch_size",
                            layout=wr.Layout(x=8, y=0, w=8, h=6)
                        ),
                        wr.LinePlot(
                            title="Gradient Norm",
                            x="Step",
                            y=["gradients/total_norm", "train/grad_norm"],
                            smoothing_factor=0.2,
                            layout=wr.Layout(x=16, y=0, w=8, h=6)
                        )
                    ]
                ),
                
                # Training performance section
                wr.H2("Training Performance"),
                wr.P(text="Analyzing processing times and training efficiency."),
                wr.PanelGrid(
                    runsets=[runset],
                    panels=[
                        wr.LinePlot(
                            title="Processing Time Breakdown",
                            x="Step",
                            y=["performance/batch_time", "performance/forward_time", "performance/backward_time"],
                            smoothing_factor=0.2,
                            layout=wr.Layout(x=0, y=0, w=8, h=6)
                        ),
                        wr.LinePlot(
                            title="Processing Efficiency",
                            x="Step",
                            y=["performance/samples_per_second"],
                            smoothing_factor=0.2,
                            layout=wr.Layout(x=8, y=0, w=8, h=6)
                        )
                    ]
                ),
                
                # Conclusion
                wr.H2("Conclusion"),
                wr.P(text="This report was automatically generated to compare the performance of different runs of the two-tower model.")
            ]
            
            # Save the report
            logger.info("Saving comparison report...")
            url = report.save()
            logger.info(f"Comparison report saved successfully at: {url}")
            
            return url
        
        except Exception as e:
            logger.error(f"Error creating or saving comparison report: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    except Exception as e:
        logger.error(f"Unexpected error in create_comparison_report: {str(e)}")
        logger.error(traceback.format_exc())
        return None 