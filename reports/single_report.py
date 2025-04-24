"""
Single run report generator for Two-Tower models.

This module provides functionality to create detailed W&B reports
for individual Two-Tower model runs.
"""

import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import wandb
import wandb_workspaces.reports.v2 as wr

from .report_utils import (
    find_experiment_files, 
    resolve_entity, 
    resolve_run_id, 
    load_genealogy,
    format_genealogy_markdown,
    create_mermaid_flowchart
)
from .blocks import (
    toc,
    intro_two_tower,
    training_dynamics_panels,
    similarity_panels,
    performance_panels,
    gradient_panels,
    config_panels,
    training_config_panels,
    genealogy_panel,
    mermaid_flowchart_panel,
    dataset_analysis_panels,
    media_browser_panel,
    applications_next_steps
)

# Get logger for this module
logger = logging.getLogger('two_tower.single_report')

def create_two_tower_report(
    project_name: str,
    entity: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    run_id: Optional[str] = None
) -> Optional[str]:
    """
    Create a W&B report for the two-tower model results
    
    Args:
        project_name: W&B project name
        entity: W&B entity (username or team)
        title: Report title
        description: Report description
        run_id: W&B run ID to create report for
        
    Returns:
        URL to the created report or None if failed
    """
    try:
        # Resolve entity with fallbacks
        entity = resolve_entity(entity)
        
        # Resolve run_id with fallbacks
        run_id = resolve_run_id(project_name, entity, run_id)
        if not run_id:
            logger.error("Could not resolve run ID")
            return None
        
        # Log diagnostic information 
        logger.info(f"Creating report for run: {run_id}")
        logger.info(f"W&B config - project: {project_name}, entity: {entity}")
        
        # Find experiment files
        experiment_files = find_experiment_files(run_id)
        if not experiment_files:
            logger.warning(f"No experiment files found for run {run_id}")
        else:
            logger.info(f"Found {len(experiment_files)} experiment files")
        
        # Get run details from W&B API
        try:
            api = wandb.Api()
            run = api.run(f"{entity}/{project_name}/{run_id}")
            logger.info(f"Retrieved run from API: {run.name}")
        except Exception as e:
            logger.error(f"Error retrieving run from W&B API: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
        # Set default title and description if not provided
        if title is None:
            title = f"Two-Tower Model Analysis: {run.name}"
        if description is None:
            description = f"Performance analysis for two-tower model run {run.name} ({run_id})"
        
        # Try to load dataset genealogy for this run if available
        dataset_genealogy = None
        if experiment_files:
            dataset_genealogy = load_genealogy(experiment_files[0])
        
        # Create the report
        report = wr.Report(
            project=project_name,
            entity=entity,
            title=title, 
            description=description,
            width='fluid'  # Make the report full width for better visualization
        )
        
        # Define the run set for data visualization
        runset = wr.Runset(
            project=project_name,
            entity=entity,
            query=f"id={run_id}" if run_id else "",
        )
        
        # Build report blocks using the reusable components
        report.blocks = [
            # Table of contents for better navigation
            toc(),
            
            # Introduction section with two-tower model explanation
            *intro_two_tower(),
            
            # Add metadata about the experiment
            wr.H1(text="🧪 Experiment Details"),
            wr.MarkdownBlock(text=(
                "## Experiment Configuration\n\n"
                "This section provides detailed information about the experiment configuration, dataset preparation, "
                "and other metadata to ensure reproducibility and clear documentation of the experimental setup.\n\n"
            )),
        ]
        
        # Add dataset genealogy panel if available
        if dataset_genealogy:
            genealogy_markdown = format_genealogy_markdown(dataset_genealogy)
            report.blocks.append(genealogy_panel(genealogy_markdown))
        
        # Section: Training Dynamics
        report.blocks.extend([
            wr.H1(text="🔄 Training Dynamics"),
            wr.MarkdownBlock(text=(
                "## Training Loss & Learning Progress\n\n"
                "This section tracks how well the model is learning over time. The loss metrics help us understand if the model is properly "
                "distinguishing between relevant and irrelevant query-document pairs.\n\n"
                "* **Batch Loss**: Shows the training loss for each batch of data. Lower values indicate better learning.\n"
                "* **Epoch Loss**: Shows the average loss for each complete pass through the training data.\n\n"
                "**What to watch for**:\n"
                "* Steadily decreasing loss indicates good learning progress\n"
                "* Plateaus suggest the model may need learning rate adjustments\n"
                "* Spikes may indicate problematic batches or instability in training"
            )),
            training_dynamics_panels(runset),
        ])
        
        # Section: Similarity Analysis
        report.blocks.extend([
            wr.H1(text="🧲 Similarity Analysis"),
            wr.MarkdownBlock(text=(
                "## Query-Document Similarity Metrics\n\n"
                "These metrics show how effectively the model is learning to distinguish between relevant (positive) and irrelevant (negative) query-document pairs.\n\n"
                "* **Positive Similarity**: Cosine similarity between query and relevant document embeddings. Higher values are better.\n"
                "* **Negative Similarity**: Cosine similarity between query and irrelevant document embeddings. Lower values are better.\n"
                "* **Similarity Gap**: The difference between positive and negative similarities. A larger gap indicates better discrimination.\n\n"
                "**Interpreting the charts**:\n"
                "* **Ideally**: Positive similarity should increase over time (approaching 1.0)\n"
                "* **Ideally**: Negative similarity should decrease over time (approaching 0.0)\n"
                "* **Similarity Gap**: Should widen over time, indicating better separation of relevant and irrelevant pairs\n"
                "* **Scatter Plot**: Points should move toward the top-left corner as training progresses (high positive similarity, low negative similarity)"
            )),
            similarity_panels(runset),
        ])
        
        # Section: Performance Analysis
        report.blocks.extend([
            wr.H1(text="⚡ Performance Analysis"),
            wr.MarkdownBlock(text=(
                "## Training Performance Metrics\n\n"
                "This section shows how efficiently the model is training in terms of computational resources and processing speed.\n\n"
                "* **Batch Processing Time**: Time taken to process each batch of data\n"
                "* **Forward/Backward Time**: Breakdown of time spent in forward pass (prediction) vs. backward pass (gradient computation)\n"
                "* **Samples Per Second**: Training throughput in terms of examples processed per second\n"
                "* **Gradient Norm**: The L2 norm of all gradients, indicating gradient magnitude during training\n\n"
                "**Why these metrics matter**:\n"
                "* **Processing Time**: Helps identify bottlenecks in training\n"
                "* **Forward/Backward Split**: Unusual ratios may indicate inefficiencies in model design\n"
                "* **Samples Per Second**: Higher is better for training efficiency\n"
                "* **Gradient Norm**: Very large or very small values may indicate training instability or vanishing/exploding gradients"
            )),
            performance_panels(runset),
            gradient_panels(runset),
        ])
        
        # Section: Architecture & Configuration Details
        report.blocks.extend([
            wr.H1(text="🔧 Model Architecture & Configuration"),
            wr.MarkdownBlock(text=(
                "## Model Architecture and Configuration Details\n\n"
                "This section provides a detailed view of the model architecture, hyperparameters, "
                "and configuration settings used in this experiment. These details are crucial for "
                "understanding the experiment setup and for reproducibility.\n\n"
                "The two-tower model consists of several key components:\n\n"
                "1. **Tokenizer**: Converts text into numeric sequences\n"
                "2. **Embedding Layer**: Translates token IDs into dense vector representations\n"
                "3. **Encoder Towers**: Process embeddings to create semantic representations\n"
                "4. **Loss Function**: Trains the model to distinguish relevant from irrelevant pairs\n\n"
                "The report below shows the specific configuration used for each component."
            )),
            config_panels(runset),
        ])
        
        # Section: Training Configuration
        report.blocks.extend([
            wr.H1(text="⚙️ Training Configuration"),
            wr.MarkdownBlock(text=(
                "## Training Hyperparameters\n\n"
                "This section shows the learning rate, batch size, and other training parameters used in this experiment. "
                "These hyperparameters can significantly impact model performance and training efficiency."
            )),
            training_config_panels(runset),
        ])
        
        # Dataset Analysis section if genealogy is available
        if dataset_genealogy:
            report.blocks.extend([
                wr.H1(text="📊 Dataset Analysis"),
                wr.MarkdownBlock(text=(
                    "## Dataset Characteristics and Preparation\n\n"
                    "This section provides insights into the dataset used for training, including information about "
                    "its size, composition, and the preprocessing steps applied to prepare it for training.\n\n"
                    "Understanding the dataset is crucial for interpreting model performance and ensuring that "
                    "the training data is representative of the target distribution."
                )),
            ])
            
            # Create dataset summary text
            dataset_text = (
                f"**Dataset Source**: MS MARCO ({dataset_genealogy.get('ms_marco_split', 'N/A')} split)\n\n"
                f"**Sampling Rate**: {dataset_genealogy.get('sample_size', 'All')} samples\n\n"
                f"**Triplets Format**: {dataset_genealogy.get('triplets_info', {}).get('row_count', 'N/A')} query-positive-negative triplets\n\n"
                f"**Random Seed**: {dataset_genealogy.get('random_seed', 'N/A')}"
            )
            
            # Create dataset flowchart
            dataset_flowchart = create_mermaid_flowchart(dataset_genealogy)
            
            # Add the panels
            report.blocks.append(dataset_analysis_panels(dataset_text, dataset_flowchart))
        
        # Examples section
        report.blocks.extend([
            wr.H1(text="📝 Examples & Retrieval Results"),
            wr.MarkdownBlock(text=(
                "## Exploring Model Inputs & Outputs\n\n"
                "This section provides concrete examples of the queries and documents being processed by the model, "
                "along with retrieval results when available.\n\n"
                "* **Queries**: Example input queries used during training/testing\n"
                "* **Positive Documents**: Relevant documents that should be retrieved for corresponding queries\n"
                "* **Negative Documents**: Irrelevant documents that should not be retrieved\n"
                "* **Retrieval Results**: For test queries, the top-K retrieved documents and their similarity scores"
            )),
            media_browser_panel(runset),
        ])
        
        # Applications and next steps
        report.blocks.extend(applications_next_steps())
        
        # Add experiment lineage if multiple related experiments
        if len(experiment_files) > 1:
            report.blocks.extend([
                wr.H1(text="⏱️ Experiment Lineage"),
                wr.MarkdownBlock(text=(
                    "## Experiment History\n\n"
                    "This section shows the progression of experiments conducted as part of this research. "
                    "Understanding the sequence and relationship between experiments can provide valuable insights "
                    "into the development process and the reasoning behind certain configuration choices."
                )),
            ])
            
            # Create a timeline of experiments in Mermaid.js format
            # (This could be moved to a helper function in report_utils)
            # For now, we'll keep it simple
            timeline_markdown = "```mermaid\ntimeline\n    title Experiment Timeline\n"
            timeline_markdown += "    section Experiments\n    ✅ Current Run : Active\n```"
            
            report.blocks.append(mermaid_flowchart_panel(timeline_markdown))
        
        # Save the report
        try:
            logger.info(f"Attempting to save report to W&B project: {project_name}, entity: {entity}")
            url = report.save()
            logger.info(f"Report created and saved successfully: {url}")
            
            return url
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            logger.error(f"Report details - project: {project_name}, entity: {entity}, run_id: {run_id}")
            logger.error(traceback.format_exc())
            return None
    
    except Exception as e:
        logger.error(f"Error creating report: {str(e)}")
        logger.error(f"Report details - project: {project_name}, entity: {entity}, run_id: {run_id}")
        logger.error(traceback.format_exc())
        return None 