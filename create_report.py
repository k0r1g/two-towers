#!/usr/bin/env python
"""
Create W&B reports for Two-Tower model runs.

This script helps generate detailed visualization reports for Two-Tower model training runs,
providing insights into model performance, training dynamics, and dataset properties.

Usage:
    # Create a report for a single run
    python create_report.py single --run-id RUN_ID 
    
    # Create a comparison report for multiple runs
    python create_report.py compare --run-ids RUN_ID1 RUN_ID2
    
    # Create a comparison report for 5 most recent runs
    python create_report.py compare
    
Requirements:
    pip install wandb wandb-workspaces
"""

import os
import argparse
import wandb
import wandb_workspaces.reports.v2 as wr
import logging
import json
import glob
from pathlib import Path
import datetime
import yaml
from typing import List, Optional
import sys
import traceback

# Import config settings
from config import WANDB_PROJECT, WANDB_ENTITY

# Make sure you have the latest wandb_workspaces package:
# pip install --upgrade wandb wandb-workspaces

# Get the logger from the main script
logger = logging.getLogger('two_tower')

def find_experiment_files(run_id=None, number=5):
    """
    Find experiment files in the workspace. 
    If run_id is provided, will look for files related to that specific run.
    Otherwise returns the most recent experiment files.
    
    Args:
        run_id (str, optional): W&B run ID to filter files
        number (int, optional): Number of files to return if run_id not specified
        
    Returns:
        list: List of experiment file paths sorted by recency
    """
    try:
        # Get all experiment files
        experiment_files = glob.glob("experiments/*.json")
        
        # Sort by modification time (most recent first)
        experiment_files.sort(key=os.path.getmtime, reverse=True)
        
        if not experiment_files:
            logger.warning("No experiment files found in the experiments/ directory")
            return []
            
        if run_id:
            # Filter files that contain the run_id
            matching_files = []
            for file_path in experiment_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
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
            logger.info(f"Returning {min(number, len(experiment_files))} most recent experiment files")
            return experiment_files[:number]
            
    except Exception as e:
        logger.error(f"Error finding experiment files: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def create_two_tower_report(project_name=None, entity=None, title=None, description=None, run_id=None):
    """
    Create a W&B report for the two-tower model results
    
    Args:
        project_name (str, optional): W&B project name. Defaults to config value.
        entity (str, optional): W&B entity (username or team). Defaults to config value.
        title (str, optional): Report title. Defaults to generated title.
        description (str, optional): Report description. Defaults to generated description.
        run_id (str, optional): W&B run ID to create report for. If None, will try to use current run.
        
    Returns:
        str: URL to the created report or None if failed
    """
    try:
        # Set default project and entity if not provided
        project_name = project_name or WANDB_PROJECT
        
        # Try to get entity with fallbacks
        if entity is None:
            try:
                if wandb.run is not None:
                    entity = wandb.run.entity
                    logger.info(f"Using entity from current wandb.run: {entity}")
                else:
                    # Try to get default entity from API
                    api = wandb.Api()
                    entity = api.default_entity
                    logger.info(f"Using default entity from wandb.Api(): {entity}")
            except Exception as e:
                logger.warning(f"Error getting default entity: {str(e)}")
                entity = WANDB_ENTITY
                logger.info(f"Falling back to config entity: {entity}")
        
        # Handle missing run_id
        if run_id is None:
            logger.warning("No run_id provided")
            
            # Try to use current run
            if wandb.run is not None:
                run_id = wandb.run.id
                logger.info(f"Using current wandb.run id: {run_id}")
            else:
                # Try to get the most recent run from the project
                logger.info("Attempting to find most recent run from API")
                try:
                    api = wandb.Api()
                    runs = api.runs(f"{entity}/{project_name}", per_page=1)
                    if len(runs) > 0:
                        run_id = runs[0].id
                        logger.info(f"Using most recent run from project: {run_id}")
                    else:
                        logger.error(f"No runs found in project {entity}/{project_name}")
                        return None
                except Exception as e:
                    logger.error(f"Error fetching runs from API: {str(e)}")
                    logger.error(traceback.format_exc())
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
            try:
                with open(experiment_files[0], 'r') as f:
                    dataset_genealogy = json.load(f)
                logger.info(f"Loaded dataset genealogy from {experiment_files[0]}")
            except Exception as e:
                logger.warning(f"Error loading dataset genealogy: {str(e)}")
        
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
            # Add filter for the specific run if provided
            query=f"id={run_id}" if run_id else "",
        )
        
        # Add table of contents for better navigation
        report.blocks = [
            wr.TableOfContents(),
        ]
        
        # Introduction section
        report.blocks.extend([
            wr.H1(text="📊 Two-Tower Model Overview"),
            wr.MarkdownBlock(text=(
                "## What is a Two-Tower Model?\n\n"
                "A two-tower model (also known as a dual encoder) is a neural network architecture designed for retrieval tasks. "
                "It consists of two separate encoder networks:\n\n"
                "* **Query Tower**: Encodes search queries into dense vector representations\n"
                "* **Document Tower**: Encodes documents/items into the same vector space\n\n"
                "During training, the model learns to place semantically related query-document pairs close together in the embedding space, "
                "while keeping unrelated pairs further apart. This architecture is particularly effective for:\n\n"
                "* **Retrieval tasks** - Finding relevant documents for a given query\n"
                "* **Recommendation systems** - Matching user queries with relevant items\n"
                "* **Information retrieval** - Semantic search across large document collections\n\n"
                "The key advantage of this architecture is that once trained, documents can be pre-encoded offline, "
                "allowing for efficient retrieval at inference time by comparing a query embedding against a database "
                "of document embeddings."
            )),
            wr.Image(
                url="https://miro.medium.com/max/1400/1*BYO9Q7NpLG2RSPv_hqq9SA.png",
                caption="Two-tower architecture: separate encoders for queries and documents projecting into a shared embedding space"
            ),
        ])
        
        # Add Experiment Metadata section
        report.blocks.extend([
            wr.H1(text="🧪 Experiment Details"),
            wr.MarkdownBlock(text=(
                "## Experiment Configuration\n\n"
                "This section provides detailed information about the experiment configuration, dataset preparation, "
                "and other metadata to ensure reproducibility and clear documentation of the experimental setup.\n\n"
            )),
        ])
        
        # Add dataset genealogy information if available
        if dataset_genealogy:
            # Format dataset genealogy info
            preset_info = ""
            if "preset_config" in dataset_genealogy:
                preset_info = "\n\n**Preset Configuration:**\n```json\n"
                preset_info += json.dumps(dataset_genealogy["preset_config"], indent=2)
                preset_info += "\n```"
            
            # Format preprocessing steps
            preprocessing_steps = ""
            if "preprocessing_steps" in dataset_genealogy:
                preprocessing_steps = "\n\n**Preprocessing Steps:**\n"
                for i, step in enumerate(dataset_genealogy["preprocessing_steps"]):
                    step_time = step.get("timestamp", "N/A").split("T")[1].split(".")[0] if "timestamp" in step else "N/A"
                    preprocessing_steps += f"{i+1}. **{step.get('step', 'unknown')}** ({step_time})\n"
                    for k, v in step.items():
                        if k not in ["step", "timestamp"]:
                            preprocessing_steps += f"   - {k}: {v}\n"
            
            # Format dataset info
            dataset_info = ""
            if "triplets_info" in dataset_genealogy:
                dataset_info += "\n\n**Triplets Dataset:**\n"
                for k, v in dataset_genealogy["triplets_info"].items():
                    dataset_info += f"- {k}: {v}\n"
            
            if "sampled_dataset_info" in dataset_genealogy:
                dataset_info += "\n\n**Sampled Dataset:**\n"
                for k, v in dataset_genealogy["sampled_dataset_info"].items():
                    dataset_info += f"- {k}: {v}\n"
            
            # Add dataset genealogy panel
            report.blocks.extend([
                wr.PanelGrid(
                    panels=[
                        wr.MarkdownPanel(
                            markdown=(
                                f"**Experiment ID:** {dataset_genealogy.get('experiment_id', 'N/A')}\n\n"
                                f"**MS MARCO Split:** {dataset_genealogy.get('ms_marco_split', 'N/A')}\n\n"
                                f"**Preset File:** {dataset_genealogy.get('preset_file', 'N/A')}\n\n"
                                f"**Random Seed:** {dataset_genealogy.get('random_seed', 'N/A')}\n\n"
                                f"**Sample Size:** {dataset_genealogy.get('sample_size', 'All')} samples"
                                f"{preset_info}"
                                f"{preprocessing_steps}"
                                f"{dataset_info}"
                            ),
                            layout=wr.Layout(w=24, h=15)
                        ),
                    ]
                ),
            ])
        
        # Section 1: Training Dynamics
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
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    # Primary training metrics
                    wr.LinePlot(
                        title="Training Loss (by Batch)",
                        x="train/batch",
                        y=["train/batch_loss"],
                        smoothing_factor=0.8,
                        layout=wr.Layout(w=12, h=8)
                    ),
                    wr.LinePlot(
                        title="Training Loss (by Epoch)",
                        x="epoch",
                        y=["train/epoch_loss"],
                        layout=wr.Layout(w=12, h=8)
                    ),
                ]
            ),
        ])
        
        # Section 2: Similarity Analysis
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
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    # Similarity metrics
                    wr.LinePlot(
                        title="Query-Document Similarity Trends",
                        x="train/batch",
                        y=["train/pos_similarity", "train/neg_similarity"],
                        smoothing_factor=0.8,
                        layout=wr.Layout(w=12, h=8)
                    ),
                    wr.LinePlot(
                        title="Similarity Gap (Pos - Neg)",
                        x="train/batch",
                        y=["train/similarity_diff"],
                        smoothing_factor=0.8,
                        layout=wr.Layout(w=12, h=8)
                    ),
                    # Distribution of similarities
                    wr.ScatterPlot(
                        title="Positive vs Negative Similarity Distribution",
                        x="train/neg_similarity",
                        y="train/pos_similarity",
                        layout=wr.Layout(w=12, h=8)
                    ),
                ]
            ),
        ])
        
        # Section 3: Performance Analysis
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
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    # Performance metrics
                    wr.LinePlot(
                        title="Batch Processing Time",
                        x="train/batch",
                        y=["performance/batch_time"],
                        smoothing_factor=0.5,
                        layout=wr.Layout(w=8, h=6)
                    ),
                    wr.LinePlot(
                        title="Forward/Backward Time Breakdown",
                        x="train/batch",
                        y=["performance/forward_time", "performance/backward_time"],
                        smoothing_factor=0.5,
                        layout=wr.Layout(w=8, h=6)
                    ),
                    wr.LinePlot(
                        title="Training Throughput (Samples/Second)",
                        x="train/batch",
                        y=["performance/samples_per_second"],
                        smoothing_factor=0.5,
                        layout=wr.Layout(w=8, h=6)
                    ),
                    # Gradient analysis
                    wr.LinePlot(
                        title="Gradient Norm (Model Health)",
                        x="train/batch",
                        y=["gradients/total_norm"],
                        smoothing_factor=0.5,
                        layout=wr.Layout(w=12, h=6)
                    ),
                ]
            ),
        ])
        
        # Section 4: Architecture & Configuration Details
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
        ])
        
        # Add config panels from W&B run
        report.blocks.extend([
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    # Config details
                    wr.RunComparer(
                        diff_only='split',
                        layout=wr.Layout(w=24, h=15)
                    ),
                ]
            ),
        ])
        
        # Section 5: Run Comparison and Parameter Analysis
        report.blocks.extend([
            wr.H1(text="🔬 Hyperparameter Analysis"),
            wr.MarkdownBlock(text=(
                "## Comparing Runs & Hyperparameters\n\n"
                "This section helps identify the impact of different hyperparameters on model performance. It's especially useful when comparing multiple training runs.\n\n"
                "* **Run Comparer**: Directly compare configuration and metrics across different runs\n"
                "* **Parallel Coordinates Plot**: Visualize relationships between hyperparameters and outcomes\n"
                "* **Parameter Importance**: Identify which hyperparameters have the most impact on model performance\n\n"
                "**Key hyperparameters for two-tower models**:\n"
                "* **Learning Rate**: Controls step size during optimization (typically 1e-3 to 1e-5)\n"
                "* **Batch Size**: Number of examples processed simultaneously (impacts training dynamics)\n"
                "* **Embedding Dimension**: Size of the shared embedding space (higher = more capacity but slower)\n"
                "* **Hidden Dimension**: Size of internal representations in the towers\n\n"
                "**What to look for**:\n"
                "* Patterns between hyperparameters and final loss/similarity gap\n"
                "* Which parameters have the most impact on performance\n"
                "* Promising hyperparameter combinations for future runs"
            )),
            wr.PanelGrid(
                runsets=[
                    # Use a broader runset for comparison that includes multiple runs
                    wr.Runset(
                        project=project_name,
                        entity=entity,
                    ),
                ],
                panels=[
                    # Run comparison
                    wr.RunComparer(
                        diff_only=True,
                        layout=wr.Layout(w=24, h=10)
                    ),
                    # Parameter analysis
                    wr.ParallelCoordinatesPlot(
                        columns=[
                            wr.ParallelCoordinatesPlotColumn(metric="c::optimizer.lr"),
                            wr.ParallelCoordinatesPlotColumn(metric="c::batch_size"),
                            wr.ParallelCoordinatesPlotColumn(metric="c::epochs"),
                            wr.ParallelCoordinatesPlotColumn(metric="c::embedding.embedding_dim"),
                            wr.ParallelCoordinatesPlotColumn(metric="c::encoder.hidden_dim"),
                            wr.ParallelCoordinatesPlotColumn(metric="train/epoch_loss"),
                            wr.ParallelCoordinatesPlotColumn(metric="train/similarity_diff"),
                        ],
                        layout=wr.Layout(w=24, h=8)
                    ),
                    # Parameter importance
                    # Commenting out this section as the API has changed
                    # wr.ParameterImportancePlot(
                    #     with_respect_to="train/epoch_loss",
                    #     parameters=["c::optimizer.lr", "c::batch_size", "c::epochs", "c::embedding.embedding_dim", "c::encoder.hidden_dim"],
                    #     layout=wr.Layout(w=12, h=8)
                    # ),
                ]
            ),
        ])
        
        # Section 6: Dataset Analysis
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
                wr.PanelGrid(
                    panels=[
                        # Dataset statistics and genealogy
                        wr.MarkdownPanel(
                            markdown=(
                                f"**Dataset Source**: MS MARCO ({dataset_genealogy.get('ms_marco_split', 'N/A')} split)\n\n"
                                f"**Sampling Rate**: {dataset_genealogy.get('sample_size', 'All')} samples\n\n"
                                f"**Triplets Format**: {dataset_genealogy.get('triplets_info', {}).get('row_count', 'N/A')} query-positive-negative triplets\n\n"
                                f"**Random Seed**: {dataset_genealogy.get('random_seed', 'N/A')}"
                            ),
                            layout=wr.Layout(w=12, h=8)
                        ),
                        # Dataset preprocessing visualization
                        wr.MarkdownPanel(
                            markdown=(
                                "```mermaid\nflowchart TD\n" +
                                "A[MS MARCO Dataset] --> B[Convert to Parquet]\n" +
                                f"B --> C[Create Triplets Format\\n{dataset_genealogy.get('triplets_info', {}).get('row_count', '?')} triplets]\n" +
                                (f"C --> D[Sample Dataset\\n{dataset_genealogy.get('sample_size', 'All')} samples]" if dataset_genealogy.get('sample_size') else "C --> D[Use Full Dataset]") +
                                "\nD --> E[Training Data]\n```"
                            ),
                            layout=wr.Layout(w=12, h=8)
                        ),
                    ]
                ),
            ])
        
        # Section 7: Examples and Retrieval Results
        report.blocks.extend([
            wr.H1(text="📝 Examples & Retrieval Results"),
            wr.MarkdownBlock(text=(
                "## Exploring Model Inputs & Outputs\n\n"
                "This section provides concrete examples of the queries and documents being processed by the model, "
                "along with retrieval results when available.\n\n"
                "* **Queries**: Example input queries used during training/testing\n"
                "* **Positive Documents**: Relevant documents that should be retrieved for corresponding queries\n"
                "* **Negative Documents**: Irrelevant documents that should not be retrieved\n"
                "* **Retrieval Results**: For test queries, the top-K retrieved documents and their similarity scores\n\n"
                "**How to interpret retrieval results**:\n"
                "* **Similarity Score**: Higher values indicate stronger relevance (closer to 1.0 = more relevant)\n"
                "* **Ranking Quality**: Relevant documents should appear at the top of the list\n"
                "* **False Positives**: Irrelevant documents with high similarity scores indicate potential areas for improvement\n"
                "* **False Negatives**: Relevant documents with low similarity scores indicate potential model weaknesses"
            )),
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    # Tables and media browser
                    wr.MediaBrowser(
                        media_keys=["examples/query", "examples/positive_doc", "examples/negative_doc"],
                        layout=wr.Layout(w=24, h=10)
                    ),
                ]
            ),
        ])
        
        # Section 8: Practical Applications & Next Steps
        report.blocks.extend([
            wr.H1(text="🚀 Applications & Next Steps"),
            wr.MarkdownBlock(text=(
                "## Practical Applications\n\n"
                "Two-tower models can be deployed in various retrieval scenarios:\n\n"
                "* **Semantic Search**: Find documents based on meaning rather than exact keyword matching\n"
                "* **Recommendation Systems**: Match user queries or profiles with relevant items\n"
                "* **Information Retrieval**: Efficiently retrieve information from large document collections\n"
                "* **Question Answering**: Retrieve passages that might contain answers to questions\n\n"
                "## Potential Improvements\n\n"
                "Based on the observed metrics, consider these potential improvements:\n\n"
                "* **Data Quality**: Ensure training data contains diverse query-document pairs\n"
                "* **Negative Sampling**: Experiment with hard negative mining to improve discrimination\n"
                "* **Model Architecture**: Try different encoder architectures or pre-trained language models\n"
                "* **Loss Function**: Experiment with different contrastive or triplet loss variants\n"
                "* **Hyperparameters**: Optimize learning rate, batch size, and embedding dimensions\n\n"
                "## Deployment Considerations\n\n"
                "When deploying the trained model:\n\n"
                "* Pre-compute document embeddings offline for efficiency\n"
                "* Use approximate nearest neighbor search for large-scale retrieval\n"
                "* Consider quantization to reduce embedding size and memory footprint\n"
                "* Monitor retrieval quality on diverse queries in production"
            )),
        ])
        
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
            
            # Create a timeline of experiments
            try:
                # Load experiment summaries
                experiment_summaries = []
                for file_path in experiment_files:
                    try:
                        with open(file_path, 'r') as f:
                            summary = json.load(f)
                            experiment_summaries.append(summary)
                    except Exception as e:
                        logger.warning(f"Error loading experiment summary {file_path}: {str(e)}")
                
                # Sort by timestamp
                experiment_summaries.sort(key=lambda x: x.get("timestamp", ""))
                
                # Create timeline markdown
                timeline_markdown = "```mermaid\ntimeline\n"
                
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
                
                report.blocks.extend([
                    wr.PanelGrid(
                        panels=[
                            wr.MarkdownPanel(
                                markdown=timeline_markdown,
                                layout=wr.Layout(w=24, h=12)
                            ),
                        ]
                    ),
                ])
            except Exception as e:
                logger.warning(f"Error creating experiment timeline: {str(e)}")
        
        # PanelGrid for performance analysis
        wr.H2("Training Performance"),
        wr.P(text="Analyzing processing times and training efficiency."),
        wr.PanelGrid(
            runsets=[runset],
            panels=[
                wr.LinePlot(
                    title="Processing Time Breakdown",
                    y=["performance/batch_time", "performance/forward_time", "performance/backward_time"],
                    smoothing_factor=0.2
                ),
                wr.LinePlot(
                    title="Processing Efficiency",
                    y=["performance/samples_per_second"],
                    smoothing_factor=0.2
                )
            ]
        ),
        
        # Save the report
        try:
            logger.info(f"Attempting to save report to W&B project: {project_name}, entity: {entity}")
            report.save()
            logger.info(f"Report created and saved successfully: {report.url}")
            
            return report.url
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

def create_comparison_report(project_name=None, entity=None, title=None, description=None, run_ids=None):
    """
    Create a W&B report comparing multiple two-tower model runs
    
    Args:
        project_name (str, optional): W&B project name. Defaults to config value.
        entity (str, optional): W&B entity (username or team). Defaults to config value.
        title (str, optional): Report title. Defaults to generated title.
        description (str, optional): Report description. Defaults to generated description.
        run_ids (list, optional): List of W&B run IDs to compare. If None, will try to find recent runs.
        
    Returns:
        str: URL to the created report or None if failed
    """
    try:
        # Set default project and entity if not provided
        project_name = project_name or WANDB_PROJECT
        
        # Try to get entity with fallbacks
        if entity is None:
            try:
                if wandb.run is not None:
                    entity = wandb.run.entity
                    logger.info(f"Using entity from current wandb.run: {entity}")
                else:
                    # Try to get default entity from API
                    api = wandb.Api()
                    entity = api.default_entity
                    logger.info(f"Using default entity from wandb.Api(): {entity}")
            except Exception as e:
                logger.warning(f"Error getting default entity: {str(e)}")
                entity = WANDB_ENTITY
                logger.info(f"Falling back to config entity: {entity}")
        
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
        import wandb_workspaces.reports.v2 as wr
        
        report = wr.Report(
            project=project_name,
            entity=entity,
            title=title,
            description=description,
            width='fluid'  # Make the report full width for better visualization
        )
        
        # Create runsets for multiple runs
        runset_all = None
        try:
            if run_ids:
                # Create a runset directly with entity, project parameters
                # Use a query string to filter by run IDs
                query = " OR ".join([f"id={run_id}" for run_id in run_ids])
                runset_all = wr.Runset(
                    entity=entity,
                    project=project_name,
                    name="All Compared Runs",
                    query=query
                )
                logger.info(f"Created runset with {len(run_ids)} runs")
            else:
                logger.error("No run IDs provided, cannot create comparison report")
                return None
        except Exception as e:
            logger.error(f"Error creating runset: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
        # Build the report structure with error handling
        try:
            # Create the report structure
            report.blocks = [
                wr.TableOfContents(),  # Table of contents
                
                # Introduction section
                wr.H1("Two-Tower Model Comparison"),
                wr.P(text="""
                This report compares multiple runs of the two-tower model, analyzing their performance,
                training dynamics, and embedding characteristics to identify the most effective configurations.
                """),
                
                # Run Information
                wr.H2("Run Information"),
                wr.P(text="Detailed information about the run configuration and parameters:"),
                wr.PanelGrid(
                    runsets=[runset_all],
                    panels=[
                        wr.RunComparer(diff_only=True)
                    ]
                ),
                
                # Performance Metrics section
                wr.H2("Performance Metrics"),
                wr.P(text="Comparison of key metrics across runs to identify performance patterns."),
                
                # Use PanelGrid for comparing metrics
                wr.PanelGrid(
                    runsets=[runset_all],
                    panels=[
                        wr.LinePlot(
                            title="Training Loss",
                            y=["train/batch_loss", "train/epoch_loss"], 
                            smoothing_factor=0.2
                        ),
                        wr.LinePlot(
                            title="Positive & Negative Similarity",
                            y=["train/pos_similarity", "train/neg_similarity"], 
                            smoothing_factor=0.2
                        ),
                        wr.LinePlot(
                            title="Similarity Gap",
                            y=["train/similarity_diff"], 
                            smoothing_factor=0.2
                        ),
                        wr.LinePlot(
                            title="Performance Metrics",
                            y=["performance/samples_per_second"],
                            smoothing_factor=0.2
                        ),
                    ]
                ),
                
                # PanelGrid for analysis over time or batch
                wr.H2("Training Configuration"),
                wr.P(text="Examining differences in learning rate, batch size, and other training parameters."),
                wr.PanelGrid(
                    runsets=[runset_all],
                    panels=[
                        wr.LinePlot(
                            title="Learning Rate",
                            y=["train/learning_rate"],
                            smoothing_factor=0.2
                        ),
                        wr.ScalarChart(
                            title="Batch Size",
                            metric="train/batch_size"
                        ),
                        wr.LinePlot(
                            title="Gradient Norm",
                            y=["gradients/total_norm", "train/grad_norm"],
                            smoothing_factor=0.2
                        )
                    ]
                ),
                
                # PanelGrid for performance analysis
                wr.H2("Training Performance"),
                wr.P(text="Analyzing processing times and training efficiency."),
                wr.PanelGrid(
                    runsets=[runset_all],
                    panels=[
                        wr.LinePlot(
                            title="Processing Time Breakdown",
                            y=["performance/batch_time", "performance/forward_time", "performance/backward_time"],
                            smoothing_factor=0.2
                        ),
                        wr.LinePlot(
                            title="Processing Efficiency",
                            y=["performance/samples_per_second"],
                            smoothing_factor=0.2
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
            
            # Display instructions for accessing the report
            print(f"\nComparison report created and available at: {url}")
            print("This report compares the selected model runs and provides insights into performance differences.")
            
            return url
        
        except Exception as e:
            logger.error(f"Error creating or saving comparison report: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    except Exception as e:
        logger.error(f"Unexpected error in create_comparison_report: {str(e)}")
        logger.error(traceback.format_exc())
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

def main():
    """
    Main function to parse arguments and create reports.
    """
    parser = argparse.ArgumentParser(description="Create W&B reports for Two-Tower model experiments")
    
    # Common arguments
    parser.add_argument("--project", default="two-tower-retrieval", help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity (username or team name)")
    parser.add_argument("--title", default=None, help="Report title")
    parser.add_argument("--description", default=None, help="Report description")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single report command
    single_parser = subparsers.add_parser("single", help="Create a report for a single run")
    single_parser.add_argument("--run-id", required=True, help="W&B run ID for the report")
    
    # Comparison report command
    compare_parser = subparsers.add_parser("compare", help="Create a comparison report for multiple runs")
    compare_parser.add_argument("--run-ids", nargs="+", help="List of W&B run IDs to include in comparison")
    compare_parser.add_argument("--num-recent", type=int, default=5, help="Number of recent runs to compare (if run-ids not specified)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check for required command
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Ensure entity is set
        if not args.entity:
            try:
                import wandb
                if wandb.api.default_entity:
                    args.entity = wandb.api.default_entity
                    logger.info(f"Using default W&B entity: {args.entity}")
                else:
                    logger.warning("No W&B entity specified and no default found. Using 'None'")
            except:
                logger.warning("Could not determine default W&B entity")
        
        # Handle single report command
        if args.command == "single":
            if not args.run_id:
                logger.error("Run ID is required for single report")
                return
                
            logger.info(f"Creating report for run ID: {args.run_id}")
            url = create_two_tower_report(
                project_name=args.project,
                entity=args.entity,
                title=args.title,
                description=args.description,
                run_id=args.run_id
            )
            
            if url:
                print(f"Report created successfully at: {url}")
            else:
                print("Failed to create report. See logs for details.")
        
        # Handle comparison report command
        elif args.command == "compare":
            run_ids = args.run_ids
            
            # If no run IDs provided, use most recent runs
            if not run_ids:
                try:
                    import wandb
                    api = wandb.Api()
                    runs = list(api.runs(f"{args.entity}/{args.project}", order="-created_at", per_page=args.num_recent))
                    if runs:
                        run_ids = [run.id for run in runs]
                        logger.info(f"Using {len(run_ids)} most recent runs for comparison")
                    else:
                        logger.error(f"No runs found in project {args.project}")
                        return
                except Exception as e:
                    logger.error(f"Error fetching recent runs: {str(e)}")
                    logger.error(traceback.format_exc())
                    return
            
            logger.info(f"Creating comparison report for run IDs: {run_ids}")
            url = create_comparison_report(
                project_name=args.project,
                entity=args.entity,
                title=args.title,
                description=args.description,
                run_ids=run_ids
            )
            
            if url:
                print(f"Comparison report created successfully at: {url}")
            else:
                print("Failed to create comparison report. See logs for details.")
    
    except Exception as e:
        logger.error(f"Unexpected error in main function: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 