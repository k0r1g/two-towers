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
import json
import glob
from pathlib import Path
import datetime
import yaml

# Import config settings
from config import WANDB_PROJECT, WANDB_ENTITY

# Get the logger from the main script
logger = logging.getLogger('two_tower')

def find_experiment_files(run_id=None):
    """
    Find experiment files (configs, summaries, dataset genealogy) related to a run.
    
    Args:
        run_id: Optional W&B run ID to search for
    
    Returns:
        Dictionary of experiment files found
    """
    experiment_files = {
        "configs": [],
        "summaries": [],
        "dataset_genealogy": []
    }
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logger.warning("Logs directory not found")
        return experiment_files
    
    # Find all config files
    experiment_files["configs"] = list(logs_dir.glob("config_*.yml"))
    
    # Find all experiment summary files
    experiment_files["summaries"] = list(logs_dir.glob("experiment_summary_*.json"))
    
    # Find all dataset genealogy files
    experiment_files["dataset_genealogy"] = list(logs_dir.glob("dataset_genealogy_*.json"))
    
    # If run_id is provided, filter to only include files related to that run
    if run_id:
        # Try to find the actual experiment ID from run_id
        api = wandb.Api()
        try:
            run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
            if 'experiment' in run.config and 'id' in run.config['experiment']:
                experiment_id = run.config['experiment']['id']
                logger.info(f"Found experiment ID {experiment_id} for run {run_id}")
                
                # Filter files to those containing the experiment ID
                experiment_files["configs"] = [f for f in experiment_files["configs"] if experiment_id in f.name]
                experiment_files["summaries"] = [f for f in experiment_files["summaries"] if experiment_id in f.name]
                experiment_files["dataset_genealogy"] = [f for f in experiment_files["dataset_genealogy"] if experiment_id in f.name]
        except Exception as e:
            logger.warning(f"Error retrieving run {run_id}: {str(e)}")
    
    logger.info(f"Found {len(experiment_files['configs'])} config files")
    logger.info(f"Found {len(experiment_files['summaries'])} experiment summary files")
    logger.info(f"Found {len(experiment_files['dataset_genealogy'])} dataset genealogy files")
    
    return experiment_files

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
            "This report analyzes the performance of the two-tower retrieval model. The model "
            "consists of two encoder networks (towers) that map queries and documents into a "
            "shared embedding space, where relevant pairs are positioned closer together than "
            "irrelevant ones."
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
        
        # Find experiment files
        experiment_files = find_experiment_files(run_id)
        
        # Try to load dataset genealogy for this run if available
        dataset_genealogy = None
        if experiment_files["dataset_genealogy"]:
            try:
                with open(experiment_files["dataset_genealogy"][0], 'r') as f:
                    dataset_genealogy = json.load(f)
                logger.info(f"Loaded dataset genealogy from {experiment_files['dataset_genealogy'][0]}")
            except Exception as e:
                logger.warning(f"Error loading dataset genealogy: {str(e)}")
        
        # Create the report
        report = wr.Report(
            project=project_name,
            entity=entity,
            title=report_title, 
            description=report_description,
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
                            title="Dataset Genealogy",
                            text=(
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
                        x="batch",
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
                        x="batch",
                        y=["train/pos_similarity", "train/neg_similarity"],
                        smoothing_factor=0.8,
                        layout=wr.Layout(w=12, h=8)
                    ),
                    wr.LinePlot(
                        title="Similarity Gap (Pos - Neg)",
                        x="batch",
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
                        x="batch",
                        y=["performance/batch_time"],
                        smoothing_factor=0.5,
                        layout=wr.Layout(w=8, h=6)
                    ),
                    wr.LinePlot(
                        title="Forward/Backward Time Breakdown",
                        x="batch",
                        y=["performance/forward_time", "performance/backward_time"],
                        smoothing_factor=0.5,
                        layout=wr.Layout(w=8, h=6)
                    ),
                    wr.LinePlot(
                        title="Training Throughput (Samples/Second)",
                        x="batch",
                        y=["performance/samples_per_second"],
                        smoothing_factor=0.5,
                        layout=wr.Layout(w=8, h=6)
                    ),
                    # Gradient analysis
                    wr.LinePlot(
                        title="Gradient Norm (Model Health)",
                        x="batch",
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
                        diff_only=False,
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
                    wr.ParameterImportancePlot(
                        with_respect_to="train/epoch_loss",
                        layout=wr.Layout(w=12, h=8)
                    ),
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
                        wr.TextPanel(
                            title="Dataset Statistics",
                            text=(
                                f"**Dataset Source**: MS MARCO ({dataset_genealogy.get('ms_marco_split', 'N/A')} split)\n\n"
                                f"**Sampling Rate**: {dataset_genealogy.get('sample_size', 'All')} samples\n\n"
                                f"**Triplets Format**: {dataset_genealogy.get('triplets_info', {}).get('row_count', 'N/A')} query-positive-negative triplets\n\n"
                                f"**Random Seed**: {dataset_genealogy.get('random_seed', 'N/A')}"
                            ),
                            layout=wr.Layout(w=12, h=8)
                        ),
                        # Dataset preprocessing visualization
                        wr.MarkdownPanel(
                            title="Dataset Preparation Pipeline",
                            text=(
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
        if len(experiment_files["summaries"]) > 1:
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
                for summary_file in experiment_files["summaries"]:
                    try:
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                            experiment_summaries.append(summary)
                    except Exception as e:
                        logger.warning(f"Error loading experiment summary {summary_file}: {str(e)}")
                
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
                                title="Experiment Timeline",
                                text=timeline_markdown,
                                layout=wr.Layout(w=24, h=12)
                            ),
                        ]
                    ),
                ])
            except Exception as e:
                logger.warning(f"Error creating experiment timeline: {str(e)}")
        
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