"""
Reusable blocks for W&B reports.

This module provides pre-defined panel blocks and components 
that can be reused across different report types.
"""

import wandb_workspaces.reports.v2 as wr
from typing import List, Optional

def toc():
    """Return a table of contents block."""
    return wr.TableOfContents()

def intro_two_tower():
    """Return an introduction section about two-tower models."""
    return [
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
    ]

def training_dynamics_panels(runset):
    """Return panels for training loss visualization."""
    return wr.PanelGrid(
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
    )

def similarity_panels(runset):
    """Return panels for query-document similarity visualization."""
    return wr.PanelGrid(
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
    )

def performance_panels(runset):
    """Return panels for training performance metrics."""
    return wr.PanelGrid(
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
        ]
    )

def gradient_panels(runset):
    """Return panels for gradient analysis."""
    return wr.PanelGrid(
        runsets=[runset],
        panels=[
            # Gradient analysis
            wr.LinePlot(
                title="Gradient Norm (Model Health)",
                x="train/batch",
                y=["gradients/total_norm", "train/grad_norm"],
                smoothing_factor=0.5,
                layout=wr.Layout(w=12, h=6)
            ),
        ]
    )

def config_panels(runset):
    """Return panels for model configuration display."""
    return wr.PanelGrid(
        runsets=[runset],
        panels=[
            # Config details
            wr.RunComparer(
                diff_only='split',
                layout=wr.Layout(w=24, h=15)
            ),
        ]
    )

def training_config_panels(runset):
    """Return panels for training configuration visualization."""
    return wr.PanelGrid(
        runsets=[runset],
        panels=[
            wr.LinePlot(
                title="Learning Rate",
                x="train/batch",
                y=["train/learning_rate"],
                smoothing_factor=0.2,
                layout=wr.Layout(w=8, h=6)
            ),
            wr.ScalarChart(
                title="Batch Size",
                metric="train/batch_size",
                layout=wr.Layout(w=8, h=6)
            ),
        ]
    )

def comparison_config_panel(runset):
    """Return a panel for comparing runs."""
    return wr.PanelGrid(
        runsets=[runset],
        panels=[
            # Run comparison
            wr.RunComparer(
                diff_only=True,
                layout=wr.Layout(w=24, h=10)
            ),
        ]
    )

def genealogy_panel(markdown_content):
    """Return a panel for dataset genealogy information."""
    return wr.PanelGrid(
        panels=[
            wr.MarkdownPanel(
                markdown=markdown_content,
                layout=wr.Layout(w=24, h=15)
            ),
        ]
    )

def mermaid_flowchart_panel(markdown_content):
    """Return a panel with a Mermaid.js flowchart."""
    return wr.PanelGrid(
        panels=[
            wr.MarkdownPanel(
                markdown=markdown_content,
                layout=wr.Layout(w=24, h=12)
            ),
        ]
    )

def dataset_analysis_panels(text_content, flowchart_content):
    """Return panels for dataset analysis."""
    return wr.PanelGrid(
        panels=[
            # Dataset statistics and genealogy
            wr.MarkdownPanel(
                markdown=text_content,
                layout=wr.Layout(w=12, h=8)
            ),
            # Dataset preprocessing visualization
            wr.MarkdownPanel(
                markdown=flowchart_content,
                layout=wr.Layout(w=12, h=8)
            ),
        ]
    )

def media_browser_panel(runset):
    """Return a media browser panel for example visualization."""
    return wr.PanelGrid(
        runsets=[runset],
        panels=[
            # Tables and media browser
            wr.MediaBrowser(
                media_keys=["examples/query", "examples/positive_doc", "examples/negative_doc"],
                layout=wr.Layout(w=24, h=10)
            ),
        ]
    )

def applications_next_steps():
    """Return a section on applications and next steps."""
    return [
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
    ] 