#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProcessConfigGenerator — Knowledge-Aware Process Clustering based on RG-GNN, CK-SC, CPR
Enhanced Visualization Version: Including algorithm iteration process and intuitive configuration results

Features:
1. Relation-Gated Graph Neural Network (RG-GNN) for process representation learning
2. Capacity-constrained Knowledge-aware Spectral Clustering (CK-SC)
3. Collaborative Process Refinement (CPR) based on graph contrastive learning
4. Equipment size constraint consideration: avoiding equipment overlap, optimizing sharing and space utilization
5. Complete algorithm iteration process visualization
6. Intuitive process configuration results display
"""

from __future__ import annotations

import time, json, os
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from py2neo import Graph
from typing import Optional, Dict, List, Tuple, Literal, Union, Any

# PyTorch相关
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 机器学习相关
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from scipy.linalg import eigh
from scipy.sparse import csgraph
from scipy.stats import wasserstein_distance

# 多目标优化
from deap import base, creator, tools, algorithms
import random

# 可视化相关
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not installed, interactive chart functionality will be disabled")

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️ NetworkX not installed, network diagram functionality will be disabled")

import warnings

warnings.filterwarnings('ignore')

# Set English font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12  # Increase base font size


# ===========================
# Enhanced Visualization Manager
# ===========================
class VisualizationManager:
    """Enhanced Visualization Manager - handles all chart generation and display"""

    def __init__(self, output_dir="process_config_results"):
        self.output_dir = output_dir
        self.iteration_data = {
            'rg_gnn_losses': [],
            'initial_clustering_steps': [],
            'kmeans_centers': [],
            'kmeans_inertias': [],
            'ga_generations': [],
            'ga_fitness_history': [],
            'cpr_optimization_steps': []
        }
        # Store labels for before/after comparison
        self.initial_labels = None
        self.final_labels = None

    def save_rg_gnn_loss(self, epoch, loss):
        """Save RG-GNN training loss"""
        self.iteration_data['rg_gnn_losses'].append({'epoch': epoch, 'loss': loss})

    def save_initial_clustering_step(self, step_name, data):
        """Save initial clustering step data"""
        self.iteration_data['initial_clustering_steps'].append({
            'step': step_name,
            'data': data,
            'timestamp': time.time()
        })

    def save_kmeans_iteration(self, iteration, centers, inertia):
        """Save K-means iteration data"""
        self.iteration_data['kmeans_centers'].append(centers.copy())
        self.iteration_data['kmeans_inertias'].append(inertia)

    def save_cpr_optimization_step(self, step, fitness_values, constraint_violations):
        """Save CPR optimization step data"""
        self.iteration_data['cpr_optimization_steps'].append({
            'step': step,
            'fitness_values': fitness_values,
            'constraint_violations': constraint_violations,
            'timestamp': time.time()
        })

    def save_ga_generation(self, generation, population_fitness, statistics):
        """Save genetic algorithm generation data with proper statistics"""
        self.iteration_data['ga_generations'].append(generation)

        # Save the statistics directly
        fitness_stats = {
            'generation': generation,
            'best_fitness': statistics['best'],
            'avg_fitness': statistics['avg'],
            'worst_fitness': statistics['worst']
        }

        self.iteration_data['ga_fitness_history'].append(fitness_stats)

    def plot_complete_algorithm_process(self, initial_embeddings, initial_labels, final_labels, df):
        """Plot the complete algorithm process with enhanced layout for HMC and resource analysis"""
        # Store labels for comparison
        self.initial_labels = initial_labels
        self.final_labels = final_labels

        fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Complete KPC Algorithm Process Visualization', fontsize=24, fontweight='bold')

        # Phase 1: RG-GNN Training Process (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_rg_gnn_training(ax1)

        # Phase 2: CPR Optimization Process (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_cpr_optimization_process(ax2)

        # Phase 3: Cluster Quality Evolution (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_cluster_quality_evolution(ax3)

        # Phase 4: Final vs Initial Clustering Comparison (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_clustering_comparison(ax4, initial_embeddings, initial_labels, final_labels, df)

        # Phase 5: Resource Utilization Before/After CPR (bottom left half)
        ax5_before = fig.add_subplot(gs[2, 0])
        ax5_after = fig.add_subplot(gs[2, 1])
        self._plot_resource_utilization_comparison(ax5_before, ax5_after, initial_labels, final_labels, df)

        # Phase 6: HMC Analysis Before/After CPR (bottom right half)
        ax6_before = fig.add_subplot(gs[2, 2])
        ax6_after = fig.add_subplot(gs[2, 3])
        self._plot_hmc_comparison(ax6_before, ax6_after, initial_labels, final_labels, df)

        plt.savefig(f"{self.output_dir}/complete_algorithm_process.png", dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"📊 Complete algorithm process visualization saved: {self.output_dir}/complete_algorithm_process.png")

    def _plot_rg_gnn_training(self, ax):
        """Plot RG-GNN training process"""
        if not self.iteration_data['rg_gnn_losses']:
            ax.text(0.5, 0.5, 'No RG-GNN training data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('RG-GNN Training Process', fontweight='bold', fontsize=16)
            return

        losses_df = pd.DataFrame(self.iteration_data['rg_gnn_losses'])

        # Plot training loss
        ax.plot(losses_df['epoch'], losses_df['loss'], 'b-', linewidth=3, marker='o', markersize=4)
        ax.set_title('RG-GNN Training Convergence', fontweight='bold', fontsize=16)
        ax.set_xlabel('Training Epoch', fontsize=14)
        ax.set_ylabel('Reconstruction Loss', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=12)

        # Add convergence indicator
        if len(losses_df) > 10:
            final_loss = losses_df['loss'].iloc[-1]
            initial_loss = losses_df['loss'].iloc[0]
            convergence_rate = (initial_loss - final_loss) / initial_loss * 100
            ax.text(0.05, 0.95, f'Convergence: {convergence_rate:.1f}%',
                    transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3",
                                                      facecolor='lightgreen', alpha=0.7), fontsize=13)

    def _plot_cpr_optimization_process(self, ax):
        """Plot CPR optimization process"""
        if not self.iteration_data['ga_fitness_history']:
            ax.text(0.5, 0.5, 'No CPR optimization data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('CPR Optimization Process', fontweight='bold', fontsize=16)
            return

        ga_df = pd.DataFrame(self.iteration_data['ga_fitness_history'])

        # Plot fitness evolution
        ax.plot(ga_df['generation'], ga_df['best_fitness'], 'r-', linewidth=3,
                label='Best Fitness', marker='^', markersize=6)
        ax.plot(ga_df['generation'], ga_df['avg_fitness'], 'orange', linewidth=3,
                label='Average Fitness', marker='o', markersize=5)
        ax.plot(ga_df['generation'], ga_df['worst_fitness'], 'gray', linewidth=2,
                label='Worst Fitness', marker='v', markersize=4)

        ax.set_title('CPR Multi-objective Optimization', fontweight='bold', fontsize=16)
        ax.set_xlabel('Generation', fontsize=14)
        ax.set_ylabel('Fitness Value', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)

        # Add improvement indicator
        if len(ga_df) > 1:
            improvement = ga_df['best_fitness'].iloc[-1] - ga_df['best_fitness'].iloc[0]
            ax.text(0.05, 0.95, f'Improvement: {improvement:.3f}',
                    transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3",
                                                      facecolor='lightblue', alpha=0.7), fontsize=13)

    def _plot_clustering_comparison(self, ax, embeddings, initial_labels, final_labels, df):
        """Plot comparison between initial and final clustering"""
        # Create side-by-side comparison
        if embeddings.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            X_2d = tsne.fit_transform(embeddings)
        else:
            X_2d = embeddings

        # Split the plot area
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        # Plot initial clustering (left side)
        unique_initial = np.unique(initial_labels)
        colors_initial = plt.cm.Set3(np.linspace(0, 1, len(unique_initial)))

        for i, label in enumerate(unique_initial):
            mask = initial_labels == label
            if mask.any():
                ax.scatter(X_2d[mask, 0] - X_2d[:, 0].max() * 0.6, X_2d[mask, 1],
                           c=[colors_initial[i]], s=50, alpha=0.6, marker='s')

        # Plot final clustering (right side)
        unique_final = np.unique(final_labels)
        colors_final = plt.cm.Set1(np.linspace(0, 1, len(unique_final)))

        for i, label in enumerate(unique_final):
            mask = final_labels == label
            if mask.any():
                ax.scatter(X_2d[mask, 0] + X_2d[:, 0].max() * 0.6, X_2d[mask, 1],
                           c=[colors_final[i]], s=50, alpha=0.8, marker='o')

        ax.set_title('Initial vs Final Clustering Comparison', fontweight='bold', fontsize=16)
        ax.text(0.25, 0.95, 'Initial', transform=ax.transAxes, ha='center',
                fontweight='bold', fontsize=14)
        ax.text(0.75, 0.95, 'Final', transform=ax.transAxes, ha='center',
                fontweight='bold', fontsize=14)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)

    def _plot_cluster_quality_evolution(self, ax):
        """Plot cluster quality metrics evolution"""
        if not self.iteration_data['kmeans_inertias']:
            ax.text(0.5, 0.5, 'No K-means iteration data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('Cluster Quality Evolution', fontweight='bold', fontsize=16)
            return

        iterations = range(len(self.iteration_data['kmeans_inertias']))
        inertias = self.iteration_data['kmeans_inertias']

        ax.plot(iterations, inertias, 'g-', linewidth=3, marker='s', markersize=5)
        ax.set_title('Cluster Quality Evolution (K-means Inertia)', fontweight='bold', fontsize=16)
        ax.set_xlabel('Iteration', fontsize=14)
        ax.set_ylabel('Within-cluster Sum of Squares', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)

        # Add quality improvement indicator
        if len(inertias) > 1:
            improvement = (inertias[0] - inertias[-1]) / inertias[0] * 100
            ax.text(0.05, 0.95, f'Quality Improvement: {improvement:.1f}%',
                    transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3",
                                                      facecolor='lightgreen', alpha=0.7), fontsize=13)

    def _plot_resource_utilization_comparison(self, ax_before, ax_after, initial_labels, final_labels, df):
        """Plot resource utilization analysis before and after CPR"""
        max_area = 10.0  # Match the max_area_per_cluster parameter

        # Before CPR
        unique_initial = np.unique(initial_labels)
        utilizations_before = []
        cluster_sizes_before = []

        for label in unique_initial:
            mask = initial_labels == label
            if mask.any():
                cluster_df = df[mask]
                total_area = cluster_df['equipment_area'].sum()
                utilization = min(1.0, total_area / max_area)
                utilizations_before.append(utilization)
                cluster_sizes_before.append(len(cluster_df))

        # After CPR
        unique_final = np.unique(final_labels)
        utilizations_after = []
        cluster_sizes_after = []

        for label in unique_final:
            mask = final_labels == label
            if mask.any():
                cluster_df = df[mask]
                total_area = cluster_df['equipment_area'].sum()
                utilization = min(1.0, total_area / max_area)
                utilizations_after.append(utilization)
                cluster_sizes_after.append(len(cluster_df))

        # Plot before
        self._plot_single_resource_utilization(ax_before, unique_initial, utilizations_before,
                                               cluster_sizes_before, "Resource Utilization (Before CPR)")

        # Plot after
        self._plot_single_resource_utilization(ax_after, unique_final, utilizations_after,
                                               cluster_sizes_after, "Resource Utilization (After CPR)")

        # Add improvement text
        avg_before = np.mean(utilizations_before) if utilizations_before else 0
        avg_after = np.mean(utilizations_after) if utilizations_after else 0
        improvement = (avg_after - avg_before) / avg_before * 100 if avg_before > 0 else 0

        ax_after.text(0.95, 0.05, f'Improvement: {improvement:+.1f}%',
                      transform=ax_after.transAxes, ha='right', va='bottom',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                      fontsize=12, fontweight='bold')

    def _plot_single_resource_utilization(self, ax, unique_labels, utilizations, cluster_sizes, title):
        """Helper function to plot single resource utilization chart"""
        if not utilizations:
            ax.text(0.5, 0.5, 'No utilization data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontweight='bold', fontsize=14)
            return

        # Create bubble chart
        bubble_sizes = [size * 60 for size in cluster_sizes]
        colors = ['green' if u >= 0.7 else 'orange' if u >= 0.4 else 'red' for u in utilizations]

        scatter = ax.scatter(unique_labels, utilizations, s=bubble_sizes, c=colors, alpha=0.6)

        # Add horizontal lines for utilization thresholds
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (≥70%)')
        ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Fair (≥40%)')

        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Space Utilization Rate', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=11)

        # Add statistics
        avg_utilization = np.mean(utilizations)
        ax.text(0.05, 0.95, f'Avg: {avg_utilization:.1%}',
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3",
                                                  facecolor='lightblue', alpha=0.7), fontsize=12)

    def _plot_hmc_comparison(self, ax_before, ax_after, initial_labels, final_labels, df):
        """Plot Human-Machine Collaboration analysis before and after CPR"""

        # Calculate HMC metrics before CPR
        hmc_before = self._calculate_hmc_metrics(initial_labels, df)

        # Calculate HMC metrics after CPR
        hmc_after = self._calculate_hmc_metrics(final_labels, df)

        # Plot before
        self._plot_single_hmc(ax_before, hmc_before, "Human-Machine Collaboration (Before CPR)")

        # Plot after
        self._plot_single_hmc(ax_after, hmc_after, "Human-Machine Collaboration (After CPR)")

        # Add improvement text
        avg_before = np.mean([m['efficiency'] for m in hmc_before])
        avg_after = np.mean([m['efficiency'] for m in hmc_after])
        improvement = (avg_after - avg_before) / avg_before * 100 if avg_before > 0 else 0

        ax_after.text(0.95, 0.05, f'Improvement: {improvement:+.1f}%',
                      transform=ax_after.transAxes, ha='right', va='bottom',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                      fontsize=12, fontweight='bold')

    def _calculate_hmc_metrics(self, labels, df):
        """Calculate Human-Machine Collaboration metrics for given labels"""
        unique_labels = np.unique(labels)
        hmc_metrics = []

        for label in unique_labels:
            mask = labels == label
            if mask.any():
                cluster_df = df[mask]

                # Collect all workers in the cluster
                all_workers = []
                for _, row in cluster_df.iterrows():
                    all_workers.extend(row['worker_list'])

                if all_workers:
                    unique_workers = len(set(all_workers))
                    total_workers = len(all_workers)
                    efficiency = unique_workers / total_workers

                    # Calculate worker load balance
                    worker_counts = {}
                    for worker in all_workers:
                        worker_counts[worker] = worker_counts.get(worker, 0) + 1

                    load_balance = 1.0 / (1.0 + np.std(list(worker_counts.values())) /
                                          (np.mean(list(worker_counts.values())) + 1e-6))
                else:
                    efficiency = 0
                    load_balance = 0
                    unique_workers = 0

                hmc_metrics.append({
                    'cluster_id': label,
                    'efficiency': efficiency,
                    'load_balance': load_balance,
                    'unique_workers': unique_workers,
                    'process_count': len(cluster_df)
                })

        return hmc_metrics

    def _plot_single_hmc(self, ax, hmc_metrics, title):
        """Helper function to plot single HMC chart"""
        if not hmc_metrics:
            ax.text(0.5, 0.5, 'No HMC data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontweight='bold', fontsize=14)
            return

        cluster_ids = [m['cluster_id'] for m in hmc_metrics]
        efficiencies = [m['efficiency'] for m in hmc_metrics]
        load_balances = [m['load_balance'] for m in hmc_metrics]
        worker_counts = [m['unique_workers'] for m in hmc_metrics]

        # ——柱状图：Worker Efficiency——
        bars = ax.bar(cluster_ids, efficiencies, alpha=0.7,
                      label='HMC Efficiency')

        # ——散点：Load Balance——
        ax2 = ax.twinx()
        scatter = ax2.scatter(cluster_ids, load_balances,
                              c='red',
                              s=[w * 10 for w in worker_counts],
                              alpha=0.6, marker='D',
                              label='Load Balance')

        # ① 标题与坐标轴
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('HMC Efficiency', fontsize=12)
        ax2.set_ylabel('Load Balance', fontsize=12)

        # ② 图例：合并两轴 hand​le/label
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=11)

        # ③ 轴范围
        ax.set_ylim(0, 1.1)
        ax2.set_ylim(0, 1.1)

        # ④ 根据效率上色
        for bar, eff in zip(bars, efficiencies):
            if eff >= 0.7:
                bar.set_color('green')
            elif eff >= 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        # ⑤ 栅格 & 字体
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=11)
        ax2.tick_params(axis='both', labelsize=11)

        # ⑥ 统计信息
        avg_efficiency = np.mean(efficiencies)
        ax.text(0.05, 0.95, f'Avg Efficiency: {avg_efficiency:.1%}',
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor='lightgreen', alpha=0.7),
                fontsize=12)

    def plot_configuration_layout_overview(self, clusters_data, df):
        """Create a separate large and clear configuration layout overview"""
        n_clusters = len(clusters_data)
        if n_clusters == 0:
            print("⚠️ No cluster data available for layout overview")
            return

        # Calculate optimal grid layout
        cols = min(6, n_clusters)  # Max 6 columns
        rows = (n_clusters + cols - 1) // cols

        # Create large figure
        fig_width = cols * 5 + 2
        fig_height = rows * 6 + 2
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        ax.set_xlim(0, cols * 5.5)
        ax.set_ylim(0, rows * 7)
        ax.set_aspect('equal')

        # Title
        ax.text(cols * 2.75, rows * 7 - 0.5, 'Process Configuration Layout Overview',
                ha='center', va='top', fontsize=24, fontweight='bold')

        cluster_ids = list(clusters_data.keys())

        for i, cluster_id in enumerate(cluster_ids):
            cluster_info = clusters_data[cluster_id]

            # Calculate position
            row = i // cols
            col = i % cols

            # Base position (bottom-left corner of workstation)
            base_x = col * 5.5 + 0.5
            base_y = (rows - row - 1) * 7 + 1

            # Workstation dimensions
            ws_width = 4.5
            ws_height = 5.5

            # Draw workstation boundary
            workstation = patches.Rectangle((base_x, base_y), ws_width, ws_height,
                                            linewidth=3, edgecolor='black',
                                            facecolor='lightgray', alpha=0.2)
            ax.add_patch(workstation)

            # Determine color based on utilization
            utilization = cluster_info.get("utilization", 0)
            if utilization >= 0.7:
                color = 'green'
                status = 'High'
            elif utilization >= 0.4:
                color = 'orange'
                status = 'Medium'
            else:
                color = 'red'
                status = 'Low'

            # Cluster header
            header_y = base_y + ws_height + 0.2
            ax.text(base_x + ws_width / 2, header_y, f'Configuration {cluster_id}',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

            # Statistics panel
            stats_y = base_y + ws_height - 0.5
            ax.text(base_x + ws_width / 2, stats_y, f'Processes: {cluster_info["process_count"]}',
                    ha='center', va='center', fontsize=11)
            ax.text(base_x + ws_width / 2, stats_y - 0.3, f'Equipment: {cluster_info["equipment_count"]}',
                    ha='center', va='center', fontsize=11)
            ax.text(base_x + ws_width / 2, stats_y - 0.6, f'Area: {cluster_info["total_area"]:.1f}m²',
                    ha='center', va='center', fontsize=11)
            ax.text(base_x + ws_width / 2, stats_y - 0.9, f'Utilization: {utilization:.1%}',
                    ha='center', va='center', fontsize=11, color=color, fontweight='bold')

            # Constraint status indicator
            constraint_ok = cluster_info.get("within_limit", False)
            status_text = '✓ Compliant' if constraint_ok else '✗ Violated'
            status_color = 'green' if constraint_ok else 'red'
            ax.text(base_x + ws_width / 2, stats_y - 1.2, status_text,
                    ha='center', va='center', fontsize=10, color=status_color, fontweight='bold')

            # Equipment layout visualization
            equipment_list = cluster_info.get("equipment_list", [])
            if equipment_list:
                # Simple grid layout for equipment
                eq_per_row = 3
                eq_rows = (len(equipment_list) + eq_per_row - 1) // eq_per_row
                eq_start_y = base_y + 0.3
                eq_height = min(3.5, (ws_height - 2) / max(1, eq_rows))

                for j, equipment in enumerate(equipment_list[:12]):  # Show max 12 equipment
                    eq_row = j // eq_per_row
                    eq_col = j % eq_per_row

                    eq_x = base_x + 0.3 + eq_col * 1.3
                    eq_y = eq_start_y + eq_row * (eq_height / eq_rows)
                    eq_width = 1.0
                    eq_h = eq_height / eq_rows - 0.1

                    # Equipment rectangle
                    eq_rect = patches.Rectangle((eq_x, eq_y), eq_width, eq_h,
                                                linewidth=1, edgecolor='darkblue',
                                                facecolor=color, alpha=0.6)
                    ax.add_patch(eq_rect)

                    # Equipment label
                    ax.text(eq_x + eq_width / 2, eq_y + eq_h / 2, equipment[:6],
                            ha='center', va='center', fontsize=7, fontweight='bold')

                # Show additional equipment count
                if len(equipment_list) > 12:
                    ax.text(base_x + ws_width / 2, base_y + 0.1, f'... +{len(equipment_list) - 12} more',
                            ha='center', va='center', fontsize=8, style='italic')

        # Legend
        legend_y = 0.5
        legend_elements = [
            patches.Patch(color='green', label='High Utilization (≥70%)'),
            patches.Patch(color='orange', label='Medium Utilization (40-70%)'),
            patches.Patch(color='red', label='Low Utilization (<40%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

        ax.set_title('Process Configuration Layout Overview', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Workshop Layout (East-West)', fontsize=12)
        ax.set_ylabel('Workshop Layout (North-South)', fontsize=12)

        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/configuration_layout_overview.png", dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"📊 Configuration layout overview saved: {self.output_dir}/configuration_layout_overview.png")

    def generate_enhanced_report(self, clusters_data, metrics, space_metrics):
        """Generate enhanced HTML report with English content"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🏭 Process Configuration Clustering Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                         background-color: #e6f3ff; border-radius: 8px; }}
                .cluster {{ margin: 20px 0; padding: 15px; border: 1px solid #ccc; 
                          border-radius: 10px; background-color: white; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                .chart-section {{ margin: 20px 0; padding: 15px; background-color: white; 
                                border-radius: 10px; border: 1px solid #ddd; }}
                .phase-section {{ margin: 20px 0; padding: 15px; background-color: #f9f9f9; 
                                border-radius: 10px; border-left: 4px solid #007bff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏭 KPC Knowledge-Aware Process Clustering Analysis Report</h1>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Algorithm: RG-GNN + CK-SC + CPR</p>
            </div>

            <div class="phase-section">
                <h2>📊 Algorithm Execution Phases</h2>
                <h3>Phase 1: RG-GNN Representation Learning</h3>
                <p>✅ Completed graph neural network training for process embeddings</p>

                <h3>Phase 2: CK-SC Capacity-Constrained Spectral Clustering</h3>
                <p>✅ Initial clustering with capacity constraints applied</p>

                <h3>Phase 3: CPR Collaborative Process Refinement</h3>
                <p>✅ Optimization through genetic algorithm for final clustering</p>
            </div>

            <div class="chart-section">
                <h2>📊 Clustering Quality Metrics</h2>
                <div class="metric">
                    <strong>Silhouette Score:</strong> {metrics.get('silhouette', 0):.3f}
                </div>
                <div class="metric">
                    <strong>Calinski-Harabasz Index:</strong> {metrics.get('calinski_harabasz', 0):.1f}
                </div>
                <div class="metric">
                    <strong>Davies-Bouldin Index:</strong> {metrics.get('davies_bouldin', 0):.3f}
                </div>
            </div>

            <div class="chart-section">
                <h2>🏗️ Space Constraint Satisfaction</h2>
                <div class="metric">
                    <strong>Average Utilization:</strong> {space_metrics.get('avg_utilization', 0):.1%}
                </div>
                <div class="metric">
                    <strong>Constraint Satisfaction Rate:</strong> 
                    {space_metrics.get('clusters_within_limit', 0)}/{space_metrics.get('total_clusters', 0)}
                </div>
            </div>

            <div class="chart-section">
                <h2>🔧 Process Configuration Details</h2>
        """

        for cluster_id, info in clusters_data.items():
            status_class = "success" if info["within_limit"] else "error"
            html_content += f"""
            <div class="cluster">
                <h3 class="{status_class}">Configuration {cluster_id}</h3>
                <p><strong>Process Count:</strong> {info["process_count"]}</p>
                <p><strong>Equipment Count:</strong> {info["equipment_count"]}</p>
                <p><strong>Total Area:</strong> {info["total_area"]:.2f} m²</p>
                <p><strong>Utilization Rate:</strong> {info["utilization"]:.1%}</p>
                <p><strong>Equipment Sharing Rate:</strong> {info["sharing_ratio"]:.1%}</p>
                <p><strong>Constraint Status:</strong> 
                   <span class="{status_class}">
                   {'✅ Satisfied' if info["within_limit"] else '❌ Violated'}
                   </span>
                </p>
                <p><strong>Equipment List:</strong> {', '.join(info["equipment_list"][:5])}{'...' if len(info["equipment_list"]) > 5 else ''}</p>
            </div>
            """

        html_content += """
            </div>

            <div class="chart-section">
                <h2>📈 Visualization Charts</h2>
                <ul>
                    <li><a href="complete_algorithm_process.png" target="_blank">Complete Algorithm Process</a></li>
                    <li><a href="configuration_layout_overview.png" target="_blank">Configuration Layout Overview</a></li>
                    <li><a href="kpc_cluster_visualization.png" target="_blank">Final Clustering Visualization</a></li>
                </ul>
            </div>
        </body>
        </html>
        """

        with open(f"{self.output_dir}/enhanced_analysis_report.html", 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📋 Enhanced analysis report generated: {self.output_dir}/enhanced_analysis_report.html")


# ===========================
# RG-GNN Module Definition
# ===========================
class RelationGatedGNN(nn.Module):
    """Relation-Gated Graph Neural Network"""

    def __init__(self, input_dim, hidden_dim, output_dim, n_relations, n_layers=3, dropout=0.1):
        super(RelationGatedGNN, self).__init__()
        self.n_layers = n_layers
        self.n_relations = n_relations
        self.dropout = dropout

        # Initialize embedding layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Relation-specific weight matrices
        self.relation_weights = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(n_relations)
        ])

        # Gating mechanism parameters
        self.gate_weights = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_relations)
        ])

        # Attention mechanism parameters
        self.attention_weights = nn.ModuleList([
            nn.Linear(2 * hidden_dim, 1)
            for _ in range(n_relations)
        ])

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_layers)
        ])

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, num_nodes=None):
        """Forward propagation"""
        # Initial embedding
        h = self.input_projection(x)
        h = F.relu(h)

        # Multi-layer graph convolution
        for layer in range(self.n_layers):
            # Aggregate messages for each relation
            messages = []

            for r in range(self.n_relations):
                # Get edges for current relation
                mask = edge_type == r
                if not mask.any():
                    continue

                r_edge_index = edge_index[:, mask]

                # Calculate attention weights
                src, dst = r_edge_index
                h_src = h[src]
                h_dst = h[dst]

                # Attention calculation
                att_input = torch.cat([h_src, h_dst], dim=-1)
                att_scores = self.attention_weights[r](att_input)
                att_scores = F.leaky_relu(att_scores, 0.2)
                att_scores = F.softmax(att_scores, dim=0)

                # Message passing
                msg = self.relation_weights[r](h_src) * att_scores

                # Aggregate messages
                msg_agg = torch.zeros(h.size(), dtype=h.dtype, device=h.device)
                msg_agg.index_add_(0, dst, msg)

                # Gating mechanism
                gate = torch.sigmoid(self.gate_weights[r](h))
                msg_gated = gate * msg_agg

                messages.append(msg_gated)

            # Aggregate all relation messages
            if messages:
                h_new = sum(messages) / len(messages)
                h = self.layer_norms[layer](h + self.dropout_layer(h_new))
                h = F.relu(h)

        # Output projection
        out = self.output_projection(h)
        return out


# ===========================
# Enhanced Main Clusterer Class
# ===========================
class ProcessClusterer:
    """Enhanced Knowledge-Aware Process Clusterer based on RG-GNN, CK-SC, CPR"""

    def __init__(self, neo4j_uri: str, neo4j_auth: tuple[str, str], log_level: str = "info", device=None):
        """Initialize clusterer"""
        self.graph = Graph(neo4j_uri, auth=neo4j_auth)
        self.log_level = log_level
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model related
        self.rg_gnn = None
        self.node_embeddings = None

        # Data related
        self.base_features = None
        self.feature_names = None
        self.edge_index = None
        self.edge_type = None

        # Clustering related
        self.cluster_method = "KPC"
        self.distance_metric = "composite"
        self.metrics = {}
        self.space_metrics = {}

        # Output directory
        self.output_dir = "process_config_results"
        os.makedirs(self.output_dir, exist_ok=True)

        # Workstation size constraints (meters)
        self.workstation_width = 3.5
        self.workstation_length = 7.0
        self.max_workstation_area = self.workstation_width * self.workstation_length

        # Process count constraints
        self.max_processes_per_cluster = 10

        # Enhanced visualization manager
        self.viz_manager = VisualizationManager(self.output_dir)

    def log(self, message: str, level: str = "info"):
        """Log output"""
        if level == "debug" and self.log_level != "debug":
            return
        tqdm.write(message)

    # ------------------------------------------------------------------
    # 1. Data Loading
    # ------------------------------------------------------------------
    def _load_data(self, plan: dict[str, int]) -> pd.DataFrame:
        """Load process data from Neo4j including equipment size information"""
        bm_str = ",".join(f"'{bm}'" for bm in plan)
        cypher = f"""
        MATCH (pr:Process)
        WHERE pr.product_bm IN [{bm_str}]
        OPTIONAL MATCH (pr)-[:REQUIRES_EQUIPMENT]->(eq:Equipment)
        OPTIONAL MATCH (pr)-[:REQUIRES_MATERIAL]->(ma:Material)
        OPTIONAL MATCH (pr)-[:REQUIRES_WORKSTATION]->(ws:Workstation)
        OPTIONAL MATCH (pr)-[:REQUIRES_WORKER]->(wo:Worker)
        RETURN pr.process_id AS pid,
               pr.product_bm AS bm,
               pr.product_version AS ver,
               COLLECT(DISTINCT eq.name) AS eqp_list,
               COLLECT(DISTINCT eq.size) AS eqp_size_list,
               COLLECT(DISTINCT ma.type) AS mat_list,
               COLLECT(DISTINCT ws.name) AS ws_list,
               COLLECT(DISTINCT wo.name) AS worker_list
        """
        data = self.graph.run(cypher).data()
        if not data:
            raise ValueError("❌ No corresponding processes found in graph, please check BM numbers")
        df = pd.DataFrame(data)
        for col in ("eqp_list", "mat_list", "ws_list", "worker_list", "eqp_size_list"):
            df[col] = df[col].apply(
                lambda lst: lst if lst and not (isinstance(lst, list) and len(lst) == 1 and lst[0] is None) else [])

        # Calculate equipment area and constraint checking
        df['equipment_area'] = df.apply(lambda row: self._calculate_equipment_area_with_mapping(
            row['eqp_list'], row['eqp_size_list']), axis=1)
        df['size_constraint_met'] = df.apply(lambda row: self._check_size_constraint_with_mapping(
            row['eqp_list'], row['eqp_size_list'], self.workstation_width, self.workstation_length), axis=1)

        # Save process counts
        counts = {}
        for bm, count in plan.items():
            filtered = df[df.bm == bm]
            counts[bm] = {"requested": count, "found": len(filtered)}

        self.log(f"✅ Loaded {len(df)} unique processes")
        self.log(f"📊 Process statistics by product: {counts}")

        return df

    # ------------------------------------------------------------------
    # 2. Build Graph Structure and Relations
    # ------------------------------------------------------------------
    def _build_graph_structure(self, df):
        """Build graph structure including node relations and edge types"""
        n_nodes = len(df)
        edges = []
        edge_types = []

        # Define relation types
        REL_EQUIPMENT = 0  # Equipment sharing relation
        REL_MATERIAL = 1  # Material sharing relation
        REL_SEQUENCE = 2  # Process sequence relation
        REL_WORKER = 3  # Worker collaboration relation

        # Build relation edges
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                row_i = df.iloc[i]
                row_j = df.iloc[j]

                # Equipment sharing relation
                if set(row_i['eqp_list']) & set(row_j['eqp_list']):
                    edges.append([i, j])
                    edge_types.append(REL_EQUIPMENT)
                    edges.append([j, i])
                    edge_types.append(REL_EQUIPMENT)

                # Material sharing relation
                if set(row_i['mat_list']) & set(row_j['mat_list']):
                    edges.append([i, j])
                    edge_types.append(REL_MATERIAL)
                    edges.append([j, i])
                    edge_types.append(REL_MATERIAL)

                # Process sequence relation (same product adjacent version)
                if row_i['bm'] == row_j['bm']:
                    ver_i = self._parse_version(row_i['ver'])
                    ver_j = self._parse_version(row_j['ver'])
                    if abs(ver_i - ver_j) == 1:
                        edges.append([i, j])
                        edge_types.append(REL_SEQUENCE)
                        edges.append([j, i])
                        edge_types.append(REL_SEQUENCE)

                # Worker collaboration relation
                if set(row_i['worker_list']) & set(row_j['worker_list']):
                    edges.append([i, j])
                    edge_types.append(REL_WORKER)
                    edges.append([j, i])
                    edge_types.append(REL_WORKER)

        # Convert to tensors
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().to(self.device)
            edge_type = torch.tensor(edge_types, dtype=torch.long).to(self.device)
        else:
            # If no edges, create self-loops
            edge_index = torch.tensor([[i, i] for i in range(n_nodes)], dtype=torch.long).t().to(self.device)
            edge_type = torch.zeros(n_nodes, dtype=torch.long).to(self.device)

        self.edge_index = edge_index
        self.edge_type = edge_type

        return edge_index, edge_type

    # ------------------------------------------------------------------
    # 3. Enhanced RG-GNN Representation Learning
    # ------------------------------------------------------------------
    def _learn_representations(self, df, hidden_dim=128, output_dim=64, n_epochs=200):
        """Learn node representations using RG-GNN - Enhanced version"""
        self.log("🧠 Starting RG-GNN representation learning...")

        # Prepare features
        features = self._prepare_node_features(df)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        # Build graph structure
        edge_index, edge_type = self._build_graph_structure(df)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)

        # Initialize RG-GNN
        input_dim = features.shape[1]
        n_relations = 4
        self.rg_gnn = RelationGatedGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_relations=n_relations,
            n_layers=3
        ).to(self.device)

        # Optimizer
        optimizer = optim.Adam(self.rg_gnn.parameters(), lr=0.01)

        # Train RG-GNN (using self-supervised learning)
        self.rg_gnn.train()
        for epoch in tqdm(range(n_epochs), desc="RG-GNN Training"):
            optimizer.zero_grad()

            # Forward propagation
            embeddings = self.rg_gnn(features_tensor, edge_index, edge_type)

            # Self-supervised loss: neighborhood reconstruction
            loss = self._compute_reconstruction_loss(embeddings, edge_index, edge_type)

            # Backward propagation
            loss.backward()
            optimizer.step()

            # Save loss for visualization
            if epoch % 10 == 0:
                self.viz_manager.save_rg_gnn_loss(epoch, loss.item())

            if epoch % 50 == 0:
                self.log(f"  Epoch {epoch}, Loss: {loss.item():.4f}", level="debug")

        # Get final embeddings
        self.rg_gnn.eval()
        with torch.no_grad():
            self.node_embeddings = self.rg_gnn(features_tensor, edge_index, edge_type).cpu().numpy()

        self.log(f"✅ RG-GNN representation learning completed, embedding dimension: {self.node_embeddings.shape}")
        return self.node_embeddings

    def _prepare_node_features(self, df):
        """Prepare node features"""
        # Get all unique equipment, materials, workstations, workers
        all_equipment = set()
        all_materials = set()
        all_workstations = set()
        all_workers = set()

        for _, row in df.iterrows():
            all_equipment.update(row['eqp_list'])
            all_materials.update(row['mat_list'])
            all_workstations.update(row['ws_list'])
            all_workers.update(row['worker_list'])

        # Create mappings
        equipment_map = {eq: i for i, eq in enumerate(all_equipment)}
        material_map = {mat: i for i, mat in enumerate(all_materials)}
        workstation_map = {ws: i for i, ws in enumerate(all_workstations)}
        worker_map = {wk: i for i, wk in enumerate(all_workers)}

        # Create feature matrix
        n_nodes = len(df)
        n_features = len(equipment_map) + len(material_map) + len(workstation_map) + len(worker_map) + 3
        features = np.zeros((n_nodes, n_features))

        for i, row in df.iterrows():
            col_idx = 0

            # Equipment features
            for eq in row['eqp_list']:
                if eq in equipment_map:
                    features[i, col_idx + equipment_map[eq]] = 1
            col_idx += len(equipment_map)

            # Material features
            for mat in row['mat_list']:
                if mat in material_map:
                    features[i, col_idx + material_map[mat]] = 1
            col_idx += len(material_map)

            # Workstation features
            for ws in row['ws_list']:
                if ws in workstation_map:
                    features[i, col_idx + workstation_map[ws]] = 1
            col_idx += len(workstation_map)

            # Worker features
            for wk in row['worker_list']:
                if wk in worker_map:
                    features[i, col_idx + worker_map[wk]] = 1
            col_idx += len(worker_map)

            # Add additional features
            features[i, -3] = row['equipment_area'] / self.max_workstation_area  # Normalized area
            features[i, -2] = 1 if row['size_constraint_met'] else 0
            features[i, -1] = len(row['eqp_list']) / 10.0  # Normalized equipment count

        self.base_features = features
        return features

    def _compute_reconstruction_loss(self, embeddings, edge_index, edge_type):
        """Compute neighborhood reconstruction loss"""
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        src, dst = edge_index
        h_src = embeddings[src]
        h_dst = embeddings[dst]

        # Calculate similarity
        sim = F.cosine_similarity(h_src, h_dst)

        # Positive sample loss
        pos_loss = -torch.log(torch.sigmoid(sim) + 1e-8).mean()

        # Negative sampling
        neg_src = torch.randint(0, embeddings.size(0), (edge_index.size(1),)).to(self.device)
        neg_dst = torch.randint(0, embeddings.size(0), (edge_index.size(1),)).to(self.device)
        h_neg_src = embeddings[neg_src]
        h_neg_dst = embeddings[neg_dst]

        # Negative sample loss
        neg_sim = F.cosine_similarity(h_neg_src, h_neg_dst)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_sim) + 1e-8).mean()

        return pos_loss + neg_loss

    # ------------------------------------------------------------------
    # 4. Enhanced CK-SC Capacity-Constrained Spectral Clustering
    # ------------------------------------------------------------------
    def _capacity_constrained_spectral_clustering(self, embeddings, df, n_clusters):
        """Enhanced capacity-constrained knowledge-aware spectral clustering"""
        self.log(f"📊 Starting CK-SC capacity-constrained spectral clustering, target clusters: {n_clusters}")

        with tqdm(total=4, desc="CK-SC Spectral Clustering") as pbar:
            # Step 1: Build multi-dimensional similarity matrix
            pbar.set_description("Computing similarity matrix")
            S = self._compute_composite_similarity_matrix(embeddings, df)
            pbar.update(1)

            # Step 2: Build Laplacian matrix
            pbar.set_description("Building Laplacian matrix")
            D = np.diag(S.sum(axis=1))
            L = D - S
            pbar.update(1)

            # Step 3: Spectral decomposition
            pbar.set_description("Performing spectral decomposition")
            eigenvalues, eigenvectors = eigh(L, D)
            V = eigenvectors[:, :n_clusters]

            # Save initial clustering step
            self.viz_manager.save_initial_clustering_step("spectral_decomposition", V)
            pbar.update(1)

            # Step 4: Capacity-constrained K-means
            pbar.set_description("Executing constrained K-means")
            labels = self._enhanced_capacity_constrained_kmeans(V, df, n_clusters)

            # Save initial clustering result
            self.viz_manager.save_initial_clustering_step("initial_clustering", labels)
            pbar.update(1)

        return labels

    def _enhanced_capacity_constrained_kmeans(self, V, df, n_clusters, max_iter=100):
        """Enhanced capacity-constrained K-means clustering - ensures no empty clusters"""
        n_nodes = V.shape[0]

        # Initialize centers using K-means++
        centers = self._initialize_centers_kmeans_plus_plus(V, n_clusters)

        # Get constraint parameters
        capacities = df['equipment_area'].values
        max_area_capacity = self.max_workstation_area
        max_process_capacity = self.max_processes_per_cluster

        with tqdm(total=max_iter, desc="Constrained K-means Iteration", leave=False) as pbar:
            for iteration in range(max_iter):
                # Initialize cluster assignments
                labels = np.full(n_nodes, -1, dtype=int)
                cluster_area_capacities = np.zeros(n_clusters)
                cluster_process_counts = np.zeros(n_clusters, dtype=int)

                # Calculate distance matrix
                distances = np.zeros((n_nodes, n_clusters))
                for k in range(n_clusters):
                    distances[:, k] = np.linalg.norm(V - centers[k], axis=1)

                # Sort nodes by minimum distance to any center
                assignment_order = np.argsort(distances.min(axis=1))

                # Assign nodes to clusters
                for i in assignment_order:
                    sorted_clusters = np.argsort(distances[i])
                    assigned = False

                    # Try to assign to feasible clusters
                    for k in sorted_clusters:
                        area_ok = cluster_area_capacities[k] + capacities[i] <= max_area_capacity
                        process_ok = cluster_process_counts[k] < max_process_capacity

                        if area_ok and process_ok:
                            labels[i] = k
                            cluster_area_capacities[k] += capacities[i]
                            cluster_process_counts[k] += 1
                            assigned = True
                            break

                    # If no feasible cluster, assign to nearest cluster
                    if not assigned:
                        best_k = sorted_clusters[0]
                        labels[i] = best_k
                        cluster_area_capacities[best_k] += capacities[i]
                        cluster_process_counts[best_k] += 1

                # Ensure no empty clusters
                labels = self._ensure_no_empty_clusters(labels, V, n_clusters)

                # Calculate inertia
                inertia = 0
                for k in range(n_clusters):
                    mask = labels == k
                    if mask.any():
                        cluster_points = V[mask]
                        inertia += np.sum((cluster_points - centers[k]) ** 2)

                self.viz_manager.save_kmeans_iteration(iteration, centers, inertia)

                # Update centers
                new_centers = np.zeros_like(centers)
                for k in range(n_clusters):
                    mask = labels == k
                    if mask.any():
                        new_centers[k] = V[mask].mean(axis=0)
                    else:
                        new_centers[k] = centers[k]

                # Check convergence
                if np.allclose(centers, new_centers, rtol=1e-4):
                    pbar.set_description(f"Converged at iteration {iteration + 1}")
                    pbar.update(max_iter - iteration)
                    break

                centers = new_centers
                pbar.update(1)

        return labels

    def _initialize_centers_kmeans_plus_plus(self, X, n_clusters):
        """Initialize centers using K-means++ algorithm"""
        n_samples, n_features = X.shape
        centers = np.empty((n_clusters, n_features))

        # Choose first center randomly
        centers[0] = X[np.random.randint(n_samples)]

        # Choose remaining centers
        for c_id in range(1, n_clusters):
            # Calculate distances to nearest center
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centers[:c_id]]) for x in X])

            # Choose next center with probability proportional to squared distance
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centers[c_id] = X[j]
                    break

        return centers

    def _ensure_no_empty_clusters(self, labels, V, n_clusters):
        """Ensure no clusters are empty by reassigning if necessary"""
        unique_labels = np.unique(labels)

        if len(unique_labels) < n_clusters:
            # Find empty cluster IDs
            all_cluster_ids = set(range(n_clusters))
            used_cluster_ids = set(unique_labels)
            empty_cluster_ids = list(all_cluster_ids - used_cluster_ids)

            # Reassign some points to empty clusters
            for empty_id in empty_cluster_ids:
                # Find the largest cluster
                cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
                largest_cluster = max(cluster_sizes, key=lambda x: x[1])[0]

                # Move one point from largest cluster to empty cluster
                largest_cluster_indices = np.where(labels == largest_cluster)[0]
                if len(largest_cluster_indices) > 1:  # Only move if cluster has more than 1 point
                    # Move the point furthest from cluster center
                    largest_cluster_points = V[largest_cluster_indices]
                    center = largest_cluster_points.mean(axis=0)
                    distances = np.linalg.norm(largest_cluster_points - center, axis=1)
                    furthest_idx = largest_cluster_indices[np.argmax(distances)]
                    labels[furthest_idx] = empty_id

        return labels

    def _compute_composite_similarity_matrix(self, embeddings, df):
        """Compute composite similarity matrix"""
        n_nodes = len(embeddings)

        # Semantic similarity
        sem_sim = cosine_similarity(embeddings)

        # Resource compatibility
        res_sim = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                eqp_overlap = len(set(df.iloc[i]['eqp_list']) & set(df.iloc[j]['eqp_list']))
                mat_overlap = len(set(df.iloc[i]['mat_list']) & set(df.iloc[j]['mat_list']))
                res_sim[i, j] = res_sim[j, i] = (eqp_overlap + mat_overlap) / 10.0

        # Process continuity
        seq_sim = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if df.iloc[i]['bm'] == df.iloc[j]['bm']:
                    seq_sim[i, j] = seq_sim[j, i] = self._compute_version_similarity(
                        df.iloc[i]['ver'], df.iloc[j]['ver']
                    )

        # Human-machine collaboration compatibility
        hmc_sim = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                worker_overlap = len(set(df.iloc[i]['worker_list']) & set(df.iloc[j]['worker_list']))
                worker_count = max(1, len(df.iloc[i]['worker_list']))
                hmc_sim[i, j] = hmc_sim[j, i] = worker_overlap / worker_count

        # Composite similarity (adjustable weights)
        S = 0.4 * sem_sim + 0.3 * res_sim + 0.2 * seq_sim + 0.1 * hmc_sim

        # Ensure symmetry and non-negativity
        S = (S + S.T) / 2
        S = np.maximum(S, 0)

        return S

    def _parse_version(self, version_str):
        """Parse version string to extract numbers for comparison"""
        if not version_str or version_str is None:
            return 0

        try:
            # If it's a pure numeric string, convert directly
            if str(version_str).isdigit():
                return int(version_str)

            # Handle formats like 'V3.1.05'
            version_str = str(version_str).upper()

            # Remove non-numeric characters, extract first number group
            import re
            numbers = re.findall(r'\d+', version_str)
            if numbers:
                # Use first number as main version
                return int(numbers[0])
            else:
                return 0

        except (ValueError, AttributeError):
            return 0

    def _compute_version_similarity(self, ver1, ver2):
        """Compute similarity between two versions"""
        v1 = self._parse_version(ver1)
        v2 = self._parse_version(ver2)

        # Smaller version difference, higher similarity
        diff = abs(v1 - v2)
        return np.exp(-diff) if diff > 0 else 1.0

    # ------------------------------------------------------------------
    # 5. Enhanced CPR Collaborative Process Refinement
    # ------------------------------------------------------------------
    def _collaborative_process_refinement(self, initial_labels, embeddings, df):
        """Enhanced collaborative process refinement based on graph contrastive learning"""
        self.log("🔧 Starting CPR collaborative process refinement...")

        with tqdm(total=2, desc="CPR Refinement") as pbar:
            # Step 1: Graph contrastive learning
            pbar.set_description("Graph contrastive learning")
            enhanced_embeddings = self._graph_contrastive_learning(embeddings, initial_labels)
            pbar.update(1)

            # Step 2: Multi-objective optimization
            pbar.set_description("Multi-objective optimization")
            final_labels = self._enhanced_multi_objective_optimization(initial_labels, enhanced_embeddings, df)
            pbar.update(1)

        return final_labels

    def _graph_contrastive_learning(self, embeddings, labels, n_epochs=50):
        """Graph contrastive learning to enhance embeddings"""
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)

        # Create projection head
        projector = nn.Sequential(
            nn.Linear(embeddings.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(self.device)

        optimizer = optim.Adam(projector.parameters(), lr=0.001)

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Project embeddings
            z = projector(embeddings_tensor)

            # Build positive and negative sample pairs
            loss = 0
            for i in range(len(labels)):
                # Positive samples: same cluster nodes
                pos_mask = labels == labels[i]
                pos_mask[i] = False

                if pos_mask.any():
                    pos_sim = F.cosine_similarity(z[i].unsqueeze(0), z[pos_mask])

                    # Negative samples: different cluster nodes
                    neg_mask = labels != labels[i]
                    if neg_mask.any():
                        neg_sim = F.cosine_similarity(z[i].unsqueeze(0), z[neg_mask])

                        # NT-Xent loss
                        tau = 0.5
                        pos_exp = torch.exp(pos_sim / tau)
                        neg_exp = torch.exp(neg_sim / tau)

                        loss += -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum() + 1e-8))

            loss = loss / len(labels)
            loss.backward()
            optimizer.step()

        # Get enhanced embeddings
        with torch.no_grad():
            enhanced = projector(embeddings_tensor).cpu().numpy()

        return enhanced

    def _enhanced_multi_objective_optimization(self, initial_labels, embeddings, df):
        """Enhanced multi-objective optimization with properly scaled fitness calculation"""

        n_clusters = len(np.unique(initial_labels))

        # Define constraint checking function
        def check_constraints(labels):
            unique_labels = np.unique(labels)
            violations = 0

            for label in unique_labels:
                mask = labels == label
                cluster_process_count = mask.sum()
                cluster_area = df.loc[mask, 'equipment_area'].sum()

                if cluster_process_count > self.max_processes_per_cluster:
                    violations += (
                                              cluster_process_count - self.max_processes_per_cluster) / self.max_processes_per_cluster
                if cluster_area > self.max_workstation_area:
                    violations += (cluster_area - self.max_workstation_area) / self.max_workstation_area

            return violations

        # Define enhanced objective function with proper scaling
        def evaluate(individual):
            labels = np.array(individual)

            # Constraint violation penalty
            constraint_violations = check_constraints(labels)
            penalty = constraint_violations * 0.1  # Moderate penalty factor

            # Objective 1: Maximize intra-cluster similarity (already normalized 0-1)
            f_sim = 0
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                if mask.sum() > 1:
                    cluster_embeddings = embeddings[mask]
                    # Cosine similarity already returns values between -1 and 1
                    # We shift and scale to 0-1
                    sim = (cosine_similarity(cluster_embeddings).mean() + 1) / 2
                    f_sim += sim
            f_sim = f_sim / max(1, len(unique_labels))

            # Objective 2: Maximize human-machine collaboration efficiency (0-1)
            f_hmc = 0
            for label in unique_labels:
                mask = labels == label
                workers = []
                for idx in np.where(mask)[0]:
                    workers.extend(df.iloc[idx]['worker_list'])
                if workers:
                    unique_workers = len(set(workers))
                    total_workers = len(workers)
                    # This is already normalized between 0 and 1
                    f_hmc += unique_workers / max(1, total_workers)
            f_hmc = f_hmc / max(1, len(unique_labels))

            # Objective 3: Resource load balancing (0-1)
            f_res = 0
            cluster_areas = []
            for label in unique_labels:
                mask = labels == label
                area = df.loc[mask, 'equipment_area'].sum()
                cluster_areas.append(area)
            if cluster_areas:
                mean_area = np.mean(cluster_areas)
                std_area = np.std(cluster_areas)
                # Normalize: perfect balance = 1, worst case approaches 0
                cv = std_area / (mean_area + 1e-6)  # Coefficient of variation
                f_res = 1.0 / (1.0 + cv)  # Transform to 0-1 scale

            # Objective 4: Process count balancing (0-1)
            f_balance = 0
            cluster_counts = []
            for label in unique_labels:
                mask = labels == label
                count = mask.sum()
                cluster_counts.append(count)
            if cluster_counts:
                mean_count = np.mean(cluster_counts)
                std_count = np.std(cluster_counts)
                cv = std_count / (mean_count + 1e-6)
                f_balance = 1.0 / (1.0 + cv)

            # Calculate final fitness values with penalty
            # All objectives are now properly scaled between 0 and 1
            # fitness_values = (
            #     max(0.001, f_sim - penalty),
            #     max(0.001, f_hmc - penalty),
            #     max(0.001, f_res - penalty),
            #     max(0.001, f_balance - penalty)
            # )
            penalty = constraint_violations * 0.05  # 温和一点
            ε = 1e-6
            fitness_values = (
                max(ε, f_sim - penalty),
                max(ε, f_hmc - penalty),
                max(ε, f_res - penalty),
                max(ε, f_balance - penalty)
            )

            # Save CPR optimization step
            self.viz_manager.save_cpr_optimization_step(
                len(self.viz_manager.iteration_data['cpr_optimization_steps']),
                fitness_values,
                constraint_violations
            )

            return fitness_values

        # Setup DEAP
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("attr_int", random.randint, 0, n_clusters - 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_int, n=len(initial_labels))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=n_clusters - 1, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)

        # Initialize population
        pop_size = 50
        n_generations = 500  # Increased generations

        pop = toolbox.population(n=pop_size)
        pop[0] = creator.Individual(initial_labels.tolist())

        # Add variations of initial solution
        for i in range(1, min(10, len(pop))):
            variant = initial_labels.copy()
            n_changes = max(1, len(variant) // 10)
            change_indices = np.random.choice(len(variant), n_changes, replace=False)
            for idx in change_indices:
                variant[idx] = np.random.randint(0, n_clusters)
            pop[i] = creator.Individual(variant.tolist())

        # Evaluate initial population
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Evolution process
        with tqdm(total=n_generations, desc="Genetic Algorithm Evolution", leave=False) as pbar:
            for gen in range(n_generations):
                # Select parents
                offspring = toolbox.select(pop, len(pop))
                offspring = [toolbox.clone(ind) for ind in offspring]

                # Apply crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.8:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Apply mutation
                for mutant in offspring:
                    if random.random() < 0.3:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Evaluate invalid individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Select next generation
                pop = toolbox.select(pop + offspring, pop_size)

                # Calculate statistics for visualization
                fit_values = []
                for ind in pop:
                    if ind.fitness.valid:
                        # Sum of all objectives
                        total_fitness = sum(ind.fitness.values)
                        fit_values.append(total_fitness)

                if fit_values:
                    stats = {
                        'best': max(fit_values),
                        'avg': np.mean(fit_values),
                        'worst': min(fit_values)
                    }
                else:
                    stats = {'best': 0, 'avg': 0, 'worst': 0}

                self.viz_manager.save_ga_generation(gen, pop, stats)

                # Progress update
                pbar.set_description(f"Gen {gen}: Best={stats['best']:.3f}, Avg={stats['avg']:.3f}")
                pbar.update(1)

        # Select best solution from Pareto front
        pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

        # Select solution with best balance of objectives and constraints
        best_individual = None
        best_score = -float('inf')

        for ind in pareto_front:
            violations = check_constraints(np.array(ind))
            # Combined score: sum of objectives minus constraint violations
            score = sum(ind.fitness.values) - violations
            if score > best_score:
                best_score = score
                best_individual = ind

        if best_individual is None:
            best_individual = pareto_front[0]

        final_labels = np.array(best_individual)
        final_labels = self._ensure_no_empty_clusters(final_labels, embeddings, n_clusters)

        self.log(f"🎯 CPR optimization completed. Best fitness: {best_score:.3f}")
        return final_labels

    # ------------------------------------------------------------------
    # Equipment Size Related Methods
    # ------------------------------------------------------------------
    def _parse_equipment_size(self, size_str):
        """Parse equipment size string to numerical array"""
        try:
            if isinstance(size_str, str):
                if size_str.startswith('[') and size_str.endswith(']'):
                    parts = size_str.strip('[]').split(',')
                    return [float(p) for p in parts]
                else:
                    parts = size_str.split(',')
                    if len(parts) >= 2:
                        return [float(parts[0]), float(parts[1])]
            elif isinstance(size_str, list) and len(size_str) >= 2:
                return [float(size_str[0]), float(size_str[1])]
            return [1.0, 1.0]  # Default size if parsing fails
        except:
            return [1.0, 1.0]  # Default size if parsing fails

    def _calculate_equipment_area(self, size_list):
        """Calculate total equipment floor area (square meters)"""
        total_area = 0.0
        for size_str in size_list:
            size = self._parse_equipment_size(size_str)
            if len(size) >= 2:
                area = (size[0] / 100) * (size[1] / 100)
                total_area += area
        return total_area

    def _calculate_equipment_area_with_mapping(self, eqp_list, eqp_size_list):
        """Calculate total equipment floor area with proper mapping between equipment and sizes"""
        total_area = 0.0

        # Create equipment to size mapping
        equipment_sizes = {}
        for i, eq_name in enumerate(eqp_list):
            if i < len(eqp_size_list):
                size = self._parse_equipment_size(eqp_size_list[i])
                if len(size) >= 2:
                    area = (size[0] / 100) * (size[1] / 100)
                    # Use the maximum area if the same equipment appears multiple times
                    if eq_name not in equipment_sizes or area > equipment_sizes[eq_name]:
                        equipment_sizes[eq_name] = area

        # Sum up unique equipment areas
        total_area = sum(equipment_sizes.values())
        return total_area

    def _check_size_constraint(self, size_list, max_width=3.5, max_length=7.0):
        """Check if equipment sizes exceed workstation limits"""
        for size_str in size_list:
            size = self._parse_equipment_size(size_str)
            if len(size) >= 2:
                width, length = size[0] / 100, size[1] / 100
                if width > max_width or length > max_length:
                    if length > max_width or width > max_length:
                        return False
        return True

    def _check_size_constraint_with_mapping(self, eqp_list, eqp_size_list, max_width=3.5, max_length=7.0):
        """Check if equipment sizes exceed workstation limits with proper mapping"""
        for i, eq_name in enumerate(eqp_list):
            if i < len(eqp_size_list):
                size = self._parse_equipment_size(eqp_size_list[i])
                if len(size) >= 2:
                    width, length = size[0] / 100, size[1] / 100
                    # Check both orientations
                    if (width > max_width or length > max_length) and \
                            (length > max_width or width > max_length):
                        return False
        return True

    def _estimate_space_efficiency(self, process_indices, df):
        """Estimate space utilization efficiency of a process group"""
        if len(process_indices) == 0:
            return {
                'utilization': 0.0,
                'sharing_ratio': 0.0,
                'total_area': 0.0,
                'equipment_count': 0
            }

        # Collect unique equipment and their areas
        equipment_dict = {}
        total_equipment_instances = 0

        for idx in process_indices:
            row = df.iloc[idx]
            eqp_list = row['eqp_list']
            eqp_size_list = row['eqp_size_list']

            for i, eq_name in enumerate(eqp_list):
                if i < len(eqp_size_list):
                    size = self._parse_equipment_size(eqp_size_list[i])
                    if len(size) >= 2:
                        area = (size[0] / 100) * (size[1] / 100)
                        # Use the maximum area if the same equipment appears multiple times
                        if eq_name not in equipment_dict or area > equipment_dict[eq_name]:
                            equipment_dict[eq_name] = area
                    total_equipment_instances += 1

        total_area = sum(equipment_dict.values())
        utilization = min(1.0, total_area / self.max_workstation_area) if self.max_workstation_area > 0 else 0

        # Calculate sharing ratio (how much equipment is reused)
        unique_equipment_count = len(equipment_dict)
        sharing_ratio = 0
        if total_equipment_instances > 0:
            sharing_ratio = (total_equipment_instances - unique_equipment_count) / total_equipment_instances

        return {
            'utilization': utilization,
            'sharing_ratio': sharing_ratio,
            'total_area': total_area,
            'equipment_count': unique_equipment_count
        }

    # ------------------------------------------------------------------
    # Evaluation and Validation
    # ------------------------------------------------------------------
    def _evaluate_clustering(self, X, labels, df):
        """Evaluate clustering results"""
        metrics = {}

        # Traditional clustering evaluation metrics
        try:
            metrics["silhouette"] = silhouette_score(X, labels)
        except:
            metrics["silhouette"] = -1

        try:
            metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
        except:
            metrics["calinski_harabasz"] = -1

        try:
            metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
        except:
            metrics["davies_bouldin"] = -1

        self.log(f"📊 Clustering metrics: Silhouette={metrics['silhouette']:.3f}, "
                 f"CH={metrics['calinski_harabasz']:.1f}, "
                 f"DB={metrics['davies_bouldin']:.3f}")

        # Space constraint evaluation
        space_metrics = {
            'avg_utilization': 0.0,
            'avg_sharing_ratio': 0.0,
            'clusters_within_limit': 0,
            'total_clusters': 0
        }

        unique_labels = np.unique(labels)
        cluster_stats = []

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]

            space_efficiency = self._estimate_space_efficiency(cluster_indices, df)
            space_metrics['avg_utilization'] += space_efficiency['utilization']
            space_metrics['avg_sharing_ratio'] += space_efficiency['sharing_ratio']

            if space_efficiency['total_area'] <= self.max_workstation_area:
                space_metrics['clusters_within_limit'] += 1

            cluster_stats.append({
                'cluster_id': label,
                'process_count': len(cluster_indices),
                'total_area': space_efficiency['total_area'],
                'utilization': space_efficiency['utilization'],
                'sharing_ratio': space_efficiency['sharing_ratio'],
                'within_limit': space_efficiency['total_area'] <= self.max_workstation_area
            })

        if len(unique_labels) > 0:
            space_metrics['avg_utilization'] /= len(unique_labels)
            space_metrics['avg_sharing_ratio'] /= len(unique_labels)
        space_metrics['total_clusters'] = len(unique_labels)

        self.metrics = metrics
        self.space_metrics = space_metrics

        compliance_rate = space_metrics['clusters_within_limit'] / max(1, space_metrics['total_clusters']) * 100
        self.log(f"📏 Workstation limit satisfaction rate: {compliance_rate:.1f}%")
        self.log(f"📊 Average space utilization: {space_metrics['avg_utilization']:.2%}")

        return metrics, space_metrics, cluster_stats

    # ------------------------------------------------------------------
    # Visualization and Report Generation
    # ------------------------------------------------------------------
    def _visualize_clusters(self, embeddings, labels, df, title="KPC Clustering Result Visualization"):
        """Visualize clustering results"""
        # Dimensionality reduction to 2D
        if embeddings.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            X_2d = tsne.fit_transform(embeddings)
        else:
            X_2d = embeddings

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        plt.figure(figsize=(12, 8))

        # Get area sizes
        sizes = df['equipment_area'].values * 20
        sizes = np.clip(sizes, 20, 200)

        # Draw scatter plot
        cmap = plt.cm.get_cmap('tab10', n_clusters)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if mask.any():  # Only plot if cluster has points
                plt.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    s=sizes[mask],
                    c=[cmap(i)],
                    label=f'Cluster {label} ({np.sum(mask)} processes)',
                    alpha=0.7
                )

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(f"{self.output_dir}/kpc_cluster_visualization.png", dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"📊 Clustering visualization saved: {self.output_dir}/kpc_cluster_visualization.png")

    def _generate_report(self, df, labels, cluster_stats):
        """Generate clustering analysis report"""
        clusters = {}
        for label in np.unique(labels):
            mask = labels == label
            stat = next((s for s in cluster_stats if s.get('cluster_id') == label), {})

            indices = np.where(mask)[0]
            process_info = []
            for idx in indices:
                row = df.iloc[idx]
                process_info.append({
                    'pid': row.get('pid', None),
                    'bm': row.get('bm', ''),
                    'ver': row.get('ver', ''),
                    'equipment': row.get('eqp_list', ''),
                    'material': row.get('mat_list', ''),
                    'worker': row.get('worker_list', ''),
                    'workstation': row.get('ws_list', ''),
                })

            eqp_set = set()
            for idx in indices:
                eqps = df.iloc[idx].get('eqp_list', [])
                if isinstance(eqps, list):
                    eqp_set.update(eqps)

            clusters[str(label)] = {
                "process_ids": process_info,
                "process_count": len(process_info),
                "equipment_list": list(eqp_set),
                "equipment_count": len(eqp_set),
                "total_area": stat.get('total_area', 0),
                "utilization": stat.get('utilization', 0),
                "sharing_ratio": stat.get('sharing_ratio', 0),
                "within_limit": stat.get('within_limit', False)
            }

        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": "KPC (RG-GNN + CK-SC + CPR)",
            "n_clusters": len(clusters),
            "total_processes": len(df),
            "metrics": self.metrics,
            "space_metrics": self.space_metrics,
            "clusters": clusters
        }

        def json_safe(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return str(obj)

        with open(f"{self.output_dir}/kpc_clusters.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=json_safe)

        # Create Excel report
        report_df = pd.DataFrame([
            {
                "Cluster ID": label,
                "Process Count": info["process_count"],
                "Equipment Count": info["equipment_count"],
                "Equipment Area (m²)": f"{info['total_area']:.2f}",
                "Space Utilization": f"{info['utilization']:.2%}",
                "Equipment Sharing Ratio": f"{info['sharing_ratio']:.2%}",
                "Meets Workstation Limit": "Yes" if info["within_limit"] else "No",
            }
            for label, info in clusters.items()
        ])

        report_df.to_excel(f"{self.output_dir}/kpc_cluster_summary.xlsx", index=False)
        return report

    # ------------------------------------------------------------------
    # Constraint Parameter Setting Interface
    # ------------------------------------------------------------------
    def set_process_constraints(self, max_processes_per_cluster=10, max_area_per_cluster=None):
        """
        Set process configuration constraint parameters

        Parameters:
            max_processes_per_cluster: Maximum number of processes per configuration
            max_area_per_cluster: Maximum equipment area per configuration (m²)
        """
        self.max_processes_per_cluster = max_processes_per_cluster

        if max_area_per_cluster is not None:
            self.max_workstation_area = max_area_per_cluster

        self.log(f"⚙️ Constraint settings - Max processes: {self.max_processes_per_cluster}, "
                 f"Max area: {self.max_workstation_area:.1f}m²")

    # ------------------------------------------------------------------
    # Enhanced Main API Method
    # ------------------------------------------------------------------
    def cluster_plan(self, plan: dict[str, int], n_clusters=None, max_processes_per_cluster=10,
                     max_area_per_cluster=None, visual=True):
        """
        Perform knowledge-aware process clustering analysis on production plan - Enhanced visualization version
        """
        start_time = time.time()
        self.log(f"🚀 Starting KPC knowledge-aware process clustering analysis: {plan}")

        # Set constraint parameters
        self.set_process_constraints(max_processes_per_cluster, max_area_per_cluster)

        # 1. Load data
        df = self._load_data(plan)

        # 2. RG-GNN representation learning
        embeddings = self._learn_representations(df)

        # 3. Determine cluster count
        if n_clusters is None:
            min_clusters_needed = len(df) // max_processes_per_cluster + (
                1 if len(df) % max_processes_per_cluster else 0)
            n_clusters = max(min_clusters_needed, 2)
            self.log(
                f"📊 Auto-determined cluster count: {n_clusters} (based on process constraint {max_processes_per_cluster})")

        # 4. CK-SC capacity-constrained spectral clustering
        initial_labels = self._capacity_constrained_spectral_clustering(embeddings, df, n_clusters)

        # 5. CPR collaborative process refinement
        final_labels = self._collaborative_process_refinement(initial_labels, embeddings, df)

        # 6. Evaluate clustering
        metrics, space_metrics, cluster_stats = self._evaluate_clustering(embeddings, final_labels, df)

        # 7. Generate report data
        report = self._generate_report(df, final_labels, cluster_stats)

        # 8. Enhanced visualization
        if visual:
            self.log("🎨 Generating enhanced visualization...")

            # Plot complete algorithm process
            self.viz_manager.plot_complete_algorithm_process(embeddings, initial_labels, final_labels, df)

            # Plot configuration layout overview (separate large image)
            self.viz_manager.plot_configuration_layout_overview(report["clusters"], df)

            # Generate enhanced report
            self.viz_manager.generate_enhanced_report(report["clusters"], metrics, space_metrics)

            # Original visualization
            self._visualize_clusters(embeddings, final_labels, df)

            elapsed = time.time() - start_time
            self.log(f"✅ KPC analysis completed, time elapsed: {elapsed:.2f}s")
            self.log(f"📊 All visualization files saved to: {self.output_dir}")

            print("✅ Enhanced visualization KPC clustering completed!")
            print("📊 Generated visualization files:")
            print("   • complete_algorithm_process.png - Complete algorithm process")
            print("   • configuration_layout_overview.png - Large layout overview")
            print("   • enhanced_analysis_report.html - Enhanced analysis report")
            print("   • kpc_cluster_visualization.png - Final clustering visualization")
            print("   • kpc_clusters.json - Detailed clustering data")
            print("   • kpc_cluster_summary.xlsx - Clustering summary table")

        return report


# Example usage
if __name__ == "__main__":
    def example():
        neo4j_uri = "bolt://localhost:7687"
        neo4j_auth = ("neo4j", "dididaodao")

        plan = {"K3473": 10, "K16842": 5, "K14386": 7, "K1786": 6, "K6286": 12, "K15022":10, "K15126":9}

        clusterer = ProcessClusterer(neo4j_uri, neo4j_auth)

        report = clusterer.cluster_plan(
            plan,
            n_clusters=36,
            max_processes_per_cluster=10,
            max_area_per_cluster=10.0
        )

        print("\n🎯 Clustering Analysis Summary:")
        print(f"   • Total clusters: {report['n_clusters']}")
        print(f"   • Total processes: {report['total_processes']}")
        print(f"   • Average space utilization: {report['space_metrics']['avg_utilization']:.1%}")
        print(
            f"   • Constraint satisfaction rate: {report['space_metrics']['clusters_within_limit']}/{report['space_metrics']['total_clusters']}")


    example()