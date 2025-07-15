# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Matrix Production Layout Optimization with Fixed 9x4 Grid
增强矩阵式生产布局优化系统（固定9×4网格版）- Bug Fixed Version

主要修正：
1. 修复PMX交叉算子的无限循环问题
2. 添加超时保护和错误处理
3. 优化内存使用和计算效率
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any
import json
import time
import random
import os
from dataclasses import dataclass, field
from collections import defaultdict, deque
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import levy
import warnings
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Knowledge graph
try:
    from py2neo import Graph
except ImportError:
    print("Warning: py2neo not installed. Graph operations will be skipped.")
    Graph = None

# Scientific visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set matplotlib parameters for publication quality
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['figure.dpi'] = 100


@dataclass
class ProcessInfo:
    """Process information with enhanced attributes"""
    process_id: str
    process_name: str
    bm_code: str  # Product code
    version: str = ""  # 版本号
    duration: float = 30.0
    setup_time: float = 5.0
    equipment_list: List[str] = field(default_factory=list)
    worker_requirements: List[str] = field(default_factory=list)
    material_requirements: List[str] = field(default_factory=list)
    space_requirement: float = 10.0
    energy_consumption: float = 1.0
    # 工艺序列信息
    sequence_number: int = 0
    next_processes: List[str] = field(default_factory=list)  # 存储完整的下一工序信息
    previous_processes: List[str] = field(default_factory=list)


@dataclass
class Individual:
    """Individual for NSGA-II with fixed grid layout"""
    cluster_assignment: List[int]  # 工艺组态分配到单元的映射（索引表示cluster，值表示unit）
    fitness: List[float] = field(default_factory=list)
    rank: int = 0
    crowding_distance: float = 0.0
    constraint_violation: float = 0.0
    is_feasible: bool = True

    # Additional tracking
    generation: int = 0
    population_id: int = 0
    improvement_history: List[float] = field(default_factory=list)

    def get_config_to_unit(self, cluster_ids: List[str]) -> Dict[str, int]:
        """Convert cluster assignment to config_to_unit dictionary"""
        config_to_unit = {}
        for cluster_idx, unit_id in enumerate(self.cluster_assignment):
            if unit_id >= 0 and cluster_idx < len(cluster_ids):
                config_to_unit[cluster_ids[cluster_idx]] = unit_id
        return config_to_unit


@dataclass
class LogisticsInfo:
    """物流信息类"""
    from_unit: int
    to_unit: int
    from_process: str  # 格式: "process_id|bm|version"
    to_process: str  # 格式: "process_id|bm|version"
    product_type: str
    frequency: float  # 基于订单的频次
    distance: float
    transport_time: float
    cost: float


class FixedGridNSGA2LayoutOptimizer:
    """
    Fixed Grid NSGA-II Layout Optimizer with Knowledge Graph-based Material Flow
    """

    def __init__(self, neo4j_uri: str, neo4j_auth: tuple,
                 layout_config: dict = None,
                 optimization_config: dict = None,
                 output_dir: str = "Cell layout output"):
        """
        Initialize fixed grid NSGA-II optimizer

        Args:
            neo4j_uri: Neo4j database connection URI
            neo4j_auth: Neo4j authentication tuple
            layout_config: Layout configuration (9x4 grid)
            optimization_config: Optimization configuration
            output_dir: Output directory for results
        """
        # Database connection
        if Graph is not None:
            self.graph = Graph(neo4j_uri, auth=neo4j_auth)
        else:
            self.graph = None
            print("⚠️ No database connection available")

        # Layout configuration for 9x4 grid
        default_layout_config = {
            "grid_x": 9,  # X方向9个单元
            "grid_y": 4,  # Y方向4个单元
            "unit_size": 7,  # 单元大小7×7
            "unit_spacing": 5,  # 单元间距5
            "agv_speed": 1.0  # AGV速度 1单位/时间步
        }

        self.layout_config = {**default_layout_config, **(layout_config or {})}

        # 计算固定的单元中心位置
        self.fixed_unit_centers = self._calculate_fixed_unit_centers()
        self.n_units = len(self.fixed_unit_centers)

        # 计算布局边界
        max_x = max(center[0] for center in self.fixed_unit_centers) + self.layout_config["unit_size"] / 2
        max_y = max(center[1] for center in self.fixed_unit_centers) + self.layout_config["unit_size"] / 2
        self.layout_bounds = (0, max_x, 0, max_y)

        self.output_dir = output_dir

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)

        # Enhanced optimization configuration
        default_config = {
            # Multi-population settings
            "num_populations": 3,
            "total_population_size": 150,
            "elite_size": 10,

            # Adaptive parameters
            "initial_crossover_prob": 0.9,
            "initial_mutation_prob": 0.1,
            "adaptive_factor": 0.8,

            # Convergence settings
            "max_generations": 200,
            "convergence_threshold": 1e-6,
            "diversity_threshold": 0.01,
            "stagnation_limit": 20,

            # Objectives weights
            "distance_weight": 1.0,
            "flow_weight": 1.0,
            "space_weight": 0.5,

            # Misc
            "random_seed": 42,
            "verbose": True
        }

        self.config = {**default_config, **(optimization_config or {})}

        # Initialize random seed
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])

        # Data storage
        self.process_data = {}  # 所有工序信息
        self.clusters_data = {}  # 工艺组态信息
        self.cluster_ids = []  # 有序的cluster ID列表
        self.process_sequences = {}  # 工艺序列信息
        self.material_flow_matrix = None
        self.logistics_list = []  # 存储LogisticsInfo对象列表
        self.unit_logistics_matrix = None  # 单元间物流矩阵
        self.order_quantities = {}  # 订单数量

        # 新增：工序映射字典，用于快速查找
        self.process_to_cluster = {}  # process_key -> cluster_id
        self.process_precedence = {}  # process_key -> [next_process_keys]

        # Multi-population storage
        self.populations = []
        self.population_sizes = []
        self.crossover_operators = []

        # Evolution tracking
        self.evolution_history = {
            "best_fitness": [],
            "average_fitness": [],
            "diversity_metrics": [],
            "convergence_metrics": [],
            "constraint_violations": []
        }

        # Optimization results
        self.best_solution = None
        self.pareto_front = []
        self.optimization_time = 0.0

        print(f"🧬 Fixed Grid NSGA-II Layout Optimizer Initialized (Bug Fixed)")
        print(f"📐 Fixed Grid Layout: {self.layout_config['grid_x']}×{self.layout_config['grid_y']}")
        print(f"📦 Unit size: {self.layout_config['unit_size']}×{self.layout_config['unit_size']}")
        print(f"📏 Unit spacing: {self.layout_config['unit_spacing']}")
        print(f"🎯 Total fixed positions: {self.n_units}")
        print(f"📁 Output directory: {output_dir}")

    def _calculate_fixed_unit_centers(self) -> List[Tuple[float, float]]:
        """计算固定的9×4网格单元中心位置"""
        grid_x = self.layout_config["grid_x"]
        grid_y = self.layout_config["grid_y"]
        unit_size = self.layout_config["unit_size"]
        unit_spacing = self.layout_config["unit_spacing"]

        unit_centers = []

        for row in range(grid_y):
            for col in range(grid_x):
                # 计算单元中心位置
                x = col * (unit_size + unit_spacing) + unit_size / 2
                y = row * (unit_size + unit_spacing) + unit_size / 2
                unit_centers.append((x, y))

        return unit_centers

    def load_clustering_results(self, clusters_file: str):
        """Load process clustering results and build process mapping"""
        print(f"📁 Loading clustering results from: {clusters_file}")

        try:
            with open(clusters_file, 'r', encoding='utf-8') as f:
                clustering_result = json.load(f)

            self.clusters_data = clustering_result.get('clusters', {})
            self.cluster_ids = list(self.clusters_data.keys())

            print(f"✅ Loaded {len(self.clusters_data)} process clusters")

            # 构建工序到工艺组态的映射
            self.process_to_cluster = {}
            total_processes = 0

            for cluster_id, cluster_info in self.clusters_data.items():
                process_count = len(cluster_info['process_ids'])
                equipment_count = len(cluster_info.get('equipment_list', []))

                # 遍历每个工艺组态中的工序
                for process_info in cluster_info['process_ids']:
                    pid = str(process_info['pid'])
                    bm = process_info.get('bm', 'Unknown')
                    version = process_info.get('ver', process_info.get('version', ''))

                    # 创建工序键值（用于唯一标识工序）
                    process_key = f"{pid}|{bm}|{version}"
                    self.process_to_cluster[process_key] = cluster_id
                    total_processes += 1

                print(f"   Cluster {cluster_id}: {process_count} processes, {equipment_count} equipment")

            print(f"📊 Total processes in clusters: {total_processes}")
            print(f"📊 Process to cluster mapping created: {len(self.process_to_cluster)} entries")

            # 检查cluster数量是否超过单元数量
            if len(self.cluster_ids) > self.n_units:
                print(f"⚠️ Warning: {len(self.cluster_ids)} clusters but only {self.n_units} units available!")

            # 加载工序紧邻关系
            self.load_process_precedence_from_graph()

            # 计算物流关系
            self.calculate_logistics_from_precedence()

        except Exception as e:
            print(f"⚠️ Error loading clustering results: {e}")
            self._create_default_clusters()

    def load_process_precedence_from_graph(self):
        """从知识图谱加载工序的紧邻关系"""
        print("🔗 Loading process precedence relationships from knowledge graph...")

        if self.graph is None:
            print("⚠️ No graph connection, using default data")
            self._create_default_precedence()
            return

        try:
            # 查询所有PRECEDES关系
            precedence_query = """
            MATCH (p1:Process)-[:PRECEDES]->(p2:Process)
            RETURN p1.process_id as from_id, 
                   p1.product_bm as from_bm,
                   p1.product_version as from_version,
                   p2.process_id as to_id,
                   p2.product_bm as to_bm,
                   p2.product_version as to_version
            """

            results = self.graph.run(precedence_query).data()

            # 构建工序紧邻关系字典
            self.process_precedence = defaultdict(list)
            precedence_count = 0

            for record in results:
                from_process_key = f"{record['from_id']}|{record['from_bm']}|{record['from_version'] or ''}"
                to_process_key = f"{record['to_id']}|{record['to_bm']}|{record['to_version'] or ''}"

                self.process_precedence[from_process_key].append(to_process_key)
                precedence_count += 1

            print(f"✅ Loaded {precedence_count} precedence relationships")
            print(f"📊 Processes with next processes: {len(self.process_precedence)}")

            # 加载订单信息
            self.load_order_quantities()

        except Exception as e:
            print(f"⚠️ Error loading precedence relationships: {e}")
            self._create_default_precedence()

    def load_order_quantities(self):
        """加载订单数量信息"""
        print("📋 Loading order quantities...")

        try:
            if self.graph:
                # 尝试从知识图谱查询订单信息
                order_query = """
                MATCH (o:Order)-[:ORDERS]->(p:Product)
                RETURN p.name as product_id,
                       sum(o.quantity) as total_quantity
                """

                results = self.graph.run(order_query).data()

                for record in results:
                    product_id = str(record['product_id'])
                    quantity = int(record.get('total_quantity', 0))
                    if quantity > 0:
                        self.order_quantities[product_id] = quantity

            # 如果没有订单数据，使用默认值
            if not self.order_quantities:
                print("⚠️ No order data found, using default quantities")
                # 从clusters中提取所有产品类型
                products = set()
                for cluster_info in self.clusters_data.values():
                    for process_info in cluster_info['process_ids']:
                        products.add(process_info.get('bm', 'Unknown'))

                # 为每个产品分配默认订单量
                for product in products:
                    if product != 'Unknown':
                        self.order_quantities[product] = random.randint(20, 100)

            print(f"✅ Order quantities loaded for {len(self.order_quantities)} products")
            for product, qty in sorted(self.order_quantities.items())[:5]:
                print(f"   {product}: {qty} units")
            if len(self.order_quantities) > 5:
                print(f"   ... and {len(self.order_quantities) - 5} more products")

        except Exception as e:
            print(f"⚠️ Error loading order quantities: {e}")
            self._create_default_order_quantities()

    def calculate_logistics_from_precedence(self):
        """基于工序紧邻关系计算单元间的物流"""
        print("🚚 Calculating logistics from process precedence...")

        self.logistics_list = []
        logistics_summary = defaultdict(lambda: defaultdict(float))

        # 遍历所有工序的紧邻关系
        for from_process_key, next_processes in self.process_precedence.items():
            # 获取起始工序所在的工艺组态（单元）
            from_cluster = self.process_to_cluster.get(from_process_key)
            if not from_cluster:
                continue

            # 解析工序信息
            from_parts = from_process_key.split('|')
            from_bm = from_parts[1] if len(from_parts) > 1 else 'Unknown'

            # 获取该产品的订单数量
            order_qty = self.order_quantities.get(from_bm, 0)
            if order_qty == 0:
                continue

            # 检查每个后继工序
            for to_process_key in next_processes:
                to_cluster = self.process_to_cluster.get(to_process_key)
                if not to_cluster:
                    continue

                # 如果后继工序在不同的单元，记录物流
                if from_cluster != to_cluster:
                    # 累加物流频次
                    logistics_summary[(from_cluster, to_cluster)][from_bm] += order_qty

        # 将累计的物流转换为LogisticsInfo对象
        for (from_cluster, to_cluster), product_flows in logistics_summary.items():
            total_flow = sum(product_flows.values())

            # 创建物流信息对象（此时还没有距离信息）
            logistics_info = LogisticsInfo(
                from_unit=-1,  # 将在布局时更新
                to_unit=-1,  # 将在布局时更新
                from_process=from_cluster,  # 暂时存储cluster id
                to_process=to_cluster,  # 暂时存储cluster id
                product_type=','.join(product_flows.keys()),  # 涉及的产品类型
                frequency=total_flow,
                distance=0.0,  # 将在布局时计算
                transport_time=0.0,  # 将在布局时计算
                cost=0.0  # 将在布局时计算
            )
            self.logistics_list.append(logistics_info)

        print(f"✅ Calculated {len(self.logistics_list)} logistics relationships")

        # 打印物流统计
        total_flow = sum(log.frequency for log in self.logistics_list)
        print(f"📊 Total material flow: {total_flow}")

        # 打印前5个最大的物流关系
        sorted_logistics = sorted(self.logistics_list, key=lambda x: x.frequency, reverse=True)
        print("🔝 Top 5 logistics relationships:")
        for i, log in enumerate(sorted_logistics[:5]):
            print(f"   {i + 1}. {log.from_process} → {log.to_process}: {log.frequency} units ({log.product_type})")

    def calculate_material_flow_matrix(self, config_to_unit: Dict[str, int]) -> np.ndarray:
        """基于物流关系计算单元间的物流强度矩阵"""
        flow_matrix = np.zeros((self.n_units, self.n_units))

        # 更新物流信息中的单元ID并计算流量矩阵
        for logistics in self.logistics_list:
            from_cluster = logistics.from_process  # 这里存储的是cluster id
            to_cluster = logistics.to_process

            # 获取对应的单元ID
            if from_cluster in config_to_unit and to_cluster in config_to_unit:
                from_unit = config_to_unit[from_cluster]
                to_unit = config_to_unit[to_cluster]

                # 检查单元ID有效性
                if 0 <= from_unit < self.n_units and 0 <= to_unit < self.n_units:
                    # 更新物流信息
                    logistics.from_unit = from_unit
                    logistics.to_unit = to_unit

                    # 计算距离
                    from_center = self.fixed_unit_centers[from_unit]
                    to_center = self.fixed_unit_centers[to_unit]
                    distance = np.linalg.norm(np.array(from_center) - np.array(to_center))

                    logistics.distance = distance
                    logistics.transport_time = distance / self.layout_config["agv_speed"]
                    logistics.cost = logistics.frequency * distance * 0.1

                    # 累加流量
                    flow_matrix[from_unit, to_unit] += logistics.frequency

        # 存储单元物流矩阵
        self.unit_logistics_matrix = flow_matrix

        return flow_matrix

    def _create_default_precedence(self):
        """创建默认的工序紧邻关系"""
        print("🔧 Creating default precedence relationships...")

        # 基于clusters中的工序创建简单的线性序列
        self.process_precedence = defaultdict(list)

        # 按产品分组工序
        product_processes = defaultdict(list)

        for cluster_info in self.clusters_data.values():
            for process_info in cluster_info['process_ids']:
                pid = str(process_info['pid'])
                bm = process_info.get('bm', 'Unknown')
                version = process_info.get('ver', '')
                process_key = f"{pid}|{bm}|{version}"
                product_processes[bm].append((process_key, int(pid)))

        # 为每个产品创建工序序列
        for product, processes in product_processes.items():
            # 按工序号排序
            sorted_processes = sorted(processes, key=lambda x: x[1])

            # 创建紧邻关系
            for i in range(len(sorted_processes) - 1):
                from_key = sorted_processes[i][0]
                to_key = sorted_processes[i + 1][0]
                self.process_precedence[from_key].append(to_key)

        print(f"✅ Created default precedence for {len(self.process_precedence)} processes")

    def _create_default_order_quantities(self):
        """创建默认订单数量"""
        # 从clusters中提取产品
        products = set()
        for cluster_info in self.clusters_data.values():
            for process_info in cluster_info['process_ids']:
                products.add(process_info.get('bm', 'Unknown'))

        # 分配默认订单量
        for product in products:
            if product != 'Unknown':
                self.order_quantities[product] = random.randint(30, 80)

    def _create_default_clusters(self):
        """创建默认工艺组态用于测试"""
        print("🔧 Creating default clusters...")

        # 创建示例工艺组态
        self.clusters_data = {
            "C01": {
                "process_ids": [
                    {"pid": "10", "bm": "K3473", "ver": "V1"},
                    {"pid": "20", "bm": "K3473", "ver": "V1"},
                    {"pid": "30", "bm": "K16842", "ver": "V1"}
                ],
                "equipment_list": ["E001", "E002"],
                "worker_list": ["W001"]
            },
            "C02": {
                "process_ids": [
                    {"pid": "40", "bm": "K3473", "ver": "V1"},
                    {"pid": "50", "bm": "K16842", "ver": "V1"},
                    {"pid": "60", "bm": "K16842", "ver": "V1"}
                ],
                "equipment_list": ["E003", "E004"],
                "worker_list": ["W002"]
            },
            "C03": {
                "process_ids": [
                    {"pid": "70", "bm": "K14386", "ver": "V1"},
                    {"pid": "80", "bm": "K14386", "ver": "V1"}
                ],
                "equipment_list": ["E005"],
                "worker_list": ["W003"]
            },
            "C04": {
                "process_ids": [
                    {"pid": "90", "bm": "K14386", "ver": "V1"},
                    {"pid": "100", "bm": "K1786", "ver": "V1"}
                ],
                "equipment_list": ["E006", "E007"],
                "worker_list": ["W004"]
            }
        }

        self.cluster_ids = list(self.clusters_data.keys())

        # 创建默认紧邻关系
        self._create_default_precedence()

        # 创建默认订单
        self._create_default_order_quantities()

        # 计算物流
        self.calculate_logistics_from_precedence()

    def calculate_objective_functions(self, individual: Individual) -> List[float]:
        """Calculate objective functions for fixed grid layout"""
        try:
            config_to_unit = individual.get_config_to_unit(self.cluster_ids)

            # 计算当前配置的物流矩阵
            flow_matrix = self.calculate_material_flow_matrix(config_to_unit)

            # Objective 1: Material handling cost (物流成本)
            material_cost = self._calculate_material_handling_cost(flow_matrix)

            # Objective 2: Space utilization balance (空间利用均衡度)
            space_balance = self._calculate_space_balance(config_to_unit)

            # Objective 3: Energy consumption (能耗)
            energy_cost = self._calculate_energy_cost(config_to_unit)

            # Convert to minimization problems
            objectives = [
                material_cost * self.config["flow_weight"],
                space_balance * self.config["space_weight"],
                energy_cost
            ]

            return objectives

        except Exception as e:
            print(f"⚠️ Error in calculate_objective_functions: {e}")
            return [float('inf')] * 3

    def _calculate_material_handling_cost(self, flow_matrix: np.ndarray) -> float:
        """计算物流成本"""
        total_cost = 0.0

        # 基于物流列表计算总成本
        for logistics in self.logistics_list:
            if logistics.from_unit >= 0 and logistics.to_unit >= 0:
                total_cost += logistics.cost

        return total_cost if total_cost > 0 else 1000.0

    def _calculate_space_balance(self, config_to_unit: Dict[str, int]) -> float:
        """计算空间利用均衡度（希望各单元负载均衡）"""
        # 计算每个单元的负载
        unit_loads = np.zeros(self.n_units)

        for cluster_id, unit_id in config_to_unit.items():
            if cluster_id in self.clusters_data and 0 <= unit_id < self.n_units:
                cluster_info = self.clusters_data[cluster_id]

                # 基于工序数量和订单计算负载
                load = 0
                for process_info in cluster_info['process_ids']:
                    bm = process_info.get('bm', 'Unknown')
                    order_qty = self.order_quantities.get(bm, 0)
                    load += order_qty

                unit_loads[unit_id] = load

        # 计算负载的标准差（越小越均衡）
        used_loads = unit_loads[unit_loads > 0]
        if len(used_loads) > 1:
            load_std = np.std(used_loads)
            avg_load = np.mean(used_loads)
            balance_score = load_std / avg_load if avg_load > 0 else 0
        else:
            balance_score = 0.0

        return balance_score

    def _calculate_energy_cost(self, config_to_unit: Dict[str, int]) -> float:
        """Calculate total energy consumption cost"""
        total_energy = 0.0

        for cluster_id in config_to_unit.keys():
            if cluster_id in self.clusters_data:
                cluster_info = self.clusters_data[cluster_id]

                # 基于工序和订单计算能耗
                for process_info in cluster_info['process_ids']:
                    bm = process_info.get('bm', 'Unknown')
                    order_qty = self.order_quantities.get(bm, 0)

                    # 简化的能耗计算
                    energy = 30.0 * order_qty * 0.01  # 基础能耗
                    total_energy += energy

        return total_energy if total_energy > 0 else 100.0

    def initialize_multi_populations(self):
        """Initialize multiple populations with fixed grid layout"""
        print("👥 Initializing multi-population NSGA-II for fixed 9×4 grid...")

        n_populations = self.config["num_populations"]
        total_size = self.config["total_population_size"]

        # Distribute population sizes
        base_size = total_size // n_populations
        remaining = total_size % n_populations

        self.population_sizes = [base_size] * n_populations
        for i in range(remaining):
            self.population_sizes[i] += 1

        # Initialize populations
        self.populations = []
        for i in range(n_populations):
            population = self._create_initial_population(self.population_sizes[i], i)
            self.populations.append(population)

        # Assign different crossover operators
        self.crossover_operators = [
            'pmx',  # Partially Mapped Crossover
            'ox',  # Order Crossover
            'cx'  # Cycle Crossover
        ]

        print(f"✅ Initialized {n_populations} populations with sizes: {self.population_sizes}")

    def _create_initial_population(self, pop_size: int, population_id: int) -> List[Individual]:
        """Create initial population with fixed grid layout"""
        population = []

        n_clusters = len(self.cluster_ids)

        # 只使用需要的单元数量
        used_units = min(n_clusters, self.n_units)

        for i in range(pop_size):
            # 创建分配方案
            if population_id == 0:
                # 顺序分配
                assignment = list(range(used_units)) + [-1] * (n_clusters - used_units)
            elif population_id == 1:
                # 随机分配
                assignment = list(range(used_units)) + [-1] * (n_clusters - used_units)
                random.shuffle(assignment)
            else:
                # 基于物流强度的启发式分配
                assignment = self._heuristic_assignment()

            # Create individual
            individual = Individual(
                cluster_assignment=assignment,
                generation=0,
                population_id=population_id
            )

            # Calculate fitness
            individual.fitness = self.calculate_objective_functions(individual)
            population.append(individual)

        return population

    def _heuristic_assignment(self) -> List[int]:
        """基于物流强度的启发式分配"""
        n_clusters = len(self.cluster_ids)
        used_units = min(n_clusters, self.n_units)

        # 计算每个cluster的物流强度
        cluster_flow_intensity = defaultdict(float)

        for logistics in self.logistics_list:
            cluster_flow_intensity[logistics.from_process] += logistics.frequency
            cluster_flow_intensity[logistics.to_process] += logistics.frequency

        # 按物流强度排序clusters
        sorted_clusters = sorted(self.cluster_ids,
                                 key=lambda x: cluster_flow_intensity.get(x, 0),
                                 reverse=True)

        # 将高物流强度的cluster分配到中心位置
        assignment = [-1] * n_clusters

        # 计算每个单元到中心的距离
        center_x = np.mean([c[0] for c in self.fixed_unit_centers])
        center_y = np.mean([c[1] for c in self.fixed_unit_centers])

        unit_distances = []
        for i, center in enumerate(self.fixed_unit_centers[:used_units]):
            dist = np.linalg.norm(np.array(center) - np.array([center_x, center_y]))
            unit_distances.append((i, dist))

        # 按距离排序单元
        unit_distances.sort(key=lambda x: x[1])

        # 分配
        for i, cluster_id in enumerate(sorted_clusters[:used_units]):
            cluster_idx = self.cluster_ids.index(cluster_id)
            unit_id = unit_distances[i][0]
            assignment[cluster_idx] = unit_id

        return assignment

    def crossover(self, parent1: Individual, parent2: Individual,
                  crossover_type: str) -> Tuple[Individual, Individual]:
        """Perform crossover operation for permutation encoding"""
        try:
            if crossover_type == 'pmx':
                return self._pmx_crossover_fixed(parent1, parent2)
            elif crossover_type == 'ox':
                return self._ox_crossover(parent1, parent2)
            elif crossover_type == 'cx':
                return self._cx_crossover(parent1, parent2)
            else:
                return parent1, parent2
        except Exception as e:
            print(f"⚠️ Crossover error ({crossover_type}): {e}")
            # 返回父代副本作为后代
            return (Individual(cluster_assignment=parent1.cluster_assignment.copy()),
                    Individual(cluster_assignment=parent2.cluster_assignment.copy()))

    def _pmx_crossover_fixed(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """修复的PMX交叉算子，避免无限循环"""
        size = len(parent1.cluster_assignment)
        child1_assignment = parent1.cluster_assignment.copy()
        child2_assignment = parent2.cluster_assignment.copy()

        # 选择两个交叉点
        if size <= 2:
            return (Individual(cluster_assignment=child1_assignment),
                    Individual(cluster_assignment=child2_assignment))

        cx_point1 = random.randint(0, size - 2)
        cx_point2 = random.randint(cx_point1 + 1, size - 1)

        # 交换片段
        segment1 = child1_assignment[cx_point1:cx_point2].copy()
        segment2 = child2_assignment[cx_point1:cx_point2].copy()

        child1_assignment[cx_point1:cx_point2] = segment2
        child2_assignment[cx_point1:cx_point2] = segment1

        # 修复child1的冲突
        for i in list(range(0, cx_point1)) + list(range(cx_point2, size)):
            if child1_assignment[i] != -1 and child1_assignment[i] in segment2:
                # 找到映射
                attempts = 0
                current_val = child1_assignment[i]
                visited = set()

                while current_val in segment2 and attempts < size:
                    if current_val in visited:
                        # 检测到循环，使用未使用的值
                        used_values = set(child1_assignment)
                        available = [v for v in range(self.n_units) if v not in used_values]
                        if available:
                            child1_assignment[i] = available[0]
                        else:
                            child1_assignment[i] = -1
                        break

                    visited.add(current_val)
                    idx = segment2.index(current_val)
                    current_val = segment1[idx]
                    attempts += 1

                if attempts < size and current_val not in segment2:
                    child1_assignment[i] = current_val

        # 修复child2的冲突
        for i in list(range(0, cx_point1)) + list(range(cx_point2, size)):
            if child2_assignment[i] != -1 and child2_assignment[i] in segment1:
                # 找到映射
                attempts = 0
                current_val = child2_assignment[i]
                visited = set()

                while current_val in segment1 and attempts < size:
                    if current_val in visited:
                        # 检测到循环，使用未使用的值
                        used_values = set(child2_assignment)
                        available = [v for v in range(self.n_units) if v not in used_values]
                        if available:
                            child2_assignment[i] = available[0]
                        else:
                            child2_assignment[i] = -1
                        break

                    visited.add(current_val)
                    idx = segment1.index(current_val)
                    current_val = segment2[idx]
                    attempts += 1

                if attempts < size and current_val not in segment1:
                    child2_assignment[i] = current_val

        # 创建后代
        offspring1 = Individual(cluster_assignment=child1_assignment)
        offspring2 = Individual(cluster_assignment=child2_assignment)

        return offspring1, offspring2

    def _ox_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Order Crossover (OX)"""
        size = len(parent1.cluster_assignment)
        child1_assignment = [-1] * size
        child2_assignment = [-1] * size

        # 选择两个交叉点
        if size <= 2:
            return (Individual(cluster_assignment=parent1.cluster_assignment.copy()),
                    Individual(cluster_assignment=parent2.cluster_assignment.copy()))

        cx_point1 = random.randint(0, size - 2)
        cx_point2 = random.randint(cx_point1 + 1, size - 1)

        # 复制父代片段
        child1_assignment[cx_point1:cx_point2] = parent1.cluster_assignment[cx_point1:cx_point2]
        child2_assignment[cx_point1:cx_point2] = parent2.cluster_assignment[cx_point1:cx_point2]

        # 填充剩余位置
        def fill_remaining(child, parent):
            used = set(child[cx_point1:cx_point2])
            j = cx_point2
            for i in range(cx_point2, cx_point2 + size):
                idx = i % size
                val = parent[idx]
                if val not in used and val != -1:
                    child[j % size] = val
                    used.add(val)
                    j += 1

        fill_remaining(child1_assignment, parent2.cluster_assignment)
        fill_remaining(child2_assignment, parent1.cluster_assignment)

        # 创建后代
        offspring1 = Individual(cluster_assignment=child1_assignment)
        offspring2 = Individual(cluster_assignment=child2_assignment)

        return offspring1, offspring2

    def _cx_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Cycle Crossover (CX)"""
        size = len(parent1.cluster_assignment)
        child1_assignment = parent1.cluster_assignment.copy()
        child2_assignment = parent2.cluster_assignment.copy()

        # 找到所有循环
        visited = [False] * size
        cycles = []

        for i in range(size):
            if not visited[i] and parent1.cluster_assignment[i] != -1:
                cycle = []
                j = i
                attempts = 0

                while j not in cycle and attempts < size:
                    cycle.append(j)
                    visited[j] = True
                    # 找到parent2中与parent1[j]相同的位置
                    val = parent1.cluster_assignment[j]
                    next_j = None

                    for k in range(size):
                        if parent2.cluster_assignment[k] == val:
                            next_j = k
                            break

                    if next_j is None or next_j == cycle[0]:
                        break

                    j = next_j
                    attempts += 1

                if len(cycle) > 0:
                    cycles.append(cycle)

        # 交替交换循环
        for i, cycle in enumerate(cycles):
            if i % 2 == 1:
                for idx in cycle:
                    child1_assignment[idx], child2_assignment[idx] = \
                        child2_assignment[idx], child1_assignment[idx]

        # 创建后代
        offspring1 = Individual(cluster_assignment=child1_assignment)
        offspring2 = Individual(cluster_assignment=child2_assignment)

        return offspring1, offspring2

    def mutate(self, individual: Individual, mutation_prob: float) -> Individual:
        """Mutation operator for permutation encoding"""
        mutated = Individual(cluster_assignment=individual.cluster_assignment.copy())

        size = len(mutated.cluster_assignment)

        # 交换变异
        if random.random() < mutation_prob:
            # 找到两个有效位置
            valid_indices = [i for i in range(size) if mutated.cluster_assignment[i] != -1]

            if len(valid_indices) >= 2:
                idx1, idx2 = random.sample(valid_indices, 2)
                # 交换
                mutated.cluster_assignment[idx1], mutated.cluster_assignment[idx2] = \
                    mutated.cluster_assignment[idx2], mutated.cluster_assignment[idx1]

        return mutated

    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """Fast non-dominated sorting algorithm"""
        fronts = [[]]

        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []

            for other in population:
                if self._dominates(individual.fitness, other.fitness):
                    individual.dominated_solutions.append(other)
                elif self._dominates(other.fitness, individual.fitness):
                    individual.domination_count += 1

            if individual.domination_count == 0:
                individual.rank = 0
                fronts[0].append(individual)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for individual in fronts[i]:
                for dominated in individual.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        dominated.rank = i + 1
                        next_front.append(dominated)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remove empty last front

    def _dominates(self, fitness1: List[float], fitness2: List[float]) -> bool:
        """Check if fitness1 dominates fitness2 (Pareto dominance)"""
        better_in_at_least_one = False

        for f1, f2 in zip(fitness1, fitness2):
            if f1 > f2:  # Worse in this objective (minimization)
                return False
            elif f1 < f2:  # Better in this objective
                better_in_at_least_one = True

        return better_in_at_least_one

    def calculate_crowding_distance(self, front: List[Individual]):
        """Calculate crowding distance for individuals in a front"""
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return

        # Initialize crowding distances
        for individual in front:
            individual.crowding_distance = 0

        n_objectives = len(front[0].fitness)

        for obj_idx in range(n_objectives):
            # Sort by objective value
            front.sort(key=lambda x: x.fitness[obj_idx])

            # Set boundary points to infinity
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Calculate crowding distance for intermediate points
            obj_range = front[-1].fitness[obj_idx] - front[0].fitness[obj_idx]

            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    distance = (front[i + 1].fitness[obj_idx] - front[i - 1].fitness[obj_idx]) / obj_range
                    front[i].crowding_distance += distance

    def environmental_selection(self, population: List[Individual],
                                population_size: int) -> List[Individual]:
        """Environmental selection using non-dominated sorting and crowding distance"""
        fronts = self.fast_non_dominated_sort(population)

        new_population = []
        front_idx = 0

        # Add complete fronts
        while front_idx < len(fronts) and len(new_population) + len(fronts[front_idx]) <= population_size:
            self.calculate_crowding_distance(fronts[front_idx])
            new_population.extend(fronts[front_idx])
            front_idx += 1

        # Add partial front if needed
        if front_idx < len(fronts) and len(new_population) < population_size:
            remaining_slots = population_size - len(new_population)
            self.calculate_crowding_distance(fronts[front_idx])

            # Sort by crowding distance (descending)
            fronts[front_idx].sort(key=lambda x: x.crowding_distance, reverse=True)
            new_population.extend(fronts[front_idx][:remaining_slots])

        return new_population

    def calculate_diversity_metric(self, population: List[Individual]) -> float:
        """Calculate population diversity metric"""
        if len(population) < 2:
            return 0.0

        # Calculate diversity in objective space
        objective_matrix = np.array([ind.fitness for ind in population])

        # Normalize objectives
        obj_min = np.min(objective_matrix, axis=0)
        obj_max = np.max(objective_matrix, axis=0)
        obj_range = obj_max - obj_min

        # Avoid division by zero
        obj_range[obj_range == 0] = 1.0

        normalized_objectives = (objective_matrix - obj_min) / obj_range

        # Calculate average pairwise distance
        distances = cdist(normalized_objectives, normalized_objectives)

        # Calculate diversity as average distance
        n_individuals = len(population)
        total_distance = np.sum(distances) / 2  # Divide by 2 for symmetric matrix
        avg_distance = total_distance / (n_individuals * (n_individuals - 1) / 2)

        return avg_distance

    def optimize_layout(self) -> Dict[str, Any]:
        """Run the NSGA-II optimization for fixed grid layout with bug fixes"""
        print("🚀 Starting Fixed Grid NSGA-II Layout Optimization (Bug Fixed Version)...")

        start_time = time.time()

        # Initialize multi-populations
        try:
            self.initialize_multi_populations()
        except Exception as e:
            print(f"❌ Error initializing populations: {e}")
            return {}

        # Evolution loop
        stagnation_counter = 0
        previous_best_fitness = float('inf')

        for generation in range(self.config["max_generations"]):
            print(f"\n🧬 Generation {generation + 1}/{self.config['max_generations']}")

            # Process each population
            for pop_idx, population in enumerate(self.populations):
                if not population:
                    print(f"⚠️ Empty population {pop_idx + 1}, skipping...")
                    continue

                crossover_type = self.crossover_operators[pop_idx % len(self.crossover_operators)]

                # 自适应参数
                progress = generation / self.config["max_generations"]
                crossover_prob = self.config["initial_crossover_prob"] * (1 - progress * 0.5)
                mutation_prob = self.config["initial_mutation_prob"] * (1 + progress)

                print(f"  Population {pop_idx + 1}: {crossover_type} crossover, "
                      f"pc={crossover_prob:.3f}, pm={mutation_prob:.3f}")

                # Create offspring
                offspring = []
                pop_size = len(population)
                max_attempts = pop_size * 3  # 限制尝试次数

                attempts = 0
                while len(offspring) < pop_size and attempts < max_attempts:
                    try:
                        # Tournament selection
                        parent1 = self._tournament_selection(population)
                        parent2 = self._tournament_selection(population)

                        # Crossover
                        if np.random.random() < crossover_prob:
                            child1, child2 = self.crossover(parent1, parent2, crossover_type)
                        else:
                            child1 = Individual(cluster_assignment=parent1.cluster_assignment.copy())
                            child2 = Individual(cluster_assignment=parent2.cluster_assignment.copy())

                        # Mutation
                        child1 = self.mutate(child1, mutation_prob)
                        child2 = self.mutate(child2, mutation_prob)

                        # Evaluate offspring
                        child1.fitness = self.calculate_objective_functions(child1)
                        child2.fitness = self.calculate_objective_functions(child2)
                        child1.generation = generation
                        child2.generation = generation
                        child1.population_id = pop_idx
                        child2.population_id = pop_idx

                        offspring.extend([child1, child2])

                    except Exception as e:
                        print(f"⚠️ Error creating offspring: {e}")
                        attempts += 1
                        continue

                    attempts += 1

                if len(offspring) < pop_size:
                    print(f"⚠️ Only created {len(offspring)}/{pop_size} offspring after {attempts} attempts")

                # Environmental selection
                try:
                    combined_population = population + offspring[:pop_size]
                    self.populations[pop_idx] = self.environmental_selection(
                        combined_population, self.population_sizes[pop_idx]
                    )
                except Exception as e:
                    print(f"⚠️ Error in environmental selection: {e}")
                    self.populations[pop_idx] = population  # 保留原种群

            # Track evolution progress
            try:
                self._track_evolution_progress(generation)
            except Exception as e:
                print(f"⚠️ Error tracking progress: {e}")

            # Check convergence
            all_individuals = []
            for pop in self.populations:
                if pop:
                    all_individuals.extend(pop)

            if all_individuals:
                current_best_fitness = min(ind.fitness[0] for ind in all_individuals)

                if abs(current_best_fitness - previous_best_fitness) < self.config["convergence_threshold"]:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0

                previous_best_fitness = current_best_fitness

                # Print progress
                if generation % 10 == 0 or stagnation_counter > self.config["stagnation_limit"]:
                    self._print_progress(generation, all_individuals)

                # Early stopping
                if stagnation_counter > self.config["stagnation_limit"]:
                    print(f"🛑 Early stopping at generation {generation + 1} due to stagnation")
                    break

        # Finalize optimization
        self.optimization_time = time.time() - start_time

        # Extract final results
        final_population = []
        for pop in self.populations:
            if pop:
                final_population.extend(pop)

        if not final_population:
            print("❌ No valid solutions found!")
            return {}

        # Find Pareto front
        fronts = self.fast_non_dominated_sort(final_population)
        self.pareto_front = fronts[0] if fronts else []

        # Select best solution
        if self.pareto_front:
            # Select solution with best first objective
            self.best_solution = min(self.pareto_front, key=lambda x: x.fitness[0])
        else:
            self.best_solution = min(final_population, key=lambda x: x.fitness[0])

        print(f"\n✅ Optimization completed in {self.optimization_time:.2f} seconds")
        print(f"🏆 Pareto front size: {len(self.pareto_front)}")
        print(f"🎯 Best solution fitness: {[f'{f:.3f}' for f in self.best_solution.fitness]}")

        # Prepare results
        results = {
            "best_solution": self.best_solution,
            "pareto_front": self.pareto_front,
            "optimization_time": self.optimization_time,
            "evolution_history": self.evolution_history,
            "config": self.config,
            "final_population": final_population
        }

        return results

    def _tournament_selection(self, population: List[Individual], tournament_size: int = 3) -> Individual:
        """Tournament selection"""
        if not population:
            raise ValueError("Empty population for tournament selection")

        candidates = random.sample(population, min(tournament_size, len(population)))

        # Select best candidate based on rank and crowding distance
        best = candidates[0]
        for candidate in candidates[1:]:
            if (candidate.rank < best.rank or
                    (candidate.rank == best.rank and candidate.crowding_distance > best.crowding_distance)):
                best = candidate

        return best

    def _track_evolution_progress(self, generation: int):
        """Track evolution progress"""
        all_individuals = []
        for pop in self.populations:
            if pop:
                all_individuals.extend(pop)

        if not all_individuals:
            return

        # Best fitness
        best_fitness = min(ind.fitness[0] for ind in all_individuals)
        self.evolution_history["best_fitness"].append(best_fitness)

        # Average fitness
        avg_fitness = np.mean([ind.fitness[0] for ind in all_individuals])
        self.evolution_history["average_fitness"].append(avg_fitness)

        # Diversity metric
        diversity = self.calculate_diversity_metric(all_individuals)
        self.evolution_history["diversity_metrics"].append(diversity)

    def _print_progress(self, generation: int, population: List[Individual]):
        """Print optimization progress"""
        if not population:
            return

        best_fitness = min(ind.fitness[0] for ind in population)
        avg_fitness = np.mean([ind.fitness[0] for ind in population])
        diversity = self.calculate_diversity_metric(population)

        print(f"  📊 Generation {generation + 1}:")
        print(f"     Best logistics cost: {best_fitness:.3f}")
        print(f"     Avg logistics cost: {avg_fitness:.3f}")
        print(f"     Diversity: {diversity:.3f}")

    def visualize_layout_solution(self, solution: Individual = None,
                                  save_plots: bool = True) -> Dict[str, Any]:
        """Enhanced visualization of fixed 9x4 grid layout solution"""
        if solution is None:
            solution = self.best_solution

        if solution is None:
            print("⚠️ No solution to visualize")
            return {}

        print("🎨 Generating fixed 9×4 grid layout visualizations...")

        # Get config mapping
        config_to_unit = solution.get_config_to_unit(self.cluster_ids)

        # Generate current flow matrix
        flow_matrix = self.calculate_material_flow_matrix(config_to_unit)

        # Create comprehensive visualization
        fig = plt.figure(figsize=(24, 16))

        # Main layout plot
        ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
        self._plot_main_layout_fixed_grid(ax_main, solution, config_to_unit, flow_matrix)

        # Material flow intensity heatmap
        ax_flow = plt.subplot2grid((3, 4), (0, 2))
        self._plot_material_flow_intensity(ax_flow, solution, flow_matrix)

        # Logistics cost breakdown
        ax_logistics = plt.subplot2grid((3, 4), (0, 3))
        self._plot_logistics_cost_breakdown(ax_logistics, solution)

        # Fitness evolution
        ax_fitness = plt.subplot2grid((3, 4), (1, 2))
        self._plot_fitness_evolution(ax_fitness)

        # Unit utilization
        ax_utilization = plt.subplot2grid((3, 4), (1, 3))
        self._plot_unit_utilization(ax_utilization, config_to_unit)

        # Objective space
        ax_objectives = plt.subplot2grid((3, 4), (2, 0))
        self._plot_objective_space(ax_objectives)

        # Pareto front
        ax_pareto = plt.subplot2grid((3, 4), (2, 1))
        self._plot_pareto_front(ax_pareto)

        # Performance metrics
        ax_metrics = plt.subplot2grid((3, 4), (2, 2))
        self._plot_performance_metrics(ax_metrics, solution)

        # Layout statistics
        ax_stats = plt.subplot2grid((3, 4), (2, 3))
        self._plot_layout_statistics(ax_stats, solution, flow_matrix)

        plt.tight_layout()

        if save_plots:
            layout_path = os.path.join(self.output_dir, "images", "fixed_9x4_grid_layout_solution.jpg")
            plt.savefig(layout_path, dpi=1200, bbox_inches='tight', format='jpg')
            print(f"🎨 Fixed 9×4 grid layout visualization saved to: {layout_path}")

        # plt.show()

        # Generate additional detailed plots
        self._generate_detailed_flow_visualization_fixed_grid(solution, flow_matrix, save_plots)

        return {
            "layout_bounds": self.layout_bounds,
            "fixed_unit_centers": self.fixed_unit_centers,
            "config_to_unit": config_to_unit,
            "fitness": solution.fitness,
            "flow_matrix": flow_matrix,
            "logistics_list": self.logistics_list
        }

    def _plot_main_layout_fixed_grid(self, ax, solution: Individual,
                                     config_to_unit: Dict[str, int], flow_matrix: np.ndarray):
        """Plot main fixed 9x4 grid layout"""

        # Draw grid lines
        grid_x = self.layout_config["grid_x"]
        grid_y = self.layout_config["grid_y"]
        unit_size = self.layout_config["unit_size"]
        unit_spacing = self.layout_config["unit_spacing"]

        # Draw grid
        for i in range(grid_x + 1):
            x = i * (unit_size + unit_spacing) - unit_spacing / 2
            if i < grid_x:
                ax.axvline(x + unit_spacing / 2, color='lightgray', linestyle='--', alpha=0.5)

        for j in range(grid_y + 1):
            y = j * (unit_size + unit_spacing) - unit_spacing / 2
            if j < grid_y:
                ax.axhline(y + unit_spacing / 2, color='lightgray', linestyle='--', alpha=0.5)

        # Color mapping for different clusters
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.cluster_ids)))
        color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(self.cluster_ids)}

        # Plot all fixed units
        for unit_id, center in enumerate(self.fixed_unit_centers):
            # Check if any cluster is assigned to this unit
            assigned_cluster = None
            for cluster_id, assigned_unit in config_to_unit.items():
                if assigned_unit == unit_id:
                    assigned_cluster = cluster_id
                    break

            if assigned_cluster:
                color = color_map.get(assigned_cluster, 'gray')
                alpha = 0.8

                # Get process count for label
                process_count = len(self.clusters_data.get(assigned_cluster, {}).get('process_ids', []))
                label = f'U{unit_id}\n{assigned_cluster}\n({process_count}p)'
            else:
                color = 'lightgray'
                alpha = 0.3
                label = f'U{unit_id}\n(Empty)'

            # Draw unit as square
            unit_square = Rectangle(
                (center[0] - unit_size / 2, center[1] - unit_size / 2),
                unit_size, unit_size,
                facecolor=color, alpha=alpha,
                edgecolor='black', linewidth=2
            )
            ax.add_patch(unit_square)

            # Add unit label
            ax.text(center[0], center[1], label,
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))

        # Plot material flows
        self._plot_flow_arrows_fixed_grid(ax, solution)

        ax.set_xlim(-5, self.layout_bounds[1] + 5)
        ax.set_ylim(-5, self.layout_bounds[3] + 5)
        ax.set_xlabel('X Coordinate (units)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Coordinate (units)', fontsize=12, fontweight='bold')
        ax.set_title('Optimized Fixed 9×4 Grid Production Layout with Material Flow',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    def _plot_flow_arrows_fixed_grid_with_pathfinding_safe(self, ax, solution: Individual):
        """Plot material flow arrows with safe pathfinding and progress bar"""
        if not self.logistics_list:
            print("⚠️ No logistics data available for visualization")
            return

        print("📊 Preparing flow visualization with pathfinding...")

        # 准备流量数据
        flow_data = []
        for logistics in self.logistics_list:
            if (logistics.from_unit >= 0 and logistics.to_unit >= 0 and
                    logistics.from_unit < len(self.fixed_unit_centers) and
                    logistics.to_unit < len(self.fixed_unit_centers)):
                flow_data.append(logistics)

        if not flow_data:
            print("⚠️ No valid flow data found")
            return

        print(f"📈 Processing {len(flow_data)} material flows...")

        # 获取流量范围用于归一化
        flow_values = [log.frequency for log in flow_data]
        min_flow = min(flow_values)
        max_flow = max(flow_values)

        if max_flow == min_flow:
            max_flow = min_flow + 1

        # 使用学术论文推荐的颜色映射
        from matplotlib.colors import LinearSegmentedColormap
        colors_academic = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1976D2', '#0D47A1']
        cmap_academic = LinearSegmentedColormap.from_list('academic_flow', colors_academic, N=256)

        # 创建路径规划网格（优化版本）
        print("🗺️ Creating pathfinding grid...")
        path_grid = self._create_pathfinding_grid_optimized()

        # 使用进度条绘制每条物流路径
        successful_paths = 0
        failed_paths = 0

        print("🎨 Drawing material flow paths...")
        with tqdm(total=len(flow_data), desc="Drawing flow paths",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

            for i, logistics in enumerate(flow_data):
                pbar.set_description(f"Drawing flow {i + 1}/{len(flow_data)}")

                try:
                    flow_intensity = logistics.frequency
                    start_unit = logistics.from_unit
                    end_unit = logistics.to_unit

                    if start_unit == end_unit:
                        pbar.update(1)
                        continue

                    # 计算路径（带超时保护）
                    start_time = time.time()
                    path_points = self._find_optimal_path_safe(start_unit, end_unit, path_grid, timeout=2.0)
                    path_time = time.time() - start_time

                    if len(path_points) < 2:
                        failed_paths += 1
                        pbar.update(1)
                        continue

                    # 标准化流量强度
                    normalized_flow = (flow_intensity - min_flow) / (max_flow - min_flow)

                    # 根据流量强度调整线条属性
                    line_width = 1.0 + normalized_flow * 3.0
                    alpha = 0.6 + normalized_flow * 0.4
                    arrow_color = cmap_academic(normalized_flow)

                    # 绘制路径
                    self._draw_path_with_arrows_safe(ax, path_points, arrow_color,
                                                     line_width, alpha, flow_intensity, normalized_flow)

                    successful_paths += 1

                except Exception as e:
                    print(f"\n⚠️ Error drawing flow {i + 1}: {e}")
                    failed_paths += 1

                pbar.update(1)

        print(f"✅ Successfully drew {successful_paths} paths, {failed_paths} failed")

        # 添加颜色条
        if successful_paths > 0:
            print("🎨 Adding colorbar...")
            self._add_colorbar_to_axis_safe(ax, cmap_academic, min_flow, max_flow)

        print(f"📊 Flow visualization completed: {successful_paths}/{len(flow_data)} paths drawn")

    def _create_pathfinding_grid_optimized(self):
        """创建优化的路径规划网格"""
        print("⚙️ Creating optimized pathfinding grid...")

        # 使用较低分辨率以提高性能
        grid_x = self.layout_config["grid_x"]
        grid_y = self.layout_config["grid_y"]
        unit_size = self.layout_config["unit_size"]
        unit_spacing = self.layout_config["unit_spacing"]

        # 降低分辨率：每2个单位1个网格点
        resolution = 2
        total_width = int((grid_x * (unit_size + unit_spacing)) // resolution)
        total_height = int((grid_y * (unit_size + unit_spacing)) // resolution)

        # 创建路径规划网格
        path_grid = np.zeros((total_height, total_width), dtype=int)

        # 将单元区域标记为障碍物
        for unit_id, center in enumerate(self.fixed_unit_centers):
            row = unit_id // grid_x
            col = unit_id % grid_x

            # 计算单元在网格中的位置（考虑分辨率）
            start_x = int((col * (unit_size + unit_spacing)) // resolution)
            end_x = int((start_x * resolution + unit_size) // resolution)
            start_y = int((row * (unit_size + unit_spacing)) // resolution)
            end_y = int((start_y * resolution + unit_size) // resolution)

            # 确保边界有效
            start_x = max(0, start_x)
            end_x = min(path_grid.shape[1], end_x)
            start_y = max(0, start_y)
            end_y = min(path_grid.shape[0], end_y)

            # 标记单元区域为障碍物
            if end_y > start_y and end_x > start_x:
                path_grid[start_y:end_y, start_x:end_x] = 1

        print(f"✅ Grid created: {total_width}x{total_height} (resolution: {resolution})")
        return path_grid

    def _find_optimal_path_safe(self, start_unit, end_unit, path_grid, timeout=5.0):
        """安全的路径规划，带超时保护"""
        start_time = time.time()

        try:
            # 获取起始和目标单元的网格坐标
            start_center = self.fixed_unit_centers[start_unit]
            end_center = self.fixed_unit_centers[end_unit]

            # 使用简化的直线路径策略
            path_points = self._simple_path_strategy(start_center, end_center, path_grid)

            # 检查超时
            if time.time() - start_time > timeout:
                print(f"⚠️ Path finding timeout for units {start_unit}->{end_unit}")
                return self._fallback_direct_path(start_center, end_center)

            return path_points

        except Exception as e:
            print(f"⚠️ Error in pathfinding for units {start_unit}->{end_unit}: {e}")
            return self._fallback_direct_path(start_center, end_center)

    def _simple_path_strategy(self, start_center, end_center, path_grid):
        """简化的路径策略，避免复杂计算"""
        # 使用简单的L形路径
        path_points = []

        # 起点
        path_points.append(start_center)

        # 计算方向
        dx = end_center[0] - start_center[0]
        dy = end_center[1] - start_center[1]

        # 添加中间点（L形路径）
        if abs(dx) > abs(dy):
            # 先水平后垂直
            intermediate = (end_center[0], start_center[1])
        else:
            # 先垂直后水平
            intermediate = (start_center[0], end_center[1])

        # 只在中间点与起点和终点不同时添加
        if (intermediate != start_center and intermediate != end_center):
            path_points.append(intermediate)

        # 终点
        path_points.append(end_center)

        return path_points

    def _fallback_direct_path(self, start_center, end_center):
        """备用的直线路径"""
        return [start_center, end_center]

    def _draw_path_with_arrows_safe(self, ax, path_points, color, line_width, alpha, flow_intensity, normalized_flow):
        """安全的路径绘制函数"""
        if len(path_points) < 2:
            return

        try:
            # 将路径点转换为数组
            path_array = np.array(path_points)
            x_coords = path_array[:, 0]
            y_coords = path_array[:, 1]

            # 绘制路径线条
            ax.plot(x_coords, y_coords,
                    color=color,
                    linewidth=line_width,
                    alpha=alpha,
                    solid_capstyle='round',
                    solid_joinstyle='round')

            # 添加简化的箭头
            if len(path_points) >= 2:
                # 在路径中点添加箭头
                mid_idx = len(path_points) // 2
                if mid_idx > 0:
                    start_pos = path_points[mid_idx - 1]
                    end_pos = path_points[mid_idx]

                    # 计算箭头方向
                    direction = np.array(end_pos) - np.array(start_pos)
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        arrow_start = np.array(start_pos) + direction * 1
                        arrow_end = np.array(start_pos) + direction * 3

                        ax.annotate('',
                                    xy=arrow_end,
                                    xytext=arrow_start,
                                    arrowprops=dict(
                                        arrowstyle='-|>',
                                        color=color,
                                        alpha=alpha,
                                        lw=line_width * 0.8,
                                        mutation_scale=8 + normalized_flow * 5
                                    ))

            # 为高强度流量添加标签
            if normalized_flow > 0.7 and len(path_points) > 1:
                mid_point = path_points[len(path_points) // 2]
                ax.text(mid_point[0], mid_point[1] + 1,
                        f'{flow_intensity:.0f}',
                        ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='#2C3E50',
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor='white',
                                  alpha=0.9,
                                  edgecolor='none'))

        except Exception as e:
            print(f"⚠️ Error drawing path: {e}")

    def _add_colorbar_to_axis_safe(self, ax, cmap, min_flow, max_flow):
        """安全地添加颜色条"""
        try:
            # 创建颜色条
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                       norm=plt.Normalize(vmin=min_flow, vmax=max_flow))
            sm.set_array([])

            # 在主图旁边添加颜色条
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)

            cbar = plt.colorbar(sm, cax=cax)
            cbar.set_label('Material Flow Frequency',
                           rotation=270,
                           labelpad=20,
                           fontsize=11,
                           fontweight='bold',
                           color='#2C3E50')
            cbar.ax.tick_params(labelsize=9, colors='#2C3E50')

            # 美化颜色条
            cbar.outline.set_edgecolor('#B0B0B0')
            cbar.outline.set_linewidth(1)

            print("✅ Colorbar added successfully")

        except Exception as e:
            print(f"⚠️ Error adding colorbar: {e}")

    def visualize_layout_solution_with_progress(self, solution: Individual = None,
                                                save_plots: bool = True) -> Dict[str, Any]:
        """Enhanced visualization with progress tracking"""
        if solution is None:
            solution = self.best_solution

        if solution is None:
            print("⚠️ No solution to visualize")
            return {}

        print("🎨 Generating fixed 9×4 grid layout visualizations with progress tracking...")

        # Get config mapping
        config_to_unit = solution.get_config_to_unit(self.cluster_ids)

        # Generate current flow matrix
        print("📊 Calculating material flow matrix...")
        flow_matrix = self.calculate_material_flow_matrix(config_to_unit)

        # Create comprehensive visualization with progress
        print("🖼️ Creating visualization plots...")

        with tqdm(total=10, desc="Creating plots",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

            fig = plt.figure(figsize=(24, 16))
            pbar.set_description("Setting up figure")
            pbar.update(1)

            # Main layout plot
            pbar.set_description("Drawing main layout")
            ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
            self._plot_main_layout_fixed_grid_with_progress(ax_main, solution, config_to_unit, flow_matrix)
            pbar.update(1)

            # Material flow intensity heatmap
            pbar.set_description("Creating flow heatmap")
            ax_flow = plt.subplot2grid((3, 4), (0, 2))
            self._plot_material_flow_intensity(ax_flow, solution, flow_matrix)
            pbar.update(1)

            # Logistics cost breakdown
            pbar.set_description("Drawing cost breakdown")
            ax_logistics = plt.subplot2grid((3, 4), (0, 3))
            self._plot_logistics_cost_breakdown(ax_logistics, solution)
            pbar.update(1)

            # Fitness evolution
            pbar.set_description("Drawing fitness evolution")
            ax_fitness = plt.subplot2grid((3, 4), (1, 2))
            self._plot_fitness_evolution(ax_fitness)
            pbar.update(1)

            # Unit utilization
            pbar.set_description("Drawing unit utilization")
            ax_utilization = plt.subplot2grid((3, 4), (1, 3))
            self._plot_unit_utilization(ax_utilization, config_to_unit)
            pbar.update(1)

            # Objective space
            pbar.set_description("Drawing objective space")
            ax_objectives = plt.subplot2grid((3, 4), (2, 0))
            self._plot_objective_space(ax_objectives)
            pbar.update(1)

            # Pareto front
            pbar.set_description("Drawing Pareto front")
            ax_pareto = plt.subplot2grid((3, 4), (2, 1))
            self._plot_pareto_front(ax_pareto)
            pbar.update(1)

            # Performance metrics
            pbar.set_description("Drawing performance metrics")
            ax_metrics = plt.subplot2grid((3, 4), (2, 2))
            self._plot_performance_metrics(ax_metrics, solution)
            pbar.update(1)

            # Layout statistics
            pbar.set_description("Finalizing layout")
            ax_stats = plt.subplot2grid((3, 4), (2, 3))
            self._plot_layout_statistics(ax_stats, solution, flow_matrix)
            pbar.update(1)

        plt.tight_layout()

        if save_plots:
            layout_path = os.path.join(self.output_dir, "images", "fixed_9x4_grid_layout_solution.jpg")
            print(f"💾 Saving main visualization to: {layout_path}")
            plt.savefig(layout_path, dpi=1200, bbox_inches='tight', format='jpg')
            print(f"✅ Main visualization saved successfully")

        plt.show()

        # Generate additional detailed plots
        print("🔄 Generating additional detailed visualizations...")
        self._generate_detailed_flow_visualization_fixed_grid(solution, flow_matrix, save_plots)

        return {
            "layout_bounds": self.layout_bounds,
            "fixed_unit_centers": self.fixed_unit_centers,
            "config_to_unit": config_to_unit,
            "fitness": solution.fitness,
            "flow_matrix": flow_matrix,
            "logistics_list": self.logistics_list
        }

    def _plot_main_layout_fixed_grid_with_progress(self, ax, solution: Individual,
                                                   config_to_unit: Dict[str, int], flow_matrix: np.ndarray):
        """Plot main fixed 9x4 grid layout with progress tracking"""

        print("🏗️ Drawing grid units...")

        # 设置学术论文风格
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
            'font.size': 10,
            'axes.linewidth': 1.2,
        })

        # Draw grid and units (existing code)
        grid_x = self.layout_config["grid_x"]
        grid_y = self.layout_config["grid_y"]
        unit_size = self.layout_config["unit_size"]
        unit_spacing = self.layout_config["unit_spacing"]

        # 统一的单元颜色
        unit_color = '#E8F4FD'
        unit_edge_color = '#B0C4DE'
        empty_unit_color = '#F8F9FA'
        empty_unit_edge_color = '#E0E0E0'

        # Plot all fixed units
        total_units = len(self.fixed_unit_centers)
        for unit_id, center in enumerate(self.fixed_unit_centers):
            if unit_id % 10 == 0:  # 每10个单元显示一次进度
                print(f"   Drawing unit {unit_id + 1}/{total_units}")

            # 绘制单元（简化版本的原代码）
            assigned_cluster = None
            for cluster_id, assigned_unit in config_to_unit.items():
                if assigned_unit == unit_id:
                    assigned_cluster = cluster_id
                    break

            if assigned_cluster:
                color = unit_color
                edge_color = unit_edge_color
                alpha = 0.9
                edge_width = 1.5
                process_count = len(self.clusters_data.get(assigned_cluster, {}).get('process_ids', []))
                label = f'U{unit_id}\n{assigned_cluster}\n({process_count}p)'
                label_color = '#2C3E50'
            else:
                color = empty_unit_color
                edge_color = empty_unit_edge_color
                alpha = 0.6
                edge_width = 1.0
                label = f'U{unit_id}\n(Empty)'
                label_color = '#7F8C8D'

            # Draw unit
            from matplotlib.patches import FancyBboxPatch
            unit_rect = FancyBboxPatch(
                (center[0] - unit_size / 2, center[1] - unit_size / 2),
                unit_size, unit_size,
                boxstyle="round,pad=0.1",
                facecolor=color, alpha=alpha,
                edgecolor=edge_color, linewidth=edge_width
            )
            ax.add_patch(unit_rect)

            # Add label
            ax.text(center[0], center[1], label,
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color=label_color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='none'))

        print("🌊 Drawing material flows...")
        # Plot material flows with safe pathfinding
        self._plot_flow_arrows_fixed_grid_with_pathfinding_safe(ax, solution)

        # Set axis properties
        ax.set_xlim(-5, self.layout_bounds[1] + 5)
        ax.set_ylim(-5, self.layout_bounds[3] + 5)
        ax.set_xlabel('X Coordinate (units)', fontsize=12, fontweight='bold', color='#2C3E50')
        ax.set_ylabel('Y Coordinate (units)', fontsize=12, fontweight='bold', color='#2C3E50')
        ax.set_title('Optimized Fixed 9×4 Grid Production Layout with Material Flow',
                     fontsize=14, fontweight='bold', color='#2C3E50', pad=20)
        ax.grid(True, alpha=0.2, color='#E5E5E5', linewidth=0.5)
        ax.set_aspect('equal')

        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#B0B0B0')
        ax.spines['bottom'].set_color('#B0B0B0')

        print("✅ Main layout completed")

    def _plot_main_layout_fixed_grid(self, ax, solution: Individual,
                                     config_to_unit: Dict[str, int], flow_matrix: np.ndarray):
        """Plot main fixed 9x4 grid layout with academic paper style"""

        # 设置学术论文风格
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
            'font.size': 10,
            'axes.linewidth': 1.2,
            'axes.edgecolor': '#333333',
            'axes.labelcolor': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'text.color': '#333333'
        })

        # Draw grid lines with subtle style
        grid_x = self.layout_config["grid_x"]
        grid_y = self.layout_config["grid_y"]
        unit_size = self.layout_config["unit_size"]
        unit_spacing = self.layout_config["unit_spacing"]

        # Draw subtle grid
        for i in range(grid_x + 1):
            x = i * (unit_size + unit_spacing) - unit_spacing / 2
            if i < grid_x:
                ax.axvline(x + unit_spacing / 2, color='#E5E5E5', linestyle='-', alpha=0.3, linewidth=0.5)

        for j in range(grid_y + 1):
            y = j * (unit_size + unit_spacing) - unit_spacing / 2
            if j < grid_y:
                ax.axhline(y + unit_spacing / 2, color='#E5E5E5', linestyle='-', alpha=0.3, linewidth=0.5)

        # 统一的单元颜色 - 使用优雅的浅蓝灰色
        unit_color = '#E8F4FD'  # 浅蓝灰色，专业且柔和
        unit_edge_color = '#B0C4DE'  # 稍深的边框色
        empty_unit_color = '#F8F9FA'  # 空单元的颜色
        empty_unit_edge_color = '#E0E0E0'  # 空单元边框色

        # Plot all fixed units with uniform color
        for unit_id, center in enumerate(self.fixed_unit_centers):
            # Check if any cluster is assigned to this unit
            assigned_cluster = None
            for cluster_id, assigned_unit in config_to_unit.items():
                if assigned_unit == unit_id:
                    assigned_cluster = cluster_id
                    break

            if assigned_cluster:
                color = unit_color
                edge_color = unit_edge_color
                alpha = 0.9
                edge_width = 1.5

                # Get process count for label
                process_count = len(self.clusters_data.get(assigned_cluster, {}).get('process_ids', []))
                label = f'U{unit_id}\n{assigned_cluster}\n({process_count}p)'
                label_color = '#2C3E50'  # 深色文字
            else:
                color = empty_unit_color
                edge_color = empty_unit_edge_color
                alpha = 0.6
                edge_width = 1.0
                label = f'U{unit_id}\n(Empty)'
                label_color = '#7F8C8D'  # 灰色文字

            # Draw unit as rounded rectangle for modern look
            from matplotlib.patches import FancyBboxPatch
            unit_rect = FancyBboxPatch(
                (center[0] - unit_size / 2, center[1] - unit_size / 2),
                unit_size, unit_size,
                boxstyle="round,pad=0.1",
                facecolor=color,
                alpha=alpha,
                edgecolor=edge_color,
                linewidth=edge_width
            )
            ax.add_patch(unit_rect)

            # Add unit label with improved styling
            ax.text(center[0], center[1], label,
                    ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color=label_color,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor='white',
                              alpha=0.9,
                              edgecolor='none'))

        # Plot material flows with enhanced styling
        self._plot_flow_arrows_fixed_grid_with_pathfinding_safe(ax, solution)

        # Set axis properties
        ax.set_xlim(-5, self.layout_bounds[1] + 5)
        ax.set_ylim(-5, self.layout_bounds[3] + 5)
        ax.set_xlabel('X Coordinate (units)', fontsize=12, fontweight='bold', color='#2C3E50')
        ax.set_ylabel('Y Coordinate (units)', fontsize=12, fontweight='bold', color='#2C3E50')
        ax.set_title('Optimized Fixed 9×4 Grid Production Layout with Material Flow',
                     fontsize=14, fontweight='bold', color='#2C3E50', pad=20)
        ax.grid(True, alpha=0.2, color='#E5E5E5', linewidth=0.5)
        ax.set_aspect('equal')

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#B0B0B0')
        ax.spines['bottom'].set_color('#B0B0B0')

    def _plot_material_flow_intensity(self, ax, solution: Individual, flow_matrix: np.ndarray):
        """Plot material flow intensity heatmap with grid"""
        if flow_matrix is None or flow_matrix.size == 0:
            ax.text(0.5, 0.5, 'No Flow Data\nAvailable',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
            ax.set_title('Material Flow\nIntensity Matrix', fontweight='bold')
            return

        # Plot heatmap
        im = ax.imshow(flow_matrix, cmap='Reds', alpha=0.8)

        # Customize ticks
        config_to_unit = solution.get_config_to_unit(self.cluster_ids)
        unit_labels = []

        # Create labels for units with assigned clusters
        for i in range(min(len(flow_matrix), self.n_units)):
            label = f'U{i}'
            unit_labels.append(label)

        # 设置主要的刻度（单元格中心）
        ax.set_xticks(range(len(unit_labels)))
        ax.set_yticks(range(len(unit_labels)))
        ax.set_xticklabels(unit_labels, rotation=45)
        ax.set_yticklabels(unit_labels)

        # 添加网格线（在单元格边界）
        # 设置次要刻度用于网格线
        ax.set_xticks([i - 0.5 for i in range(len(unit_labels) + 1)], minor=True)
        ax.set_yticks([i - 0.5 for i in range(len(unit_labels) + 1)], minor=True)

        # 启用网格并设置样式
        ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5, alpha=0.8)
        ax.grid(which='major', color='none')  # 关闭主要网格

        # 设置网格在图像上层
        ax.set_axisbelow(False)

        # Add text annotations for significant flows
        for i in range(len(unit_labels)):
            for j in range(len(unit_labels)):
                if flow_matrix[i, j] > 0:
                    text = ax.text(j, i, f'{flow_matrix[i, j]:.0f}',
                                   ha="center", va="center", color="black", fontsize=8,
                                   fontweight='bold')

        ax.set_title('Material Flow\nIntensity Matrix', fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, label='Flow Quantity')
        cbar.ax.tick_params(labelsize=8)

    def _plot_logistics_cost_breakdown(self, ax, solution: Individual):
        """Plot logistics cost breakdown"""
        if not self.logistics_list:
            ax.text(0.5, 0.5, 'No Logistics\nData Available',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12)
            return

        # Calculate costs by distance range
        distance_ranges = ['0-10', '10-20', '20-30', '30+']
        cost_by_range = [0, 0, 0, 0]

        for logistics in self.logistics_list:
            if logistics.distance > 0 and logistics.cost > 0:
                if logistics.distance < 10:
                    cost_by_range[0] += logistics.cost
                elif logistics.distance < 20:
                    cost_by_range[1] += logistics.cost
                elif logistics.distance < 30:
                    cost_by_range[2] += logistics.cost
                else:
                    cost_by_range[3] += logistics.cost

        # Plot pie chart
        if sum(cost_by_range) > 0:
            colors = ['green', 'yellow', 'orange', 'red']
            ax.pie(cost_by_range, labels=distance_ranges, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax.set_title('Logistics Cost by\nDistance Range', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Logistics\nCosts', ha='center', va='center', transform=ax.transAxes)

    def _plot_fitness_evolution(self, ax):
        """Plot fitness evolution over generations"""
        if not self.evolution_history["best_fitness"]:
            ax.text(0.5, 0.5, 'No Evolution\nData Available',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12)
            return

        generations = range(len(self.evolution_history["best_fitness"]))

        ax.plot(generations, self.evolution_history["best_fitness"],
                'b-', linewidth=2, label='Best Logistics Cost', alpha=0.8)
        ax.plot(generations, self.evolution_history["average_fitness"],
                'r--', linewidth=1.5, label='Average Logistics Cost', alpha=0.7)

        ax.set_xlabel('Generation')
        ax.set_ylabel('Logistics Cost')
        ax.set_title('Logistics Cost Evolution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_unit_utilization(self, ax, config_to_unit: Dict[str, int]):
        """Plot unit utilization based on assigned processes"""
        unit_loads = {}

        # Calculate load for each unit
        for i in range(min(12, self.n_units)):  # Show first 12 units
            unit_loads[f'U{i}'] = 0

        for cluster_id, unit_id in config_to_unit.items():
            if cluster_id in self.clusters_data and unit_id < 12:
                cluster_info = self.clusters_data[cluster_id]

                # Calculate load based on processes and orders
                total_load = 0
                for process_info in cluster_info['process_ids']:
                    product_id = process_info.get('bm', 'Unknown')
                    order_qty = self.order_quantities.get(product_id, 0)
                    total_load += order_qty

                unit_loads[f'U{unit_id}'] = total_load

        if unit_loads:
            units = list(unit_loads.keys())
            loads = list(unit_loads.values())

            bars = ax.bar(range(len(units)), loads, color='steelblue', alpha=0.7)
            ax.set_xticks(range(len(units)))
            ax.set_xticklabels(units, rotation=45)
            ax.set_ylabel('Workload (orders)')
            ax.set_title('Unit Workload\nDistribution', fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, load in zip(bars, loads):
                if load > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + max(loads) * 0.01,
                            f'{load}', ha='center', va='bottom', fontsize=8)

    def _plot_objective_space(self, ax):
        """Plot objective space distribution"""
        if not self.pareto_front:
            ax.text(0.5, 0.5, 'No Pareto Front\nData Available',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12)
            return

        # Extract first two objectives
        obj1 = [ind.fitness[0] for ind in self.pareto_front]
        obj2 = [ind.fitness[1] for ind in self.pareto_front]

        ax.scatter(obj1, obj2, c='red', s=50, alpha=0.7, edgecolors='black')
        ax.set_xlabel('Logistics Cost')
        ax.set_ylabel('Space Balance')
        ax.set_title('Objective Space', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_pareto_front(self, ax):
        """Plot Pareto front"""
        if len(self.pareto_front) < 2:
            ax.text(0.5, 0.5, 'Insufficient Pareto\nFront Data',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12)
            return

        # Sort Pareto front by first objective
        sorted_front = sorted(self.pareto_front, key=lambda x: x.fitness[0])

        obj1 = [ind.fitness[0] for ind in sorted_front]
        obj2 = [ind.fitness[1] for ind in sorted_front]

        ax.plot(obj1, obj2, 'ro-', linewidth=2, markersize=6, alpha=0.8)
        ax.set_xlabel('Logistics Cost')
        ax.set_ylabel('Space Balance')
        ax.set_title('Pareto Front', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_performance_metrics(self, ax, solution: Individual):
        """Plot performance metrics"""
        metrics = {
            'Logistics Cost': solution.fitness[0],
            'Space Balance': solution.fitness[1],
            'Energy Cost': solution.fitness[2]
        }

        # Create bar plot
        names = list(metrics.keys())
        values = list(metrics.values())

        # Normalize values for better visualization
        max_val = max(values) if values else 1
        normalized_values = [v / max_val for v in values]

        bars = ax.bar(range(len(names)), normalized_values,
                      color=['red', 'green', 'orange'], alpha=0.7)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Performance Metrics', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)

    def _plot_layout_statistics(self, ax, solution: Individual, flow_matrix: np.ndarray):
        """Plot layout statistics"""
        config_to_unit = solution.get_config_to_unit(self.cluster_ids)
        n_assigned = len(config_to_unit)

        # Total material flow
        total_flow = sum(log.frequency for log in self.logistics_list)

        # Active logistics connections
        active_connections = len([log for log in self.logistics_list if log.frequency > 0])

        # Average distance
        if self.logistics_list:
            avg_distance = np.mean([log.distance for log in self.logistics_list if log.distance > 0])
        else:
            avg_distance = 0

        # Space utilization
        space_utilization = n_assigned / self.n_units if self.n_units > 0 else 0

        stats_text = f"""Layout Statistics:

Grid: {self.layout_config['grid_x']}×{self.layout_config['grid_y']}
Total Units: {self.n_units}
Assigned Units: {n_assigned}
Empty Units: {self.n_units - n_assigned}

Active Logistics: {active_connections}
Total Flow: {total_flow:.0f}
Avg Distance: {avg_distance:.1f}

Space Util.: {space_utilization:.1%}
AGV Speed: {self.layout_config['agv_speed']} u/ts"""

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Layout Statistics', fontweight='bold')

    def _generate_detailed_flow_visualization_fixed_grid(self, solution: Individual,
                                                         flow_matrix: np.ndarray, save_plots: bool):
        """Generate detailed flow visualization for fixed grid layout"""
        if not self.logistics_list:
            print("⚠️ No logistics data for detailed visualization")
            return

        print("🌊 Generating detailed fixed grid flow visualization...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # 1. Network graph representation
        G = nx.DiGraph()
        config_to_unit = solution.get_config_to_unit(self.cluster_ids)

        # Add nodes
        for cluster_id, unit_id in config_to_unit.items():
            G.add_node(f'U{unit_id}')

        # Add edges based on logistics list
        for logistics in self.logistics_list:
            if (logistics.from_unit >= 0 and logistics.to_unit >= 0 and
                    logistics.frequency > 0):
                G.add_edge(f'U{logistics.from_unit}', f'U{logistics.to_unit}',
                           weight=logistics.frequency)

        # Position nodes based on fixed unit centers
        pos = {}
        for cluster_id, unit_id in config_to_unit.items():
            if unit_id < len(self.fixed_unit_centers):
                pos[f'U{unit_id}'] = self.fixed_unit_centers[unit_id]

        # Draw network
        if G.edges():
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightblue',
                                   node_size=1000, alpha=0.8)

            # Draw edges with varying thickness
            for (u, v, d) in G.edges(data=True):
                weight = d['weight']
                width = (weight / max_weight) * 15
                alpha = min(1.0, weight / max_weight + 0.3)

                nx.draw_networkx_edges(G, pos, [(u, v)], width=width,
                                       alpha=alpha, edge_color='red', ax=ax1,
                                       connectionstyle="arc3,rad=0.1")

            # Draw labels
            nx.draw_networkx_labels(G, pos, ax=ax1, font_size=10, font_weight='bold')

        ax1.set_title('Material Flow Network (Fixed Grid)', fontsize=14, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # 2. Flow intensity distribution
        flow_values = [log.frequency for log in self.logistics_list if log.frequency > 0]

        if flow_values:
            ax2.hist(flow_values, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Flow Intensity')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Flow Intensity Distribution', fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Add statistics
            ax2.axvline(np.mean(flow_values), color='red', linestyle='--',
                        label=f'Mean: {np.mean(flow_values):.1f}')
            ax2.legend()

        # 3. Distance vs Flow scatter plot
        distances = []
        flows = []

        for logistics in self.logistics_list:
            if logistics.distance > 0 and logistics.frequency > 0:
                distances.append(logistics.distance)
                flows.append(logistics.frequency)

        if distances and flows:
            ax3.scatter(distances, flows, alpha=0.7, s=50, edgecolors='black')
            ax3.set_xlabel('Distance between Units')
            ax3.set_ylabel('Flow Intensity')
            ax3.set_title('Distance vs Flow Intensity', fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # Add trend line
            if len(distances) > 1:
                z = np.polyfit(distances, flows, 1)
                p = np.poly1d(z)
                ax3.plot(sorted(distances), p(sorted(distances)), "r--", alpha=0.8, label='Trend')
                ax3.legend()

        # 4. AGV transport time analysis
        transport_times = []
        flow_weights = []

        for logistics in self.logistics_list:
            if logistics.transport_time > 0 and logistics.frequency > 0:
                transport_times.append(logistics.transport_time)
                flow_weights.append(logistics.frequency)

        if transport_times:
            # Weighted histogram
            ax4.hist(transport_times, bins=15, weights=flow_weights,
                     alpha=0.7, color='orange', edgecolor='black')
            ax4.set_xlabel('AGV Transport Time (time steps)')
            ax4.set_ylabel('Weighted Frequency')
            ax4.set_title(f'AGV Transport Time Distribution\n(Speed: {self.layout_config["agv_speed"]} units/step)',
                          fontweight='bold')
            ax4.grid(True, alpha=0.3)

            # Add average transport time
            avg_time = np.average(transport_times, weights=flow_weights)
            ax4.axvline(avg_time, color='red', linestyle='--',
                        label=f'Weighted Avg: {avg_time:.1f} steps')
            ax4.legend()

        plt.tight_layout()

        if save_plots:
            flow_path = os.path.join(self.output_dir, "images", "detailed_fixed_grid_flow_analysis.jpg")
            plt.savefig(flow_path, dpi=1200, bbox_inches='tight', format='jpg')
            print(f"🌊 Detailed fixed grid flow visualization saved to: {flow_path}")

        # plt.show()

    def save_optimization_results(self, results: Dict[str, Any]):
        """Save comprehensive optimization results"""
        print("💾 Saving optimization results...")

        # Get config mapping
        config_to_unit = results["best_solution"].get_config_to_unit(self.cluster_ids)

        # Calculate flow matrix for best solution
        flow_matrix = self.calculate_material_flow_matrix(config_to_unit)

        # Convert logistics list to serializable format
        logistics_data = []
        for log in self.logistics_list:
            logistics_data.append({
                "from_unit": log.from_unit,
                "to_unit": log.to_unit,
                "from_process": log.from_process,
                "to_process": log.to_process,
                "product_type": log.product_type,
                "frequency": log.frequency,
                "distance": log.distance,
                "transport_time": log.transport_time,
                "cost": log.cost
            })

        # Save layout optimization data
        layout_data = {
            "cluster_assignment": results["best_solution"].cluster_assignment,
            "config_to_unit": config_to_unit,
            "fixed_unit_centers": [list(c) for c in self.fixed_unit_centers],
            "fitness_values": results["best_solution"].fitness,
            "optimization_time": results["optimization_time"],
            "layout_bounds": self.layout_bounds,
            "layout_config": self.layout_config,
            "flow_matrix": flow_matrix.tolist() if flow_matrix is not None else None,
            "logistics_list": logistics_data,
            "order_quantities": self.order_quantities,
            "process_precedence_count": len(self.process_precedence),
            "total_material_flow": sum(log.frequency for log in self.logistics_list)
        }

        layout_file = os.path.join(self.output_dir, "data", "fixed_grid_layout_optimization_data.json")
        with open(layout_file, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, ensure_ascii=False, indent=2)

        # Save Pareto front
        pareto_data = {
            "pareto_front": [
                {
                    "cluster_assignment": ind.cluster_assignment,
                    "config_to_unit": ind.get_config_to_unit(self.cluster_ids),
                    "fitness": ind.fitness,
                    "rank": ind.rank,
                    "crowding_distance": ind.crowding_distance
                }
                for ind in results["pareto_front"]
            ]
        }

        pareto_file = os.path.join(self.output_dir, "data", "pareto_front_data.json")
        with open(pareto_file, 'w', encoding='utf-8') as f:
            json.dump(pareto_data, f, ensure_ascii=False, indent=2)

        # Save evolution history
        evolution_file = os.path.join(self.output_dir, "data", "evolution_history.json")
        with open(evolution_file, 'w', encoding='utf-8') as f:
            json.dump(results["evolution_history"], f, ensure_ascii=False, indent=2)

        # Save configuration
        config_file = os.path.join(self.output_dir, "data", "optimization_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(results["config"], f, ensure_ascii=False, indent=2)

        print(f"💾 Results saved to: {self.output_dir}/data/")
        print(f"   • Layout data: fixed_grid_layout_optimization_data.json")
        print(f"   • Pareto front: pareto_front_data.json")
        print(f"   • Evolution history: evolution_history.json")
        print(f"   • Configuration: optimization_config.json")

    def generate_optimization_report(self, results: Dict[str, Any]):
        """Generate comprehensive optimization report"""
        print("📋 Generating optimization report...")

        best_solution = results["best_solution"]
        pareto_front = results["pareto_front"]
        config_to_unit = best_solution.get_config_to_unit(self.cluster_ids)
        flow_matrix = self.calculate_material_flow_matrix(config_to_unit)

        # Calculate logistics metrics
        total_material_flow = sum(log.frequency for log in self.logistics_list)
        avg_transport_distance = 0
        total_transport_time = 0

        if self.logistics_list:
            distances = [log.distance for log in self.logistics_list if log.frequency > 0]
            transport_times = [log.transport_time * log.frequency for log in self.logistics_list]

            if distances:
                weights = [log.frequency for log in self.logistics_list if log.frequency > 0]
                avg_transport_distance = np.average(distances, weights=weights)
                total_transport_time = sum(transport_times)

        report_content = f"""
==========================================
FIXED 9×4 GRID LAYOUT OPTIMIZATION REPORT
==========================================

📅 Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
⏱️ Optimization Time: {results["optimization_time"]:.2f} seconds

═══════════════════════════════════════════════════════════════════════════════
🏭 LAYOUT CONFIGURATION
═══════════════════════════════════════════════════════════════════════════════

📐 Fixed Grid Layout:
• Grid Size: {self.layout_config['grid_x']}×{self.layout_config['grid_y']} (X×Y)
• Total Positions: {self.n_units}
• Unit Size: {self.layout_config['unit_size']}×{self.layout_config['unit_size']} units
• Unit Spacing: {self.layout_config['unit_spacing']} units
• Total Layout Area: {self.layout_bounds[1]:.0f}×{self.layout_bounds[3]:.0f} units

🚚 Logistics Configuration:
• AGV Speed: {self.layout_config['agv_speed']} units/time_step
• Total Material Flow: {total_material_flow:.0f} units
• Average Transport Distance: {avg_transport_distance:.2f} units
• Total Transport Time: {total_transport_time:.1f} time_steps
• Active Logistics Connections: {len([log for log in self.logistics_list if log.frequency > 0])}

═══════════════════════════════════════════════════════════════════════════════
🏆 OPTIMIZATION RESULTS
═══════════════════════════════════════════════════════════════════════════════

🎯 Best Solution Performance:
• Logistics Cost: {best_solution.fitness[0]:.3f}
• Space Balance Score: {best_solution.fitness[1]:.3f}
• Energy Cost: {best_solution.fitness[2]:.3f}

📊 Material Flow Analysis:
• Process Precedence Relationships: {len(self.process_precedence)}
• Logistics Relationships Identified: {len(self.logistics_list)}
• Active Flow Connections: {np.sum(flow_matrix > 0) if flow_matrix is not None else 0}
• Max Single Flow: {max((log.frequency for log in self.logistics_list), default=0):.0f}
• Average Flow Intensity: {np.mean([log.frequency for log in self.logistics_list if log.frequency > 0]) if self.logistics_list else 0:.1f}

📈 Production Planning:
• Total Orders: {sum(self.order_quantities.values())}
• Product Types: {len(self.order_quantities)}
• Process Clusters: {len(self.clusters_data)}
• Assigned Units: {len(config_to_unit)}
• Empty Units: {self.n_units - len(config_to_unit)}

🏗️ Unit Assignments:"""

        for cluster_id, unit_id in sorted(config_to_unit.items(), key=lambda x: x[1]):
            if cluster_id in self.clusters_data:
                cluster_info = self.clusters_data[cluster_id]
                n_processes = len(cluster_info['process_ids'])

                # Calculate workload
                total_workload = 0
                products = set()
                for process_info in cluster_info['process_ids']:
                    product_id = process_info.get('bm', 'Unknown')
                    products.add(product_id)
                    order_qty = self.order_quantities.get(product_id, 0)
                    total_workload += order_qty

                # Get unit position
                unit_center = self.fixed_unit_centers[unit_id]
                row = unit_id // self.layout_config["grid_x"]
                col = unit_id % self.layout_config["grid_x"]

                report_content += f"""
• Unit {unit_id} (Row {row + 1}, Col {col + 1}) ← Cluster {cluster_id}: 
  - Position: ({unit_center[0]:.1f}, {unit_center[1]:.1f})
  - Processes: {n_processes}
  - Products: {', '.join(products)}
  - Workload: {total_workload} units"""

        # Add logistics flow details
        report_content += f"""

📦 Top Material Flows:"""

        sorted_logistics = sorted(self.logistics_list, key=lambda x: x.frequency, reverse=True)
        for i, log in enumerate(sorted_logistics[:10]):
            if log.frequency > 0:
                report_content += f"""
• Flow {i + 1}: Unit {log.from_unit} → Unit {log.to_unit}
  - Frequency: {log.frequency:.0f} units
  - Distance: {log.distance:.1f} units
  - Transport Time: {log.transport_time:.1f} time_steps
  - Products: {log.product_type}"""

        # Add Pareto front analysis
        if pareto_front:
            obj1_values = [ind.fitness[0] for ind in pareto_front]
            obj2_values = [ind.fitness[1] for ind in pareto_front]
            obj3_values = [ind.fitness[2] for ind in pareto_front]

            report_content += f"""

📊 Pareto Front Analysis:
• Pareto Front Size: {len(pareto_front)}
• Logistics Cost Range: [{min(obj1_values):.3f}, {max(obj1_values):.3f}]
• Space Balance Range: [{min(obj2_values):.3f}, {max(obj2_values):.3f}]
• Energy Cost Range: [{min(obj3_values):.3f}, {max(obj3_values):.3f}]"""

        # Add evolution analysis
        if results["evolution_history"]["best_fitness"]:
            initial_fitness = results["evolution_history"]["best_fitness"][0]
            final_fitness = results["evolution_history"]["best_fitness"][-1]
            improvement = ((initial_fitness - final_fitness) / initial_fitness) * 100 if initial_fitness > 0 else 0

            report_content += f"""

📈 Evolution Analysis:
• Initial Best Logistics Cost: {initial_fitness:.3f}
• Final Best Logistics Cost: {final_fitness:.3f}
• Improvement: {improvement:.1f}%
• Generations Run: {len(results["evolution_history"]["best_fitness"])}"""

        # Add logistics efficiency analysis
        if self.logistics_list:
            # Calculate efficiency metrics
            n_units = len(config_to_unit)
            max_possible_flows = n_units * (n_units - 1)
            active_flows = len([log for log in self.logistics_list if log.frequency > 0])
            flow_density = active_flows / max_possible_flows if max_possible_flows > 0 else 0

            # Distance efficiency
            if active_flows > 0:
                weighted_avg_distance = avg_transport_distance
                max_distance = max((log.distance for log in self.logistics_list if log.frequency > 0), default=0)
                distance_efficiency = 1 - (weighted_avg_distance / max_distance) if max_distance > 0 else 0
            else:
                distance_efficiency = 0

            report_content += f"""

🚚 Logistics Efficiency Analysis:
• Flow Density: {flow_density:.3f} ({active_flows}/{max_possible_flows} possible connections)
• Average Flow Distance: {avg_transport_distance:.2f} units
• Maximum Flow Distance: {max((log.distance for log in self.logistics_list if log.frequency > 0), default=0):.2f} units
• Distance Efficiency: {distance_efficiency:.3f} (higher = better)
• Total AGV Utilization: {total_transport_time:.1f} time_steps
• Space Utilization: {len(config_to_unit) / self.n_units:.1%} ({len(config_to_unit)}/{self.n_units} units)"""

        report_content += f"""

═══════════════════════════════════════════════════════════════════════════════
💡 RECOMMENDATIONS AND INSIGHTS
═══════════════════════════════════════════════════════════════════════════════

🔧 Layout Optimization:"""

        # Generate specific recommendations
        if len(config_to_unit) < len(self.cluster_ids):
            report_content += f"""
• ⚠️ Not all clusters assigned ({len(self.cluster_ids) - len(config_to_unit)} unassigned) - consider increasing grid size"""

        if flow_density < 0.3:
            report_content += f"""
• 🔄 Low flow density ({flow_density:.1%}) - consider consolidating related processes"""

        if distance_efficiency < 0.5:
            report_content += f"""
• 📏 Low distance efficiency ({distance_efficiency:.1%}) - optimize cluster assignments"""

        if avg_transport_distance > 20:
            report_content += f"""
• 🚚 Long transport distances (avg: {avg_transport_distance:.1f}) - consider rearranging high-flow clusters"""

        # AGV recommendations
        if total_transport_time > 1000:
            optimal_agv_speed = avg_transport_distance / 10  # Heuristic
            report_content += f"""
• ⚡ High transport time load - consider increasing AGV fleet or speed to {optimal_agv_speed:.1f} units/step"""

        # Empty units recommendation
        empty_units = self.n_units - len(config_to_unit)
        if empty_units > self.n_units * 0.3:
            report_content += f"""
• 📦 Many empty units ({empty_units}/{self.n_units}) - consider smaller grid or expanding production"""

        report_content += f"""

📁 Generated Files:
• Fixed grid layout: images/fixed_9x4_grid_layout_solution.jpg
• Detailed flow analysis: images/detailed_fixed_grid_flow_analysis.jpg
• Layout data: data/fixed_grid_layout_optimization_data.json
• Pareto front: data/pareto_front_data.json
• Evolution history: data/evolution_history.json
• Configuration: data/optimization_config.json

==========================================
END OF OPTIMIZATION REPORT
==========================================
"""

        # Save report
        report_path = os.path.join(self.output_dir, "reports", "fixed_9x4_grid_optimization_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"📋 Fixed 9×4 grid optimization report saved to: {report_path}")


def main():
    """Main function for fixed 9x4 grid layout optimization"""

    # Fixed grid configuration
    layout_config = {
        "grid_x": 9,  # X方向9个单元
        "grid_y": 4,  # Y方向4个单元
        "unit_size": 7,  # 单元大小7×7
        "unit_spacing": 5,  # 单元间距5
        "agv_speed": 1.0  # AGV速度
    }

    optimization_config = {
        # Multi-population settings
        "num_populations": 3,
        "total_population_size": 120,
        "elite_size": 15,

        # Adaptive parameters
        "initial_crossover_prob": 0.85,
        "initial_mutation_prob": 0.15,
        "adaptive_factor": 0.7,

        # Convergence settings
        "max_generations": 500,
        "convergence_threshold": 1e-7,
        "diversity_threshold": 0.005,
        "stagnation_limit": 25,

        # Objective weights
        "distance_weight": 1.5,
        "flow_weight": 2.0,  # Higher weight for material flow
        "space_weight": 0.8,

        # Misc
        "random_seed": 42,
        "verbose": True
    }

    # Initialize optimizer with fixed grid configuration
    optimizer = FixedGridNSGA2LayoutOptimizer(
        neo4j_uri="bolt://localhost:7687",
        neo4j_auth=("neo4j", "dididaodao"),
        layout_config=layout_config,
        optimization_config=optimization_config,
        output_dir="Cell layout output"
    )

    # Load clustering results and process precedence from knowledge graph
    optimizer.load_clustering_results("process_config_results/kpc_clusters.json")

    # Run optimization
    results = optimizer.optimize_layout()

    if results:
        # Visualize results
        optimizer.visualize_layout_solution(results["best_solution"])

        # Save results
        optimizer.save_optimization_results(results)

        # Generate report
        optimizer.generate_optimization_report(results)

        print("✅ Optimization completed successfully!")
    else:
        print("❌ Optimization failed!")

    return optimizer, results


if __name__ == "__main__":
    main()