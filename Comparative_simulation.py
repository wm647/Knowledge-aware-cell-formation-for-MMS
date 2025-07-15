#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Algorithm Production Unit Layout Optimization System
使用多种群体智能算法的生产单元布局优化系统
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from py2neo import Graph
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import random
import time
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.family': ['Arial', 'SimHei'],
    'font.size': 12,
    'axes.linewidth': 1.5,
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8F9FA'
})


@dataclass
class Process:
    """工序信息"""
    process_id: str
    product_type: str
    operation_name: str
    duration: float
    equipment_required: List[str] = field(default_factory=list)
    workers_required: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)


@dataclass
class ProductionUnit:
    """生产单元"""
    unit_id: int
    position: Tuple[int, int]
    assigned_processes: List[Process] = field(default_factory=list)
    capacity: float = 1.0
    utilization: float = 0.0

    def add_process(self, process: Process):
        self.assigned_processes.append(process)

    def calculate_utilization(self, production_plan: Dict[str, int]):
        """计算单元利用率"""
        total_time = 0
        for process in self.assigned_processes:
            if process.product_type in production_plan:
                total_time += process.duration * production_plan[process.product_type]

        # 假设每天工作8小时
        available_time = 8 * 60  # 480分钟
        self.utilization = min(total_time / available_time, 1.0)
        return self.utilization


class ProductionLayoutOptimizer:
    """生产布局优化器"""

    def __init__(self, neo4j_uri: str, neo4j_auth: tuple,
                 grid_size: Tuple[int, int] = (9, 4),
                 output_dir: str = "layout_optimization_results"):
        """
        初始化优化器

        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_auth: Neo4j认证信息
            grid_size: 网格大小 (行, 列)
            output_dir: 输出目录
        """
        self.graph = Graph(neo4j_uri, auth=neo4j_auth)
        self.grid_rows, self.grid_cols = grid_size
        self.n_units = self.grid_rows * self.grid_cols
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)

        # 数据存储
        self.processes = {}  # process_id -> Process
        self.products = {}  # product_type -> List[Process]
        self.flow_matrix = None
        self.distance_matrix = None

        # 算法性能记录
        self.algorithm_history = {
            'GA': {'costs': [], 'utilizations': [], 'time': 0},
            'PSO': {'costs': [], 'utilizations': [], 'time': 0},
            'SA': {'costs': [], 'utilizations': [], 'time': 0},
            'ACO': {'costs': [], 'utilizations': [], 'time': 0},
            'TS': {'costs': [], 'utilizations': [], 'time': 0}
        }

        print(f"🏭 Production Layout Optimizer initialized")
        print(f"📊 Grid size: {self.grid_rows}×{self.grid_cols} = {self.n_units} units")

    def load_production_data(self, production_plan: Dict[str, int]):
        """从Neo4j加载生产数据"""
        print("📚 Loading production data from Neo4j...")

        # 获取产品列表
        product_types = list(production_plan.keys())

        # 查询每个产品的工序信息
        for product_type in product_types:
            query = f"""
            MATCH (prod:Product {{name: '{product_type}'}})-[:PROCESSED_BY]->(ps:Process_set)-[:INCLUDES]->(p:Process)
            OPTIONAL MATCH (p)-[:REQUIRES_EQUIPMENT]->(e:Equipment)
            OPTIONAL MATCH (p)-[:REQUIRES_WORKER]->(w:Worker)
            OPTIONAL MATCH (p)-[:PRECEDES]->(next:Process)
            OPTIONAL MATCH (prev:Process)-[:PRECEDES]->(p)
            RETURN p.process_id as process_id,
                   p.name as process_name,
                   p.process_description as description,
                   coalesce(p.cycle_time, p.standard_work_hours, 30.0) as duration,
                   collect(DISTINCT e.name) as equipment,
                   collect(DISTINCT w.name) as workers,
                   collect(DISTINCT prev.process_id) as predecessors,
                   collect(DISTINCT next.process_id) as successors
            ORDER BY p.process_id
            """

            try:
                results = self.graph.run(query).data()

                if not results:
                    print(f"⚠️ No process data found for {product_type}, using default")
                    self._create_default_processes(product_type)
                else:
                    processes = []
                    for record in results:
                        process = Process(
                            process_id=str(record['process_id']),
                            product_type=product_type,
                            operation_name=record.get('process_name', f"Process_{record['process_id']}"),
                            duration=float(record.get('duration', 30)),
                            equipment_required=record.get('equipment', []),
                            workers_required=record.get('workers', []),
                            predecessors=[str(p) for p in record.get('predecessors', []) if p],
                            successors=[str(s) for s in record.get('successors', []) if s]
                        )

                        self.processes[process.process_id] = process
                        processes.append(process)

                    self.products[product_type] = processes
                    print(f"✅ Loaded {len(processes)} processes for {product_type}")

            except Exception as e:
                print(f"⚠️ Error loading data for {product_type}: {e}")
                self._create_default_processes(product_type)

        # 构建物流矩阵
        self._build_flow_matrix(production_plan)

        # 构建距离矩阵
        self._build_distance_matrix()

        print(f"✅ Production data loaded: {len(self.processes)} total processes")

    def _create_default_processes(self, product_type: str):
        """创建默认工序"""
        n_processes = random.randint(5, 10)
        processes = []

        for i in range(n_processes):
            process = Process(
                process_id=f"{product_type}_P{i + 1}",
                product_type=product_type,
                operation_name=f"Operation_{i + 1}",
                duration=20 + random.randint(10, 40),
                equipment_required=[f"Equipment_{random.randint(1, 5)}"],
                workers_required=[f"Worker_Type_{random.randint(1, 3)}"]
            )

            # 设置前后工序关系
            if i > 0:
                process.predecessors = [f"{product_type}_P{i}"]
            if i < n_processes - 1:
                process.successors = [f"{product_type}_P{i + 2}"]

            self.processes[process.process_id] = process
            processes.append(process)

        self.products[product_type] = processes

    def _build_flow_matrix(self, production_plan: Dict[str, int]):
        """构建工序间物流矩阵"""
        n_processes = len(self.processes)
        process_ids = list(self.processes.keys())
        process_id_to_idx = {pid: i for i, pid in enumerate(process_ids)}

        # 初始化物流矩阵
        self.flow_matrix = np.zeros((n_processes, n_processes))

        # 基于工序前后关系和生产数量计算物流量
        for product_type, quantity in production_plan.items():
            if product_type in self.products:
                processes = self.products[product_type]

                for i in range(len(processes) - 1):
                    from_idx = process_id_to_idx[processes[i].process_id]
                    to_idx = process_id_to_idx[processes[i + 1].process_id]

                    # 物流量 = 产品数量 * 物流频率因子
                    flow_amount = quantity * random.uniform(0.8, 1.2)
                    self.flow_matrix[from_idx, to_idx] += flow_amount

        print(f"📊 Flow matrix built: {n_processes}×{n_processes}")

    def _build_distance_matrix(self):
        """构建单元间距离矩阵"""
        self.distance_matrix = np.zeros((self.n_units, self.n_units))

        for i in range(self.n_units):
            row_i, col_i = i // self.grid_cols, i % self.grid_cols

            for j in range(self.n_units):
                if i != j:
                    row_j, col_j = j // self.grid_cols, j % self.grid_cols

                    # 曼哈顿距离
                    distance = abs(row_i - row_j) + abs(col_i - col_j)
                    self.distance_matrix[i, j] = distance

    def evaluate_solution(self, solution: np.ndarray, production_plan: Dict[str, int]) -> Tuple[float, float]:
        """
        评估解决方案

        Args:
            solution: 工序到单元的分配方案
            production_plan: 生产计划

        Returns:
            (物流成本, 平均利用率)
        """
        # 计算物流成本
        total_cost = 0
        n_processes = len(self.processes)

        for i in range(n_processes):
            for j in range(n_processes):
                if self.flow_matrix[i, j] > 0:
                    unit_i = solution[i]
                    unit_j = solution[j]
                    cost = self.flow_matrix[i, j] * self.distance_matrix[unit_i, unit_j]
                    total_cost += cost

        # 计算资源利用率
        units = [ProductionUnit(unit_id=i, position=(i // self.grid_cols, i % self.grid_cols))
                 for i in range(self.n_units)]

        process_list = list(self.processes.values())
        for i, process in enumerate(process_list):
            unit_id = solution[i]
            units[unit_id].add_process(process)

        utilizations = [unit.calculate_utilization(production_plan) for unit in units]
        avg_utilization = np.mean(utilizations)

        return total_cost, avg_utilization

    # ==================== 优化算法实现 ====================

    def genetic_algorithm(self, production_plan: Dict[str, int],
                          pop_size: int = 100, generations: int = 200,
                          crossover_rate: float = 0.8, mutation_rate: float = 0.1):
        """遗传算法"""
        print("\n🧬 Running Genetic Algorithm...")
        start_time = time.time()

        n_processes = len(self.processes)

        # 初始化种群
        population = [np.random.randint(0, self.n_units, n_processes) for _ in range(pop_size)]

        best_solution = None
        best_cost = float('inf')
        best_utilization = 0

        history = {'costs': [], 'utilizations': []}

        for gen in range(generations):
            # 评估种群
            fitness_scores = []
            for individual in population:
                cost, utilization = self.evaluate_solution(individual, production_plan)
                # 多目标适应度：最小化成本，最大化利用率
                fitness = cost - 1000 * utilization  # 权衡因子
                fitness_scores.append((fitness, cost, utilization, individual))

            # 排序选择最优
            fitness_scores.sort(key=lambda x: x[0])

            if fitness_scores[0][1] < best_cost:
                best_cost = fitness_scores[0][1]
                best_utilization = fitness_scores[0][2]
                best_solution = fitness_scores[0][3].copy()

            history['costs'].append(best_cost)
            history['utilizations'].append(best_utilization)

            # 选择
            parents = [x[3] for x in fitness_scores[:pop_size // 2]]

            # 交叉和变异
            new_population = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    parent1, parent2 = parents[i], parents[i + 1]

                    # 交叉
                    if random.random() < crossover_rate:
                        crossover_point = random.randint(1, n_processes - 1)
                        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()

                    # 变异
                    for child in [child1, child2]:
                        if random.random() < mutation_rate:
                            mutation_point = random.randint(0, n_processes - 1)
                            child[mutation_point] = random.randint(0, self.n_units - 1)

                    new_population.extend([child1, child2])

            population = parents + new_population[:pop_size - len(parents)]

            if gen % 50 == 0:
                print(f"  Generation {gen}: Cost={best_cost:.2f}, Utilization={best_utilization:.2%}")

        elapsed_time = time.time() - start_time
        self.algorithm_history['GA']['costs'] = history['costs']
        self.algorithm_history['GA']['utilizations'] = history['utilizations']
        self.algorithm_history['GA']['time'] = elapsed_time

        print(f"✅ GA completed in {elapsed_time:.2f}s: Cost={best_cost:.2f}, Utilization={best_utilization:.2%}")
        return best_solution, best_cost, best_utilization

    def particle_swarm_optimization(self, production_plan: Dict[str, int],
                                    n_particles: int = 50, iterations: int = 200,
                                    w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """粒子群优化算法"""
        print("\n🐦 Running Particle Swarm Optimization...")
        start_time = time.time()

        n_processes = len(self.processes)

        # 初始化粒子
        particles = np.random.rand(n_particles, n_processes) * self.n_units
        velocities = np.random.randn(n_particles, n_processes) * 0.1

        # 个体最优和全局最优
        p_best = particles.copy()
        p_best_scores = np.full(n_particles, float('inf'))
        g_best = None
        g_best_score = float('inf')
        g_best_utilization = 0

        history = {'costs': [], 'utilizations': []}

        for iter in range(iterations):
            for i in range(n_particles):
                # 将连续位置转换为离散单元分配
                position = np.round(particles[i]).astype(int)
                position = np.clip(position, 0, self.n_units - 1)

                cost, utilization = self.evaluate_solution(position, production_plan)

                # 更新个体最优
                if cost < p_best_scores[i]:
                    p_best_scores[i] = cost
                    p_best[i] = particles[i].copy()

                # 更新全局最优
                if cost < g_best_score:
                    g_best_score = cost
                    g_best_utilization = utilization
                    g_best = particles[i].copy()

            history['costs'].append(g_best_score)
            history['utilizations'].append(g_best_utilization)

            # 更新速度和位置
            for i in range(n_particles):
                r1, r2 = random.random(), random.random()
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (p_best[i] - particles[i]) +
                                 c2 * r2 * (g_best - particles[i]))
                particles[i] += velocities[i]

                # 边界处理
                particles[i] = np.clip(particles[i], 0, self.n_units - 1)

            if iter % 50 == 0:
                print(f"  Iteration {iter}: Cost={g_best_score:.2f}, Utilization={g_best_utilization:.2%}")

        # 最终解
        best_solution = np.round(g_best).astype(int)
        best_solution = np.clip(best_solution, 0, self.n_units - 1)

        elapsed_time = time.time() - start_time
        self.algorithm_history['PSO']['costs'] = history['costs']
        self.algorithm_history['PSO']['utilizations'] = history['utilizations']
        self.algorithm_history['PSO']['time'] = elapsed_time

        print(f"✅ PSO completed in {elapsed_time:.2f}s: Cost={g_best_score:.2f}, Utilization={g_best_utilization:.2%}")
        return best_solution, g_best_score, g_best_utilization

    def simulated_annealing(self, production_plan: Dict[str, int],
                            initial_temp: float = 1000, cooling_rate: float = 0.95,
                            iterations: int = 10000):
        """模拟退火算法"""
        print("\n🔥 Running Simulated Annealing...")
        start_time = time.time()

        n_processes = len(self.processes)

        # 初始解
        current_solution = np.random.randint(0, self.n_units, n_processes)
        current_cost, current_utilization = self.evaluate_solution(current_solution, production_plan)

        best_solution = current_solution.copy()
        best_cost = current_cost
        best_utilization = current_utilization

        temperature = initial_temp
        history = {'costs': [], 'utilizations': []}

        for iter in range(iterations):
            # 生成邻域解
            neighbor = current_solution.copy()

            # 随机选择交换或移动操作
            if random.random() < 0.5:
                # 交换两个工序的单元分配
                i, j = random.sample(range(n_processes), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            else:
                # 随机移动一个工序到新单元
                i = random.randint(0, n_processes - 1)
                neighbor[i] = random.randint(0, self.n_units - 1)

            neighbor_cost, neighbor_utilization = self.evaluate_solution(neighbor, production_plan)

            # 接受准则
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_solution = neighbor
                current_cost = neighbor_cost
                current_utilization = neighbor_utilization

                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
                    best_utilization = current_utilization

            # 降温
            if iter % 100 == 0:
                temperature *= cooling_rate
                history['costs'].append(best_cost)
                history['utilizations'].append(best_utilization)

            if iter % 2000 == 0:
                print(
                    f"  Iteration {iter}: Cost={best_cost:.2f}, Utilization={best_utilization:.2%}, Temp={temperature:.2f}")

        elapsed_time = time.time() - start_time
        self.algorithm_history['SA']['costs'] = history['costs']
        self.algorithm_history['SA']['utilizations'] = history['utilizations']
        self.algorithm_history['SA']['time'] = elapsed_time

        print(f"✅ SA completed in {elapsed_time:.2f}s: Cost={best_cost:.2f}, Utilization={best_utilization:.2%}")
        return best_solution, best_cost, best_utilization

    def ant_colony_optimization(self, production_plan: Dict[str, int],
                                n_ants: int = 50, iterations: int = 100,
                                alpha: float = 1.0, beta: float = 2.0,
                                evaporation_rate: float = 0.1):
        """蚁群算法"""
        print("\n🐜 Running Ant Colony Optimization...")
        start_time = time.time()

        n_processes = len(self.processes)

        # 初始化信息素
        pheromone = np.ones((n_processes, self.n_units)) * 0.1

        best_solution = None
        best_cost = float('inf')
        best_utilization = 0

        history = {'costs': [], 'utilizations': []}

        for iter in range(iterations):
            solutions = []

            # 每只蚂蚁构建解决方案
            for ant in range(n_ants):
                solution = np.zeros(n_processes, dtype=int)

                for i in range(n_processes):
                    # 计算选择概率
                    probabilities = []

                    for j in range(self.n_units):
                        # 启发式信息：单元负载均衡
                        unit_load = np.sum(solution == j)
                        eta = 1.0 / (1.0 + unit_load)

                        # 概率计算
                        prob = (pheromone[i, j] ** alpha) * (eta ** beta)
                        probabilities.append(prob)

                    # 轮盘赌选择
                    probabilities = np.array(probabilities)
                    probabilities /= probabilities.sum()

                    solution[i] = np.random.choice(self.n_units, p=probabilities)

                cost, utilization = self.evaluate_solution(solution, production_plan)
                solutions.append((solution, cost, utilization))

                if cost < best_cost:
                    best_cost = cost
                    best_utilization = utilization
                    best_solution = solution.copy()

            history['costs'].append(best_cost)
            history['utilizations'].append(best_utilization)

            # 更新信息素
            pheromone *= (1 - evaporation_rate)  # 蒸发

            for solution, cost, _ in solutions:
                # 信息素增强
                for i in range(n_processes):
                    pheromone[i, solution[i]] += 1.0 / cost

            if iter % 20 == 0:
                print(f"  Iteration {iter}: Cost={best_cost:.2f}, Utilization={best_utilization:.2%}")

        elapsed_time = time.time() - start_time
        self.algorithm_history['ACO']['costs'] = history['costs']
        self.algorithm_history['ACO']['utilizations'] = history['utilizations']
        self.algorithm_history['ACO']['time'] = elapsed_time

        print(f"✅ ACO completed in {elapsed_time:.2f}s: Cost={best_cost:.2f}, Utilization={best_utilization:.2%}")
        return best_solution, best_cost, best_utilization

    def tabu_search(self, production_plan: Dict[str, int],
                    tabu_size: int = 20, iterations: int = 500):
        """禁忌搜索算法"""
        print("\n🚫 Running Tabu Search...")
        start_time = time.time()

        n_processes = len(self.processes)

        # 初始解
        current_solution = np.random.randint(0, self.n_units, n_processes)
        current_cost, current_utilization = self.evaluate_solution(current_solution, production_plan)

        best_solution = current_solution.copy()
        best_cost = current_cost
        best_utilization = current_utilization

        # 禁忌表
        tabu_list = []

        history = {'costs': [], 'utilizations': []}

        for iter in range(iterations):
            # 生成候选邻域
            candidates = []

            for _ in range(20):  # 生成20个候选解
                neighbor = current_solution.copy()

                # 随机选择移动类型
                move_type = random.choice(['swap', 'relocate'])

                if move_type == 'swap':
                    i, j = random.sample(range(n_processes), 2)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    move = ('swap', i, j)
                else:
                    i = random.randint(0, n_processes - 1)
                    new_unit = random.randint(0, self.n_units - 1)
                    neighbor[i] = new_unit
                    move = ('relocate', i, new_unit)

                if move not in tabu_list:
                    cost, utilization = self.evaluate_solution(neighbor, production_plan)
                    candidates.append((neighbor, cost, utilization, move))

            if candidates:
                # 选择最佳候选
                candidates.sort(key=lambda x: x[1])
                best_candidate = candidates[0]

                current_solution = best_candidate[0]
                current_cost = best_candidate[1]
                current_utilization = best_candidate[2]

                # 更新禁忌表
                tabu_list.append(best_candidate[3])
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0)

                # 更新全局最优
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
                    best_utilization = current_utilization

            if iter % 10 == 0:
                history['costs'].append(best_cost)
                history['utilizations'].append(best_utilization)

            if iter % 100 == 0:
                print(f"  Iteration {iter}: Cost={best_cost:.2f}, Utilization={best_utilization:.2%}")

        elapsed_time = time.time() - start_time
        self.algorithm_history['TS']['costs'] = history['costs']
        self.algorithm_history['TS']['utilizations'] = history['utilizations']
        self.algorithm_history['TS']['time'] = elapsed_time

        print(f"✅ TS completed in {elapsed_time:.2f}s: Cost={best_cost:.2f}, Utilization={best_utilization:.2%}")
        return best_solution, best_cost, best_utilization

    def run_all_algorithms(self, production_plan: Dict[str, int]):
        """运行所有算法并比较"""
        print("\n🚀 Running all optimization algorithms...")

        results = {}

        # 遗传算法
        ga_solution, ga_cost, ga_utilization = self.genetic_algorithm(production_plan)
        results['GA'] = {'solution': ga_solution, 'cost': ga_cost, 'utilization': ga_utilization}

        # 粒子群算法
        pso_solution, pso_cost, pso_utilization = self.particle_swarm_optimization(production_plan)
        results['PSO'] = {'solution': pso_solution, 'cost': pso_cost, 'utilization': pso_utilization}

        # 模拟退火
        sa_solution, sa_cost, sa_utilization = self.simulated_annealing(production_plan)
        results['SA'] = {'solution': sa_solution, 'cost': sa_cost, 'utilization': sa_utilization}

        # 蚁群算法
        aco_solution, aco_cost, aco_utilization = self.ant_colony_optimization(production_plan)
        results['ACO'] = {'solution': aco_solution, 'cost': aco_cost, 'utilization': aco_utilization}

        # 禁忌搜索
        ts_solution, ts_cost, ts_utilization = self.tabu_search(production_plan)
        results['TS'] = {'solution': ts_solution, 'cost': ts_cost, 'utilization': ts_utilization}

        # 找出最佳算法
        best_algorithm = min(results.keys(), key=lambda x: results[x]['cost'])

        print(f"\n🏆 Best Algorithm: {best_algorithm}")
        print(f"   Cost: {results[best_algorithm]['cost']:.2f}")
        print(f"   Utilization: {results[best_algorithm]['utilization']:.2%}")

        return results, best_algorithm

    def visualize_results(self, results: Dict, production_plan: Dict[str, int]):
        """可视化优化结果"""
        print("\n📊 Generating visualization...")

        # 创建图形布局
        fig = plt.figure(figsize=(24, 20))
        gs = plt.GridSpec(4, 3, height_ratios=[1.5, 1.5, 1.5, 0.8],
                          hspace=0.3, wspace=0.25)

        # 1. 算法性能对比
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_algorithm_comparison(ax1, results)

        # 2. 收敛曲线
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_convergence_curves(ax2, 'costs', 'Material Flow Cost')

        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_convergence_curves(ax3, 'utilizations', 'Resource Utilization')

        # 3. 最佳布局可视化
        ax4 = fig.add_subplot(gs[1, 2])
        best_algorithm = min(results.keys(), key=lambda x: results[x]['cost'])
        self._plot_layout(ax4, results[best_algorithm]['solution'], production_plan)

        # 4. 利用率热力图
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_utilization_heatmap(ax5, results[best_algorithm]['solution'], production_plan)

        # 5. 物流强度矩阵
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_flow_intensity_matrix(ax6, results[best_algorithm]['solution'])

        # 6. 算法运行时间对比
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_runtime_comparison(ax7)

        # 7. 总结统计
        summary_ax = fig.add_subplot(gs[3, :])
        self._create_summary_statistics(summary_ax, results, production_plan)

        # 保存图形
        plt.suptitle('Multi-Algorithm Production Layout Optimization Results',
                     fontsize=24, fontweight='bold', y=0.98)

        output_path = os.path.join(self.output_dir, "images", "optimization_results.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Visualization saved to: {output_path}")

    def _plot_algorithm_comparison(self, ax, results):
        """绘制算法性能对比"""
        algorithms = list(results.keys())
        costs = [results[alg]['cost'] for alg in algorithms]
        utilizations = [results[alg]['utilization'] * 100 for alg in algorithms]

        x = np.arange(len(algorithms))
        width = 0.35

        # 创建双Y轴
        ax2 = ax.twinx()

        # 物流成本条形图
        bars1 = ax.bar(x - width / 2, costs, width, label='Material Flow Cost',
                       color='#FF6B6B', alpha=0.8)

        # 利用率条形图
        bars2 = ax2.bar(x + width / 2, utilizations, width, label='Utilization (%)',
                        color='#4ECDC4', alpha=0.8)

        # 添加数值标签
        for bar, cost in zip(bars1, costs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{cost:.0f}', ha='center', va='bottom', fontsize=10)

        for bar, util in zip(bars2, utilizations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{util:.1f}%', ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Optimization Algorithms', fontsize=14, fontweight='bold')
        ax.set_ylabel('Material Flow Cost', fontsize=14, fontweight='bold', color='#FF6B6B')
        ax2.set_ylabel('Resource Utilization (%)', fontsize=14, fontweight='bold', color='#4ECDC4')
        ax.set_title('Algorithm Performance Comparison', fontsize=16, fontweight='bold', pad=20)

        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.tick_params(axis='y', labelcolor='#FF6B6B')
        ax2.tick_params(axis='y', labelcolor='#4ECDC4')

        # 添加图例
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax.grid(True, alpha=0.3)

    def _plot_convergence_curves(self, ax, metric, title):
        """绘制收敛曲线"""
        for alg, data in self.algorithm_history.items():
            if data[metric]:
                iterations = range(len(data[metric]))
                ax.plot(iterations, data[metric], label=alg, linewidth=2, alpha=0.8)

        ax.set_xlabel('Iterations', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} Convergence', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_layout(self, ax, solution, production_plan):
        """绘制布局方案"""
        # 创建单元网格
        for i in range(self.n_units):
            row, col = i // self.grid_cols, i % self.grid_cols

            # 计算该单元的利用率
            unit = ProductionUnit(unit_id=i, position=(row, col))
            process_list = list(self.processes.values())

            for j, process in enumerate(process_list):
                if solution[j] == i:
                    unit.add_process(process)

            utilization = unit.calculate_utilization(production_plan)

            # 根据利用率选择颜色
            if utilization > 0.6:
                color = '#FF6B6B'  # 高利用率 - 红色
            elif utilization > 0.3:
                color = '#FFE66D'  # 中利用率 - 黄色
            else:
                color = '#A8E6CF'  # 低利用率 - 绿色

            rect = Rectangle((col, row), 1, 1, facecolor=color,
                             edgecolor='black', linewidth=1, alpha=0.8)
            ax.add_patch(rect)

            # 添加单元编号和利用率
            ax.text(col + 0.5, row + 0.5, f'U{i}\n{utilization:.0%}',
                    ha='center', va='center', fontsize=8, fontweight='bold')

        ax.set_xlim(0, self.grid_cols)
        ax.set_ylim(0, self.grid_rows)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title('Optimal Production Layout', fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)

        # 添加网格线
        for i in range(self.grid_rows + 1):
            ax.axhline(i, color='gray', linewidth=0.5)
        for j in range(self.grid_cols + 1):
            ax.axvline(j, color='gray', linewidth=0.5)

    def _plot_utilization_heatmap(self, ax, solution, production_plan):
        """绘制利用率热力图"""
        utilization_matrix = np.zeros((self.grid_rows, self.grid_cols))

        for i in range(self.n_units):
            row, col = i // self.grid_cols, i % self.grid_cols

            unit = ProductionUnit(unit_id=i, position=(row, col))
            process_list = list(self.processes.values())

            for j, process in enumerate(process_list):
                if solution[j] == i:
                    unit.add_process(process)

            utilization = unit.calculate_utilization(production_plan)
            utilization_matrix[row, col] = utilization

        im = ax.imshow(utilization_matrix, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')

        # 添加数值标注
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                ax.text(j, i, f'{utilization_matrix[i, j]:.0%}',
                        ha='center', va='center', fontsize=10)

        ax.set_title('Resource Utilization Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Utilization Rate', fontsize=12)

    def _plot_flow_intensity_matrix(self, ax, solution):
        """绘制物流强度矩阵"""
        # 计算单元间物流量
        unit_flow_matrix = np.zeros((self.n_units, self.n_units))

        process_list = list(self.processes.values())
        for i in range(len(process_list)):
            for j in range(len(process_list)):
                if self.flow_matrix[i, j] > 0:
                    unit_i = solution[i]
                    unit_j = solution[j]
                    unit_flow_matrix[unit_i, unit_j] += self.flow_matrix[i, j]

        # 只显示前20个单元的物流关系
        display_units = min(20, self.n_units)
        display_matrix = unit_flow_matrix[:display_units, :display_units]

        im = ax.imshow(display_matrix, cmap='Reds', alpha=0.8)

        # 设置刻度
        ax.set_xticks(range(display_units))
        ax.set_yticks(range(display_units))
        ax.set_xticklabels([f'U{i}' for i in range(display_units)], rotation=45)
        ax.set_yticklabels([f'U{i}' for i in range(display_units)])

        # 添加网格
        ax.set_xticks([i - 0.5 for i in range(display_units + 1)], minor=True)
        ax.set_yticks([i - 0.5 for i in range(display_units + 1)], minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

        # 添加数值标注（只显示较大的值）
        threshold = np.max(display_matrix) * 0.3
        for i in range(display_units):
            for j in range(display_units):
                if display_matrix[i, j] > threshold:
                    ax.text(j, i, f'{display_matrix[i, j]:.0f}',
                            ha='center', va='center', fontsize=8)

        ax.set_title('Material Flow Intensity Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

    def _plot_runtime_comparison(self, ax):
        """绘制运行时间对比"""
        algorithms = list(self.algorithm_history.keys())
        runtimes = [self.algorithm_history[alg]['time'] for alg in algorithms]

        bars = ax.bar(algorithms, runtimes, color='#667eea', alpha=0.8)

        # 添加数值标签
        for bar, time in zip(bars, runtimes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{time:.1f}s', ha='center', va='bottom')

        ax.set_ylabel('Runtime (seconds)', fontsize=12)
        ax.set_title('Algorithm Runtime Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    def _create_summary_statistics(self, ax, results, production_plan):
        """创建总结统计信息"""
        ax.axis('off')

        # 计算统计数据
        best_algorithm = min(results.keys(), key=lambda x: results[x]['cost'])
        best_cost = results[best_algorithm]['cost']
        best_utilization = results[best_algorithm]['utilization']

        avg_cost = np.mean([results[alg]['cost'] for alg in results])
        avg_utilization = np.mean([results[alg]['utilization'] for alg in results])

        total_processes = len(self.processes)
        total_products = len(production_plan)
        total_quantity = sum(production_plan.values())

        # 创建统计文本
        summary_text = f"""
        📊 OPTIMIZATION SUMMARY

        🏆 Best Algorithm: {best_algorithm}
        💰 Best Material Flow Cost: {best_cost:.2f}
        ⚡ Best Resource Utilization: {best_utilization:.2%}

        📈 Average Performance:
        • Average Cost: {avg_cost:.2f}
        • Average Utilization: {avg_utilization:.2%}

        🏭 Production Details:
        • Total Products: {total_products}
        • Total Processes: {total_processes}
        • Total Production Quantity: {total_quantity} units
        • Grid Size: {self.grid_rows}×{self.grid_cols} = {self.n_units} units
        """

        improvement_text = f"""
        💡 PERFORMANCE IMPROVEMENTS

        🎯 Cost Reduction:
        • Best vs Average: {((avg_cost - best_cost) / avg_cost * 100):.1f}%
        • Best vs Worst: {((max(results[alg]['cost'] for alg in results) - best_cost) / max(results[alg]['cost'] for alg in results) * 100):.1f}%

        📊 Utilization Enhancement:
        • Best Algorithm: {(best_utilization * 100):.1f}%
        • Improvement over random: ~{((best_utilization - 0.3) / 0.3 * 100):.1f}%

        ⏱️ Computational Efficiency:
        • Fastest Algorithm: {min(self.algorithm_history.keys(), key=lambda x: self.algorithm_history[x]['time'])}
        • Total Runtime: {sum(self.algorithm_history[alg]['time'] for alg in self.algorithm_history):.1f} seconds
        """

        # 显示统计信息
        ax.text(0.05, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#E8F4FD', alpha=0.8),
                verticalalignment='center', fontweight='bold')

        ax.text(0.55, 0.5, improvement_text, transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#F0F9FF', alpha=0.8),
                verticalalignment='center')

        # 添加时间戳
        ax.text(0.98, 0.02, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                color='gray', alpha=0.7, style='italic')

    def save_results(self, results, production_plan):
        """保存优化结果"""
        print("\n💾 Saving optimization results...")

        # 保存算法比较数据
        comparison_data = {
            'algorithms': {},
            'production_plan': production_plan,
            'grid_size': {'rows': self.grid_rows, 'cols': self.grid_cols},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        for alg, result in results.items():
            comparison_data['algorithms'][alg] = {
                'cost': float(result['cost']),
                'utilization': float(result['utilization']),
                'runtime': float(self.algorithm_history[alg]['time']),
                'solution': result['solution'].tolist()
            }

        # 保存JSON文件
        json_path = os.path.join(self.output_dir, "data", "optimization_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Results saved to: {json_path}")

        # 保存最佳解决方案的详细信息
        best_algorithm = min(results.keys(), key=lambda x: results[x]['cost'])
        best_solution = results[best_algorithm]['solution']

        # 创建单元分配报告
        unit_assignment = defaultdict(list)
        process_list = list(self.processes.values())

        for i, process in enumerate(process_list):
            unit_id = best_solution[i]
            unit_assignment[unit_id].append({
                'process_id': process.process_id,
                'product_type': process.product_type,
                'operation': process.operation_name,
                'duration': process.duration
            })

        # 保存单元分配报告
        assignment_path = os.path.join(self.output_dir, "data", "unit_assignments.json")
        with open(assignment_path, 'w', encoding='utf-8') as f:
            json.dump(dict(unit_assignment), f, ensure_ascii=False, indent=2)

        print(f"✅ Unit assignments saved to: {assignment_path}")


def main():
    """主函数"""
    # Neo4j连接配置
    neo4j_uri = "bolt://localhost:7687"
    neo4j_auth = ("neo4j", "dididaodao")

    # 生产计划
    production_plan = {
        "K3473": 10,
        "K16842": 5,
        "K14386": 7,
        "K1786": 6,
        "K6286": 12,
        "K15022": 10,
        "K15126": 9
    }

    # 创建优化器
    optimizer = ProductionLayoutOptimizer(neo4j_uri, neo4j_auth, grid_size=(9, 4))

    # 加载生产数据
    optimizer.load_production_data(production_plan)

    # 运行所有优化算法
    results, best_algorithm = optimizer.run_all_algorithms(production_plan)

    # 可视化结果
    optimizer.visualize_results(results, production_plan)

    # 保存结果
    optimizer.save_results(results, production_plan)

    print("\n✅ Optimization completed successfully!")
    print(f"📁 Results saved in: {optimizer.output_dir}")

    return optimizer, results


if __name__ == "__main__":
    optimizer, results = main()