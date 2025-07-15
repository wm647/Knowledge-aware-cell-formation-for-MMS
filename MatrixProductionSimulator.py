#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix Production Simulation System with Enhanced Visualization
基于布局结果的矩阵式生产仿真系统（增强可视化版）

Features:
1. Discrete event simulation with proper waiting time tracking
2. Strict resource constraints (one job per unit at a time)
3. High-quality scientific visualization
4. Gantt chart with waiting time visualization (both HTML and image)
5. All outputs to organized directory structure
6. High-resolution image export (DPI=1200)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any
import json
import time
import heapq
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import random
import os

# Knowledge graph
from py2neo import Graph

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


class JobStatus(Enum):
    """Job status enumeration"""
    WAITING = "waiting"
    PROCESSING = "processing"
    TRANSPORTING = "transporting"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    IN_QUEUE = "in_queue"


class EventType(Enum):
    """Event type enumeration"""
    JOB_ARRIVAL = "job_arrival"
    PROCESS_START = "process_start"
    PROCESS_COMPLETE = "process_complete"
    TRANSPORT_START = "transport_start"
    TRANSPORT_COMPLETE = "transport_complete"
    ENTER_QUEUE = "enter_queue"
    EXIT_QUEUE = "exit_queue"


@dataclass
class ProcessInfo:
    """Process information"""
    process_id: str
    process_name: str
    duration: float  # Processing time (minutes)
    setup_time: float = 0.0  # Setup time
    required_equipment: List[str] = field(default_factory=list)
    required_workers: List[str] = field(default_factory=list)
    required_materials: List[str] = field(default_factory=list)
    skill_level: int = 1
    priority: int = 1


@dataclass
class Job:
    """Job object with enhanced tracking"""
    job_id: str
    product_type: str
    quantity: int
    order_date: datetime
    due_date: datetime
    current_process_idx: int = 0
    status: JobStatus = JobStatus.WAITING
    current_unit: Optional[int] = None
    process_sequence: List[str] = field(default_factory=list)
    completed_processes: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    completion_time: Optional[float] = None

    # Enhanced time tracking
    queue_enter_times: List[float] = field(default_factory=list)
    queue_exit_times: List[float] = field(default_factory=list)
    process_start_times: List[float] = field(default_factory=list)
    process_end_times: List[float] = field(default_factory=list)
    wait_times: List[float] = field(default_factory=list)
    processing_times: List[float] = field(default_factory=list)
    transport_times: List[float] = field(default_factory=list)

    # For constraint checking
    last_process_end_time: float = 0.0
    is_processing: bool = False


@dataclass
class SimulationEvent:
    """Simulation event"""
    time: float
    event_type: EventType
    job_id: str
    unit_id: Optional[int] = None
    process_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        return self.time < other.time


@dataclass
class ProductionUnit:
    """Production unit with enhanced state tracking"""
    unit_id: int
    config_id: str
    position: Tuple[float, float]
    available_processes: List[str]
    available_equipment: List[str]
    available_workers: List[str]
    current_job: Optional[Job] = None
    queue: deque = field(default_factory=deque)
    utilization: float = 0.0
    total_busy_time: float = 0.0
    total_idle_time: float = 0.0
    setup_time_remaining: float = 0.0
    last_process_type: Optional[str] = None

    # Enhanced tracking
    is_busy: bool = False
    busy_start_time: Optional[float] = None
    queue_wait_times: List[float] = field(default_factory=list)


class MatrixProductionSimulator:
    """Matrix production simulator with enhanced constraints"""

    def __init__(self, neo4j_uri: str, neo4j_auth: tuple,
                 layout_data: dict, clusters_data: dict,
                 simulation_config: dict = None,
                 output_dir: str = "Cell layout output"):
        """
        Initialize production simulator

        Args:
            neo4j_uri: Neo4j database URI
            neo4j_auth: Neo4j authentication
            layout_data: Layout data
            clusters_data: Process configuration data
            simulation_config: Simulation configuration
            output_dir: Output directory
        """
        self.graph = Graph(neo4j_uri, auth=neo4j_auth)
        self.layout_data = layout_data
        self.clusters_data = clusters_data
        self.output_dir = output_dir

        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)

        # Default simulation configuration
        default_config = {
            "time_unit": "minutes",
            "simulation_speed": 1.0,
            "transport_speed": 5.0,
            "setup_time_factor": 0.1,
            "queue_capacity": 10,
            "random_seed": 42
        }
        self.config = {**default_config, **(simulation_config or {})}

        # Simulation state
        self.current_time = 0.0
        self.event_queue = []
        self.jobs = {}
        self.units = {}
        self.process_info = {}
        self.product_routes = {}

        # Statistics
        self.performance_metrics = {
            "completed_jobs": 0,
            "total_throughput": 0,
            "average_cycle_time": 0.0,
            "average_wait_time": 0.0,
            "average_queue_time": 0.0,
            "unit_utilizations": {},
            "wip_levels": [],
            "queue_lengths": [],
            "constraint_violations": 0
        }

        # Visualization data
        self.visualization_data = {
            "time_series": [],
            "unit_states": [],
            "job_flows": [],
            "performance_history": []
        }

        # Enhanced Gantt data
        self.gantt_data = []

        # Constraint tracking
        self.active_jobs_per_unit = defaultdict(list)
        self.job_process_history = defaultdict(list)

        random.seed(self.config["random_seed"])
        print(f"🏭 Matrix Production Simulator Initialized")
        print(f"⏱️ Time unit: {self.config['time_unit']}")
        print(f"📁 Output directory: {self.output_dir}")

    def initialize_simulation(self):
        """Initialize simulation environment"""
        print("🔧 Initializing simulation environment...")

        # 1. Initialize production units
        self._initialize_production_units()

        # 2. Load process information from knowledge graph
        self._load_process_information()

        # 3. Build product routes
        self._build_product_routes()

        # 4. Initialize transport matrix
        self._initialize_transport_matrix()

        print("✅ Simulation environment initialized")

    def _initialize_production_units(self):
        """Initialize production units"""
        unit_centers = np.array(self.layout_data['fixed_unit_centers'])
        config_to_unit = self.layout_data['config_to_unit']

        for config_id, unit_id in config_to_unit.items():
            if config_id in self.clusters_data:
                config_data = self.clusters_data[config_id]
                position = tuple(unit_centers[unit_id])

                # Get available processes
                available_processes = []
                for process_info in config_data['process_ids']:
                    available_processes.append(str(process_info['pid']))

                self.units[unit_id] = ProductionUnit(
                    unit_id=unit_id,
                    config_id=config_id,
                    position=position,
                    available_processes=available_processes,
                    available_equipment=config_data['equipment_list'],
                    available_workers=config_data.get('worker_list', [])
                )

        print(f"📍 Initialized {len(self.units)} production units")

    def _load_process_information(self):
        """Load process information from knowledge graph"""
        print("📊 Loading process information from knowledge graph...")

        # Get all process IDs
        all_process_ids = []
        for unit in self.units.values():
            all_process_ids.extend(unit.available_processes)

        if not all_process_ids:
            print("⚠️ No process data found, using defaults")
            self._create_default_process_info()
            return

        # Query process details
        cypher = f"""
        MATCH (p:Process)
        WHERE p.process_id IN {all_process_ids}
        OPTIONAL MATCH (p)-[:USES_EQUIPMENT]->(e:Equipment)
        OPTIONAL MATCH (p)-[:REQUIRES_SKILL]->(s:Skill)
        OPTIONAL MATCH (p)-[:USES_MATERIAL]->(m:Material)
        RETURN p.process_id as process_id,
               p.name as process_name,
               coalesce(p.duration, 30.0) as duration,
               coalesce(p.setup_time, 5.0) as setup_time,
               coalesce(p.priority, 1) as priority,
               coalesce(p.skill_level, 1) as skill_level,
               collect(DISTINCT e.equipment_id) as equipment,
               collect(DISTINCT s.skill_name) as skills,
               collect(DISTINCT m.material_id) as materials
        """

        try:
            results = self.graph.run(cypher).data()

            for record in results:
                process_id = str(record['process_id'])

                self.process_info[process_id] = ProcessInfo(
                    process_id=process_id,
                    process_name=record.get('process_name', f'Process_{process_id}'),
                    duration=float(record.get('duration', 30.0)),
                    setup_time=float(record.get('setup_time', 5.0)),
                    required_equipment=record.get('equipment', []),
                    required_workers=record.get('skills', []),
                    required_materials=record.get('materials', []),
                    skill_level=int(record.get('skill_level', 1)),
                    priority=int(record.get('priority', 1))
                )

            print(f"✅ Loaded {len(self.process_info)} process information records")

        except Exception as e:
            print(f"⚠️ Error loading process information: {e}")
            self._create_default_process_info()

    def _create_default_process_info(self):
        """Create default process information"""
        for unit in self.units.values():
            for process_id in unit.available_processes:
                if process_id not in self.process_info:
                    # Generate reasonable processing time based on process ID
                    base_time = 20 + (hash(process_id) % 40)  # 20-60 minutes

                    self.process_info[process_id] = ProcessInfo(
                        process_id=process_id,
                        process_name=f"Process_{process_id}",
                        duration=base_time,
                        setup_time=base_time * 0.1,
                        priority=1
                    )

    def _build_product_routes(self):
        """Build product process routes"""
        print("🛣️ Building product process routes...")

        # Infer product routes from process configuration data
        product_processes = defaultdict(list)

        for config_id, config_data in self.clusters_data.items():
            for process_info in config_data['process_ids']:
                product_type = process_info.get('bm', 'unknown')
                process_id = str(process_info['pid'])

                if product_type != 'unknown':
                    product_processes[product_type].append(process_id)

        # Build process routes (simplified: sort by process ID)
        for product_type, process_list in product_processes.items():
            sorted_processes = sorted(process_list)
            self.product_routes[product_type] = sorted_processes

        print(f"🎯 Built {len(self.product_routes)} product process routes")
        for product, route in self.product_routes.items():
            print(f"   {product}: {len(route)} processes")

    def _initialize_transport_matrix(self):
        """Initialize transport time matrix"""
        n_units = len(self.units)
        unit_ids = list(self.units.keys())

        self.transport_matrix = {}

        for i, unit1 in enumerate(unit_ids):
            for j, unit2 in enumerate(unit_ids):
                if unit1 != unit2:
                    pos1 = self.units[unit1].position
                    pos2 = self.units[unit2].position

                    # Calculate Euclidean distance
                    distance = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

                    # Transport time = distance / speed + loading/unloading time
                    transport_time = distance / self.config['transport_speed'] + 2.0

                    self.transport_matrix[(unit1, unit2)] = transport_time
                else:
                    self.transport_matrix[(unit1, unit2)] = 0.0

    def create_production_plan(self, order_plan: Dict[str, int]) -> List[Job]:
        """Create production jobs from order plan"""
        print(f"📋 Creating production plan: {order_plan}")

        jobs = []
        job_counter = 0

        for product_type, quantity in order_plan.items():
            if product_type not in self.product_routes:
                print(f"⚠️ Product {product_type} has no process route, skipping")
                continue

            # Create individual job for each product quantity
            for i in range(quantity):
                job_id = f"{product_type}_{i + 1:03d}"

                # Set due date (simplified: all orders due in 14 days)
                order_date = datetime.now()
                due_date = order_date + timedelta(days=14)

                job = Job(
                    job_id=job_id,
                    product_type=product_type,
                    quantity=1,
                    order_date=order_date,
                    due_date=due_date,
                    process_sequence=self.product_routes[product_type].copy()
                )

                jobs.append(job)
                self.jobs[job_id] = job
                job_counter += 1

        print(f"✅ Created {job_counter} production jobs")
        return jobs

    def schedule_job_arrival(self, jobs: List[Job], arrival_pattern: str = "batch"):
        """Schedule job arrival events"""
        print(f"📅 Scheduling job arrivals: {arrival_pattern} pattern")

        if arrival_pattern == "batch":
            # Batch arrival: all jobs arrive at time 0
            for job in jobs:
                event = SimulationEvent(
                    time=0.0,
                    event_type=EventType.JOB_ARRIVAL,
                    job_id=job.job_id
                )
                heapq.heappush(self.event_queue, event)

        elif arrival_pattern == "uniform":
            # Uniform arrival: distributed over first 8 hours
            for i, job in enumerate(jobs):
                arrival_time = (i / len(jobs)) * 480  # 8 hours = 480 minutes
                event = SimulationEvent(
                    time=arrival_time,
                    event_type=EventType.JOB_ARRIVAL,
                    job_id=job.job_id
                )
                heapq.heappush(self.event_queue, event)

        elif arrival_pattern == "poisson":
            # Poisson arrival: random intervals
            current_time = 0.0
            for job in jobs:
                # Average interval 10 minutes
                interval = np.random.exponential(10.0)
                current_time += interval

                event = SimulationEvent(
                    time=current_time,
                    event_type=EventType.JOB_ARRIVAL,
                    job_id=job.job_id
                )
                heapq.heappush(self.event_queue, event)

    def find_best_unit_for_process(self, job: Job) -> Optional[int]:
        """Find best production unit for current process"""
        if job.current_process_idx >= len(job.process_sequence):
            return None

        current_process = job.process_sequence[job.current_process_idx]

        # Find all units capable of executing this process
        capable_units = []
        for unit_id, unit in self.units.items():
            if current_process in unit.available_processes:
                capable_units.append(unit_id)

        if not capable_units:
            return None

        # Selection strategy: choose unit with shortest queue
        best_unit = min(capable_units,
                        key=lambda u: len(self.units[u].queue) + (1 if self.units[u].is_busy else 0))

        return best_unit

    def calculate_setup_time(self, unit: ProductionUnit, new_process: str) -> float:
        """Calculate setup time"""
        if unit.last_process_type is None or unit.last_process_type == new_process:
            return 0.0

        # Setup time between different processes
        base_setup = self.process_info.get(new_process, ProcessInfo("", "", 0)).setup_time
        return base_setup * self.config['setup_time_factor']

    def check_constraints(self, job: Job, unit_id: int) -> bool:
        """Check if constraints are satisfied"""
        unit = self.units[unit_id]

        # Constraint 1: Unit can only process one job at a time
        if unit.is_busy:
            return False

        # Constraint 2: Job cannot be processed multiple times simultaneously
        if job.is_processing:
            return False

        # Constraint 3: Process sequence must be respected
        if job.current_process_idx > 0:
            # Check if previous process is completed
            if self.current_time < job.last_process_end_time:
                return False

        return True

    def process_event(self, event: SimulationEvent):
        """Process simulation event"""
        self.current_time = event.time

        if event.event_type == EventType.JOB_ARRIVAL:
            self._handle_job_arrival(event)

        elif event.event_type == EventType.PROCESS_START:
            self._handle_process_start(event)

        elif event.event_type == EventType.PROCESS_COMPLETE:
            self._handle_process_complete(event)

        elif event.event_type == EventType.TRANSPORT_START:
            self._handle_transport_start(event)

        elif event.event_type == EventType.TRANSPORT_COMPLETE:
            self._handle_transport_complete(event)

        elif event.event_type == EventType.ENTER_QUEUE:
            self._handle_enter_queue(event)

        elif event.event_type == EventType.EXIT_QUEUE:
            self._handle_exit_queue(event)

    def _handle_job_arrival(self, event: SimulationEvent):
        """Handle job arrival event"""
        job = self.jobs[event.job_id]
        job.start_time = self.current_time
        job.status = JobStatus.WAITING

        # Find suitable production unit
        best_unit_id = self.find_best_unit_for_process(job)

        if best_unit_id is not None:
            unit = self.units[best_unit_id]

            if self.check_constraints(job, best_unit_id):
                # Unit is idle and constraints satisfied, start processing
                self._start_processing(job, best_unit_id)
            else:
                # Unit is busy or constraints not met, add to queue
                self._add_to_queue(job, best_unit_id)

        # Record arrival data
        self._record_simulation_data("job_arrival", {
            "job_id": job.job_id,
            "product_type": job.product_type,
            "time": self.current_time
        })

    def _handle_process_start(self, event: SimulationEvent):
        """Handle process start event with constraint checking"""
        job = self.jobs[event.job_id]
        unit = self.units[event.unit_id]

        # Double-check constraints
        if not self.check_constraints(job, event.unit_id):
            self.performance_metrics["constraint_violations"] += 1
            print(f"⚠️ Constraint violation detected at time {self.current_time}")
            return

        # Update states
        job.status = JobStatus.PROCESSING
        job.current_unit = event.unit_id
        job.is_processing = True
        job.process_start_times.append(self.current_time)

        unit.current_job = job
        unit.is_busy = True
        unit.busy_start_time = self.current_time

        # Calculate processing time
        process_id = job.process_sequence[job.current_process_idx]
        process_duration = self.process_info.get(process_id, ProcessInfo("", "", 30)).duration

        # Add randomness (±10%)
        actual_duration = process_duration * (0.9 + 0.2 * random.random())

        # Record Gantt data
        self._record_gantt_data(
            'processing', job.job_id, event.unit_id,
            self.current_time, self.current_time + actual_duration, process_id
        )

        # Schedule process completion
        completion_event = SimulationEvent(
            time=self.current_time + actual_duration,
            event_type=EventType.PROCESS_COMPLETE,
            job_id=job.job_id,
            unit_id=event.unit_id,
            process_id=process_id
        )
        heapq.heappush(self.event_queue, completion_event)

        # Update statistics
        unit.total_busy_time += actual_duration

        self._record_simulation_data("process_start", {
            "job_id": job.job_id,
            "unit_id": event.unit_id,
            "process_id": process_id,
            "duration": actual_duration
        })

    def _handle_process_complete(self, event: SimulationEvent):
        """Handle process completion event"""
        job = self.jobs[event.job_id]
        unit = self.units[event.unit_id]

        # Update job state
        job.process_end_times.append(self.current_time)
        job.processing_times.append(self.current_time - job.process_start_times[-1])
        job.completed_processes.append(event.process_id)
        job.current_process_idx += 1
        job.is_processing = False
        job.last_process_end_time = self.current_time

        # Release unit
        unit.current_job = None
        unit.is_busy = False
        unit.total_idle_time += self.current_time - unit.busy_start_time
        unit.last_process_type = event.process_id
        job.current_unit = None

        # Check if all processes completed
        if job.current_process_idx >= len(job.process_sequence):
            # Job completed
            job.status = JobStatus.COMPLETED
            job.completion_time = self.current_time
            self.performance_metrics["completed_jobs"] += 1

            self._record_simulation_data("job_complete", {
                "job_id": job.job_id,
                "completion_time": self.current_time,
                "total_cycle_time": self.current_time - (job.start_time or 0)
            })

        else:
            # Need to transfer to next process
            next_unit_id = self.find_best_unit_for_process(job)

            if next_unit_id is not None:
                if next_unit_id == event.unit_id:
                    # Continue processing in same unit
                    if self.check_constraints(job, next_unit_id):
                        self._start_processing(job, next_unit_id)
                    else:
                        self._add_to_queue(job, next_unit_id)
                else:
                    # Need transport
                    self._start_transport(job, event.unit_id, next_unit_id)
            else:
                # No available unit, job blocked
                job.status = JobStatus.BLOCKED

        # Process unit queue
        self._process_unit_queue(event.unit_id)

    def _handle_transport_start(self, event: SimulationEvent):
        """Handle transport start event"""
        job = self.jobs[event.job_id]
        job.status = JobStatus.TRANSPORTING

        from_unit = event.additional_data['from_unit']
        to_unit = event.additional_data['to_unit']
        transport_time = self.transport_matrix.get((from_unit, to_unit), 5.0)

        # Record Gantt data - transport
        self._record_gantt_data(
            'transport', job.job_id, from_unit,
            self.current_time, self.current_time + transport_time,
            f"Transport_{from_unit}_to_{to_unit}"
        )

        # Schedule transport completion
        transport_complete_event = SimulationEvent(
            time=self.current_time + transport_time,
            event_type=EventType.TRANSPORT_COMPLETE,
            job_id=job.job_id,
            unit_id=to_unit,
            additional_data={'transport_time': transport_time}
        )
        heapq.heappush(self.event_queue, transport_complete_event)

        job.transport_times.append(transport_time)

    def _handle_transport_complete(self, event: SimulationEvent):
        """Handle transport completion event"""
        job = self.jobs[event.job_id]
        target_unit = self.units[event.unit_id]

        if self.check_constraints(job, event.unit_id):
            # Target unit is idle, start processing
            self._start_processing(job, event.unit_id)
        else:
            # Target unit is busy, add to queue
            self._add_to_queue(job, event.unit_id)

    def _handle_enter_queue(self, event: SimulationEvent):
        """Handle queue entry event"""
        job = self.jobs[event.job_id]
        job.status = JobStatus.IN_QUEUE
        job.queue_enter_times.append(self.current_time)

        # Record waiting start
        self._record_gantt_data(
            'waiting', job.job_id, event.unit_id,
            self.current_time, self.current_time,  # End time will be updated
            f"Queue_Unit_{event.unit_id}"
        )

    def _handle_exit_queue(self, event: SimulationEvent):
        """Handle queue exit event"""
        job = self.jobs[event.job_id]
        job.queue_exit_times.append(self.current_time)

        # Calculate wait time
        if job.queue_enter_times:
            wait_time = self.current_time - job.queue_enter_times[-1]
            job.wait_times.append(wait_time)

            # Update Gantt data for waiting period
            for i in range(len(self.gantt_data) - 1, -1, -1):
                if (self.gantt_data[i]['job_id'] == event.job_id and
                        self.gantt_data[i]['event_type'] == 'waiting' and
                        self.gantt_data[i]['end_time'] == self.gantt_data[i]['start_time']):
                    self.gantt_data[i]['end_time'] = self.current_time
                    self.gantt_data[i]['duration'] = wait_time
                    break

    def _add_to_queue(self, job: Job, unit_id: int):
        """Add job to unit queue"""
        unit = self.units[unit_id]
        unit.queue.append(job)

        # Generate enter queue event
        enter_queue_event = SimulationEvent(
            time=self.current_time,
            event_type=EventType.ENTER_QUEUE,
            job_id=job.job_id,
            unit_id=unit_id
        )
        heapq.heappush(self.event_queue, enter_queue_event)

    def _start_processing(self, job: Job, unit_id: int):
        """Start processing with setup time"""
        unit = self.units[unit_id]
        process_id = job.process_sequence[job.current_process_idx]

        # Calculate setup time
        setup_time = self.calculate_setup_time(unit, process_id)

        # Record setup time if any
        if setup_time > 0:
            self._record_gantt_data(
                'setup', job.job_id, unit_id,
                self.current_time, self.current_time + setup_time,
                f"Setup_{process_id}"
            )

        # Schedule process start
        start_event = SimulationEvent(
            time=self.current_time + setup_time,
            event_type=EventType.PROCESS_START,
            job_id=job.job_id,
            unit_id=unit_id,
            process_id=process_id
        )
        heapq.heappush(self.event_queue, start_event)

        if setup_time > 0:
            unit.setup_time_remaining = setup_time

    def _start_transport(self, job: Job, from_unit: int, to_unit: int):
        """Start transport"""
        transport_event = SimulationEvent(
            time=self.current_time,
            event_type=EventType.TRANSPORT_START,
            job_id=job.job_id,
            additional_data={'from_unit': from_unit, 'to_unit': to_unit}
        )
        heapq.heappush(self.event_queue, transport_event)

    def _process_unit_queue(self, unit_id: int):
        """Process unit queue for next job"""
        unit = self.units[unit_id]

        while unit.queue and not unit.is_busy:
            next_job = unit.queue.popleft()

            # Generate exit queue event
            exit_queue_event = SimulationEvent(
                time=self.current_time,
                event_type=EventType.EXIT_QUEUE,
                job_id=next_job.job_id,
                unit_id=unit_id
            )
            heapq.heappush(self.event_queue, exit_queue_event)

            if self.check_constraints(next_job, unit_id):
                self._start_processing(next_job, unit_id)
                break
            else:
                # Put back in queue if constraints not met
                unit.queue.appendleft(next_job)
                break

    def _record_simulation_data(self, event_type: str, data: dict):
        """Record simulation data for visualization"""
        record = {
            "time": self.current_time,
            "event_type": event_type,
            **data
        }
        self.visualization_data["time_series"].append(record)

    def _record_gantt_data(self, event_type: str, job_id: str, unit_id: int,
                           start_time: float, end_time: float, process_id: str = None):
        """Record Gantt chart data"""
        job = self.jobs[job_id]

        self.gantt_data.append({
            'job_id': job_id,
            'product_type': job.product_type,
            'unit_id': unit_id,
            'process_id': process_id,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'event_type': event_type
        })

    def run_simulation(self, duration: float = 1440.0):  # Default 24 hours
        """Run simulation"""
        print(f"🚀 Starting simulation, duration: {duration} minutes")

        start_time = time.time()
        events_processed = 0

        while self.event_queue and self.current_time < duration:
            event = heapq.heappop(self.event_queue)

            if event.time > duration:
                break

            self.process_event(event)
            events_processed += 1

            # Periodic statistics update
            if events_processed % 100 == 0:
                self._update_performance_metrics()

        # Final statistics
        self._finalize_simulation()

        execution_time = time.time() - start_time
        print(f"✅ Simulation completed!")
        print(f"   • Simulation time: {self.current_time:.1f} minutes")
        print(f"   • Events processed: {events_processed}")
        print(f"   • Execution time: {execution_time:.2f} seconds")
        print(f"   • Completed jobs: {self.performance_metrics['completed_jobs']}")
        print(f"   • Constraint violations: {self.performance_metrics['constraint_violations']}")

    def _update_performance_metrics(self):
        """Update performance metrics"""
        # Calculate unit utilizations
        for unit_id, unit in self.units.items():
            if self.current_time > 0:
                utilization = unit.total_busy_time / self.current_time
                self.performance_metrics["unit_utilizations"][unit_id] = utilization
                unit.utilization = utilization

        # Record WIP level
        wip_level = len([job for job in self.jobs.values()
                         if job.status in [JobStatus.WAITING, JobStatus.PROCESSING,
                                           JobStatus.TRANSPORTING, JobStatus.IN_QUEUE]])
        self.performance_metrics["wip_levels"].append((self.current_time, wip_level))

        # Record queue lengths
        total_queue_length = sum(len(unit.queue) for unit in self.units.values())
        self.performance_metrics["queue_lengths"].append((self.current_time, total_queue_length))

    def _finalize_simulation(self):
        """Finalize simulation statistics"""
        completed_jobs = [job for job in self.jobs.values() if job.status == JobStatus.COMPLETED]

        if completed_jobs:
            # Calculate average cycle time
            cycle_times = [(job.completion_time - job.start_time)
                           for job in completed_jobs if job.start_time and job.completion_time]
            if cycle_times:
                self.performance_metrics["average_cycle_time"] = np.mean(cycle_times)

            # Calculate average wait time
            all_wait_times = []
            for job in completed_jobs:
                all_wait_times.extend(job.wait_times)
            if all_wait_times:
                self.performance_metrics["average_wait_time"] = np.mean(all_wait_times)

            # Calculate average queue time
            queue_times = []
            for job in completed_jobs:
                if len(job.queue_enter_times) == len(job.queue_exit_times):
                    for enter, exit in zip(job.queue_enter_times, job.queue_exit_times):
                        queue_times.append(exit - enter)
            if queue_times:
                self.performance_metrics["average_queue_time"] = np.mean(queue_times)

            # Total throughput
            self.performance_metrics["total_throughput"] = len(completed_jobs)

    def generate_gantt_chart(self):
        """Generate enhanced Gantt chart with both HTML and image formats"""
        print("📊 Generating Gantt charts...")

        if not self.gantt_data:
            print("⚠️ No Gantt data available")
            return

        # Generate HTML version
        self._generate_html_gantt()

        # Generate image version
        self._generate_image_gantt()

        # Generate statistics
        self._generate_gantt_statistics()

    def _generate_html_gantt(self):
        """Generate interactive HTML Gantt chart"""
        # Create figure
        fig = go.Figure()

        # Color mapping
        product_types = list(set([item['product_type'] for item in self.gantt_data]))
        colors = px.colors.qualitative.Set3[:len(product_types)]
        color_map = {pt: colors[i % len(colors)] for i, pt in enumerate(product_types)}

        # Event type styling
        event_styles = {
            'processing': {'alpha': 1.0, 'pattern': 'solid'},
            'setup': {'alpha': 0.6, 'pattern': 'diagonal'},
            'transport': {'alpha': 0.4, 'pattern': 'horizontal'},
            'waiting': {'alpha': 0.3, 'pattern': 'vertical'}
        }

        # Get all unit IDs and sort
        unit_ids = sorted(list(set([item['unit_id'] for item in self.gantt_data])))

        # Create Gantt bars
        for item in self.gantt_data:
            y_pos = unit_ids.index(item['unit_id'])

            # Get color and style
            base_color = color_map[item['product_type']]
            style = event_styles.get(item['event_type'], {'alpha': 0.8, 'pattern': 'solid'})

            # Create bar
            fig.add_trace(go.Scatter(
                x=[item['start_time'], item['end_time'], item['end_time'],
                   item['start_time'], item['start_time']],
                y=[y_pos - 0.4, y_pos - 0.4, y_pos + 0.4, y_pos + 0.4, y_pos - 0.4],
                fill='toself',
                fillcolor=f'rgba{base_color[3:-1]}, {style["alpha"]})',
                line=dict(color='black', width=1),
                mode='lines',
                name=f"{item['product_type']} - {item['event_type']}",
                showlegend=False,
                hovertemplate=(
                    f"<b>Job:</b> {item['job_id']}<br>"
                    f"<b>Product:</b> {item['product_type']}<br>"
                    f"<b>Unit:</b> Unit {item['unit_id']}<br>"
                    f"<b>Type:</b> {item['event_type']}<br>"
                    f"<b>Process:</b> {item['process_id']}<br>"
                    f"<b>Start:</b> {item['start_time']:.1f} min<br>"
                    f"<b>End:</b> {item['end_time']:.1f} min<br>"
                    f"<b>Duration:</b> {item['duration']:.1f} min"
                    "<extra></extra>"
                )
            ))

            # Add text labels for processing events
            if item['event_type'] == 'processing' and item['duration'] > 5:
                fig.add_annotation(
                    x=(item['start_time'] + item['end_time']) / 2,
                    y=y_pos,
                    text=f"{item['job_id']}<br>P:{item['process_id']}",
                    showarrow=False,
                    font=dict(size=8, color='white'),
                    bgcolor='rgba(0,0,0,0.7)',
                    bordercolor='white',
                    borderwidth=1
                )

        # Add legends
        # Product type legend
        for product_type, color in color_map.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=15, color=color),
                name=product_type,
                showlegend=True
            ))

        # Event type legend
        event_types = [
            ('Processing', 'processing', 1.0, '■'),
            ('Setup', 'setup', 0.6, '◪'),
            ('Transport', 'transport', 0.4, '▬'),
            ('Waiting', 'waiting', 0.3, '│')
        ]

        for event_name, event_type, alpha, symbol in event_types:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers+text',
                marker=dict(size=10, color=f'rgba(128,128,128,{alpha})'),
                text=[symbol],
                textposition="middle center",
                name=event_name,
                showlegend=True
            ))

        # Layout settings
        fig.update_layout(
            title=dict(
                text="Matrix Production Gantt Chart - Unit Processing Timeline",
                font=dict(size=20)
            ),
            xaxis=dict(
                title="Time (minutes)",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                range=[0, max([item['end_time'] for item in self.gantt_data]) * 1.05]
            ),
            yaxis=dict(
                title="Production Units",
                tickmode='array',
                tickvals=list(range(len(unit_ids))),
                ticktext=[f"Unit {uid}" for uid in unit_ids],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            height=max(600, len(unit_ids) * 60),
            width=1400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=100, r=50, t=100, b=100)
        )

        # Save HTML gantt chart
        html_path = os.path.join(self.output_dir, "production_gantt_chart.html")
        pyo.plot(fig, filename=html_path, auto_open=False)
        print(f"📊 HTML Gantt chart saved to: {html_path}")

    def _generate_image_gantt(self):
        """Generate high-quality image Gantt chart using matplotlib"""
        print("🎨 Generating image Gantt chart...")

        # Prepare data
        df_gantt = pd.DataFrame(self.gantt_data)
        unit_ids = sorted(df_gantt['unit_id'].unique())
        product_types = sorted(df_gantt['product_type'].unique())

        # Create color maps
        colors = plt.cm.Set3(np.linspace(0, 1, len(product_types)))
        product_color_map = {pt: colors[i] for i, pt in enumerate(product_types)}

        # Event type patterns and alphas
        event_patterns = {
            'processing': {'hatch': None, 'alpha': 0.9, 'edgecolor': 'black'},
            'setup': {'hatch': '///', 'alpha': 0.7, 'edgecolor': 'gray'},
            'transport': {'hatch': '---', 'alpha': 0.5, 'edgecolor': 'blue'},
            'waiting': {'hatch': '...', 'alpha': 0.3, 'edgecolor': 'red'}
        }

        # Create figure
        fig, ax = plt.subplots(figsize=(20, max(8, len(unit_ids) * 0.6)))

        # Plot Gantt bars
        for _, row in df_gantt.iterrows():
            y_pos = unit_ids.index(row['unit_id'])
            start_time = row['start_time']
            duration = row['duration']
            product_type = row['product_type']
            event_type = row['event_type']

            # Get colors and patterns
            base_color = product_color_map[product_type]
            pattern_info = event_patterns.get(event_type, {'hatch': None, 'alpha': 0.8, 'edgecolor': 'black'})

            # Draw bar
            bar = ax.barh(
                y_pos, duration, left=start_time, height=0.6,
                color=base_color, alpha=pattern_info['alpha'],
                hatch=pattern_info['hatch'],
                edgecolor=pattern_info['edgecolor'],
                linewidth=1.5
            )

            # Add text labels for processing events
            if event_type == 'processing' and duration > 10:
                # Extract job number and process ID for compact display
                job_short = row['job_id'].split('_')[-1] if '_' in row['job_id'] else row['job_id'][:4]
                process_short = str(row['process_id'])[-4:] if row['process_id'] else ""

                ax.text(
                    start_time + duration / 2, y_pos,
                    f"{job_short}\n{process_short}",
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white' if pattern_info['alpha'] > 0.7 else 'black',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.5)
                )

        # Customize axes
        ax.set_xlim(0, df_gantt['end_time'].max() * 1.05)
        ax.set_ylim(-0.5, len(unit_ids) - 0.5)

        # Y-axis (units)
        ax.set_yticks(range(len(unit_ids)))
        ax.set_yticklabels([f"Unit {uid}" for uid in unit_ids])
        ax.set_ylabel('Production Units', fontsize=14, fontweight='bold')

        # X-axis (time)
        ax.set_xlabel('Time (minutes)', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')

        # Title
        plt.title('Production Gantt Chart - Unit Processing Timeline',
                  fontsize=18, fontweight='bold', pad=20)

        # Create legends
        # Product type legend
        product_legend_elements = [
            patches.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.8, label=product)
            for product, color in product_color_map.items()
        ]

        # Event type legend
        event_legend_elements = []
        for event_type, pattern_info in event_patterns.items():
            element = patches.Rectangle(
                (0, 0), 1, 1,
                facecolor='gray', alpha=pattern_info['alpha'],
                hatch=pattern_info['hatch'],
                edgecolor=pattern_info['edgecolor'],
                label=event_type.capitalize()
            )
            event_legend_elements.append(element)

        # Add legends
        product_legend = ax.legend(
            handles=product_legend_elements,
            loc='upper left', bbox_to_anchor=(0, 1),
            title='Product Types', title_fontsize=12,
            fontsize=10, frameon=True, fancybox=True
        )

        event_legend = ax.legend(
            handles=event_legend_elements,
            loc='upper left', bbox_to_anchor=(0.2, 1),
            title='Event Types', title_fontsize=12,
            fontsize=10, frameon=True, fancybox=True
        )

        # Add product legend back (matplotlib removes previous legend)
        ax.add_artist(product_legend)

        # Adjust layout
        plt.tight_layout()

        # Save high-resolution image
        image_path = os.path.join(self.output_dir, "images", "production_gantt_chart.jpg")
        plt.savefig(image_path, dpi=1200, bbox_inches='tight', format='jpg', facecolor='white')
        plt.close()

        print(f"🎨 High-resolution Gantt chart image saved to: {image_path}")

    # def _generate_gantt_statistics(self):
    #     """Generate Gantt chart statistics"""
    #     if not self.gantt_data:
    #         return
    #
    #     df_gantt = pd.DataFrame(self.gantt_data)
    #
    #     # Calculate statistics by event type
    #     stats_by_event = df_gantt.groupby('event_type').agg({
    #         'duration': ['count', 'sum', 'mean', 'std'],
    #         'job_id': 'nunique'
    #     }).round(2)
    #
    #     # Calculate statistics by unit
    #     stats_by_unit = df_gantt.groupby('unit_id').agg({
    #         'duration': ['sum', 'mean'],
    #         'job_id': 'nunique'
    #     }).round(2)
    #
    #     # Calculate statistics by product type
    #     stats_by_product = df_gantt.groupby('product_type').agg({
    #         'duration': ['sum', 'mean'],
    #         'job_id': 'nunique'
    #     }).round(2)
    #
    #     # Save statistics
    #     stats = {
    #         'by_event_type': stats_by_event.to_dict(),
    #         'by_unit': stats_by_unit.to_dict(),
    #         'by_product_type': stats_by_product.to_dict(),
    #         'total_events': len(df_gantt),
    #         'total_duration': df_gantt['duration'].sum(),
    #         'simulation_end_time': df_gantt['end_time'].max()
    #     }
    #
    #     stats_path = os.path.join(self.output_dir, "data", "gantt_statistics.json")
    #     with open(stats_path, 'w', encoding='utf-8') as f:
    #         json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    #
    #     print(f"📊 Gantt statistics saved to: {stats_path}")

    def _generate_gantt_statistics(self):
        """Generate Gantt chart statistics"""
        if not self.gantt_data:
            return

        df_gantt = pd.DataFrame(self.gantt_data)

        # Calculate statistics by event type
        stats_by_event = df_gantt.groupby('event_type').agg({
            'duration': ['count', 'sum', 'mean', 'std'],
            'job_id': 'nunique'
        }).round(2)

        # Calculate statistics by unit
        stats_by_unit = df_gantt.groupby('unit_id').agg({
            'duration': ['sum', 'mean'],
            'job_id': 'nunique'
        }).round(2)

        # Calculate statistics by product type
        stats_by_product = df_gantt.groupby('product_type').agg({
            'duration': ['sum', 'mean'],
            'job_id': 'nunique'
        }).round(2)

        # 修复：将多级列索引转换为JSON可序列化的格式
        def convert_multiindex_dict(df):
            """Convert DataFrame with MultiIndex columns to JSON-serializable dict"""
            result = {}
            for col_tuple, series in df.items():
                # 将元组键转换为字符串键
                if isinstance(col_tuple, tuple):
                    key = '_'.join(map(str, col_tuple))
                else:
                    key = str(col_tuple)
                result[key] = series.to_dict()
            return result

        # Save statistics with converted keys
        stats = {
            'by_event_type': convert_multiindex_dict(stats_by_event),
            'by_unit': convert_multiindex_dict(stats_by_unit),
            'by_product_type': convert_multiindex_dict(stats_by_product),
            'total_events': len(df_gantt),
            'total_duration': float(df_gantt['duration'].sum()),  # 确保是普通的float
            'simulation_end_time': float(df_gantt['end_time'].max())  # 确保是普通的float
        }

        stats_path = os.path.join(self.output_dir, "data", "gantt_statistics.json")

        # 确保目录存在
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)

        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
            print(f"📊 Gantt statistics saved to: {stats_path}")
        except Exception as e:
            print(f"⚠️ Error saving Gantt statistics: {e}")
            # 保存简化版本的统计数据
            simplified_stats = {
                'total_events': len(df_gantt),
                'total_duration': float(df_gantt['duration'].sum()),
                'simulation_end_time': float(df_gantt['end_time'].max()),
                'event_types': df_gantt['event_type'].value_counts().to_dict(),
                'units_involved': df_gantt['unit_id'].nunique(),
                'products_involved': df_gantt['product_type'].nunique()
            }

            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(simplified_stats, f, ensure_ascii=False, indent=2)
            print(f"📊 Simplified Gantt statistics saved to: {stats_path}")

    def visualize_simulation_results(self):
        """Generate comprehensive simulation result visualizations"""
        print("📈 Generating simulation result visualizations...")

        # Generate Gantt charts
        self.generate_gantt_chart()

        # Generate performance metrics visualization
        self._generate_performance_visualization()

        # Generate utilization analysis
        self._generate_utilization_analysis()

        # Generate flow analysis
        self._generate_flow_analysis()

        # Generate final report
        self._generate_simulation_report()

    def _generate_performance_visualization(self):
        """Generate performance metrics visualization"""
        print("📊 Generating performance metrics visualization...")

        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Unit Utilization
        units = list(self.performance_metrics["unit_utilizations"].keys())
        utilizations = list(self.performance_metrics["unit_utilizations"].values())

        bars = ax1.bar(range(len(units)), utilizations, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Production Units')
        ax1.set_ylabel('Utilization Rate')
        ax1.set_title('Unit Utilization Analysis', fontweight='bold')
        ax1.set_xticks(range(len(units)))
        ax1.set_xticklabels([f'U{u}' for u in units], rotation=45)
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, util in zip(bars, utilizations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{util:.1%}', ha='center', va='bottom', fontsize=9)

        # 2. WIP Levels Over Time
        if self.performance_metrics["wip_levels"]:
            wip_times, wip_levels = zip(*self.performance_metrics["wip_levels"])
            ax2.plot(wip_times, wip_levels, 'g-', linewidth=2, alpha=0.8)
            ax2.fill_between(wip_times, wip_levels, alpha=0.3, color='green')
            ax2.set_xlabel('Time (minutes)')
            ax2.set_ylabel('Work in Progress Level')
            ax2.set_title('WIP Levels Over Time', fontweight='bold')
            ax2.grid(True, alpha=0.3)

        # 3. Queue Lengths Over Time
        if self.performance_metrics["queue_lengths"]:
            queue_times, queue_lengths = zip(*self.performance_metrics["queue_lengths"])
            ax3.plot(queue_times, queue_lengths, 'r-', linewidth=2, alpha=0.8)
            ax3.fill_between(queue_times, queue_lengths, alpha=0.3, color='red')
            ax3.set_xlabel('Time (minutes)')
            ax3.set_ylabel('Total Queue Length')
            ax3.set_title('Queue Lengths Over Time', fontweight='bold')
            ax3.grid(True, alpha=0.3)

        # 4. Key Performance Indicators
        kpis = {
            'Completed Jobs': self.performance_metrics["completed_jobs"],
            'Avg Cycle Time': f"{self.performance_metrics['average_cycle_time']:.1f} min",
            'Avg Wait Time': f"{self.performance_metrics['average_wait_time']:.1f} min",
            'Constraint Violations': self.performance_metrics["constraint_violations"]
        }

        # Create text-based KPI display
        ax4.axis('off')
        ax4.text(0.5, 0.9, 'Key Performance Indicators',
                 ha='center', va='top', fontsize=16, fontweight='bold',
                 transform=ax4.transAxes)

        y_pos = 0.7
        for kpi, value in kpis.items():
            ax4.text(0.1, y_pos, f'{kpi}:',
                     ha='left', va='center', fontsize=12, fontweight='bold',
                     transform=ax4.transAxes)
            ax4.text(0.9, y_pos, str(value),
                     ha='right', va='center', fontsize=12,
                     transform=ax4.transAxes)
            y_pos -= 0.15

        # Add performance score
        avg_utilization = np.mean(list(self.performance_metrics["unit_utilizations"].values()))
        performance_score = avg_utilization * 100 * (1 - self.performance_metrics["constraint_violations"] / max(1,
                                                                                                                 self.performance_metrics[
                                                                                                                     "completed_jobs"]))

        ax4.text(0.5, 0.1, f'Performance Score: {performance_score:.1f}%',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
                 transform=ax4.transAxes)

        plt.tight_layout()

        # Save performance visualization
        perf_path = os.path.join(self.output_dir, "images", "performance_metrics.jpg")
        plt.savefig(perf_path, dpi=1200, bbox_inches='tight', format='jpg')
        plt.close()

        print(f"📊 Performance metrics visualization saved to: {perf_path}")

    def _generate_utilization_analysis(self):
        """Generate detailed utilization analysis"""
        print("⚡ Generating utilization analysis...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Detailed unit utilization with busy/idle breakdown
        units = list(self.units.keys())
        busy_times = [self.units[u].total_busy_time for u in units]
        idle_times = [self.units[u].total_idle_time for u in units]
        total_times = [busy + idle for busy, idle in zip(busy_times, idle_times)]

        # Stacked bar chart
        x_pos = np.arange(len(units))
        ax1.bar(x_pos, busy_times, label='Busy Time', color='orange', alpha=0.8)
        ax1.bar(x_pos, idle_times, bottom=busy_times, label='Idle Time', color='lightgray', alpha=0.8)

        ax1.set_xlabel('Production Units')
        ax1.set_ylabel('Time (minutes)')
        ax1.set_title('Unit Time Utilization Breakdown', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'Unit {u}' for u in units], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Utilization distribution
        utilization_values = list(self.performance_metrics["unit_utilizations"].values())
        ax2.hist(utilization_values, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(utilization_values), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(utilization_values):.1%}')
        ax2.set_xlabel('Utilization Rate')
        ax2.set_ylabel('Number of Units')
        ax2.set_title('Utilization Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save utilization analysis
        util_path = os.path.join(self.output_dir, "images", "utilization_analysis.jpg")
        plt.savefig(util_path, dpi=1200, bbox_inches='tight', format='jpg')
        plt.close()

        print(f"⚡ Utilization analysis saved to: {util_path}")

    def _generate_flow_analysis(self):
        """Generate job flow analysis"""
        print("🔄 Generating job flow analysis...")

        if not self.gantt_data:
            print("⚠️ No job flow data available")
            return

        df_gantt = pd.DataFrame(self.gantt_data)

        # Create flow visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Processing time by product type
        processing_data = df_gantt[df_gantt['event_type'] == 'processing']
        if not processing_data.empty:
            product_processing = processing_data.groupby('product_type')['duration'].agg(['mean', 'std'])

            x_pos = np.arange(len(product_processing))
            ax1.bar(x_pos, product_processing['mean'],
                    yerr=product_processing['std'], capsize=5,
                    color='skyblue', alpha=0.8, edgecolor='black')
            ax1.set_xlabel('Product Type')
            ax1.set_ylabel('Average Processing Time (min)')
            ax1.set_title('Average Processing Time by Product Type', fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(product_processing.index, rotation=45)
            ax1.grid(True, alpha=0.3)

        # 2. Event type distribution
        event_counts = df_gantt['event_type'].value_counts()
        colors = ['gold', 'lightcoral', 'lightgreen', 'lightblue']
        ax2.pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax2.set_title('Event Type Distribution', fontweight='bold')

        # 3. Processing time distribution
        if not processing_data.empty:
            ax3.hist(processing_data['duration'], bins=15, color='green', alpha=0.7, edgecolor='black')
            ax3.axvline(processing_data['duration'].mean(), color='red', linestyle='--',
                        linewidth=2, label=f'Mean: {processing_data["duration"].mean():.1f} min')
            ax3.set_xlabel('Processing Time (minutes)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Processing Time Distribution', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Unit workload comparison
        unit_workload = df_gantt.groupby('unit_id')['duration'].sum()
        ax4.bar(range(len(unit_workload)), unit_workload.values, color='purple', alpha=0.7)
        ax4.set_xlabel('Production Units')
        ax4.set_ylabel('Total Workload (minutes)')
        ax4.set_title('Total Workload by Unit', fontweight='bold')
        ax4.set_xticks(range(len(unit_workload)))
        ax4.set_xticklabels([f'U{u}' for u in unit_workload.index], rotation=45)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save flow analysis
        flow_path = os.path.join(self.output_dir, "images", "job_flow_analysis.jpg")
        plt.savefig(flow_path, dpi=1200, bbox_inches='tight', format='jpg')
        plt.close()

        print(f"🔄 Job flow analysis saved to: {flow_path}")

    def _generate_simulation_report(self):
        """Generate comprehensive simulation report"""
        print("📋 Generating comprehensive simulation report...")

        # Calculate additional metrics
        completed_jobs = [job for job in self.jobs.values() if job.status == JobStatus.COMPLETED]

        report_content = f"""
        ==========================================
        PRODUCTION SIMULATION ANALYSIS REPORT
        ==========================================

        📅 Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        ⏱️ Simulation Duration: {self.current_time:.1f} minutes ({self.current_time / 60:.1f} hours)

        ═══════════════════════════════════════════════════════════════════════════════
        📊 PRODUCTION PERFORMANCE SUMMARY
        ═══════════════════════════════════════════════════════════════════════════════

        🎯 Overall Performance:
        • Total Jobs Processed: {len(self.jobs)}
        • Completed Jobs: {self.performance_metrics['completed_jobs']}
        • Completion Rate: {(self.performance_metrics['completed_jobs'] / len(self.jobs) * 100):.1f}%
        • Average Cycle Time: {self.performance_metrics['average_cycle_time']:.2f} minutes
        • Average Wait Time: {self.performance_metrics['average_wait_time']:.2f} minutes
        • Average Queue Time: {self.performance_metrics['average_queue_time']:.2f} minutes
        • Constraint Violations: {self.performance_metrics['constraint_violations']}

        ⚡ Resource Utilization:
        • Average Unit Utilization: {(sum(self.performance_metrics['unit_utilizations'].values()) / len(self.performance_metrics['unit_utilizations']) * 100):.1f}%
        • Highest Utilization: {(max(self.performance_metrics['unit_utilizations'].values()) * 100):.1f}%
        • Lowest Utilization: {(min(self.performance_metrics['unit_utilizations'].values()) * 100):.1f}%

        🏭 Production Units Analysis:
        """

        # Add unit-specific analysis
        for unit_id, utilization in sorted(self.performance_metrics['unit_utilizations'].items()):
            unit = self.units[unit_id]
            report_content += f"""
        • Unit {unit_id}: {utilization:.1%} utilization, {len(unit.available_processes)} processes"""

        # Add product type analysis
        if completed_jobs:
            product_completion = defaultdict(int)
            for job in completed_jobs:
                product_completion[job.product_type] += 1

            report_content += f"""

        📦 Product Type Analysis:"""
            for product, count in sorted(product_completion.items()):
                report_content += f"""
        • {product}: {count} units completed"""

        # Add timing analysis
        if completed_jobs:
            cycle_times = [(job.completion_time - job.start_time) for job in completed_jobs if
                           job.start_time and job.completion_time]
            if cycle_times:
                report_content += f"""

        ⏱️ Timing Analysis:
        • Fastest Completion: {min(cycle_times):.1f} minutes
        • Slowest Completion: {max(cycle_times):.1f} minutes
        • Cycle Time Std Dev: {np.std(cycle_times):.1f} minutes"""

        report_content += f"""

        ═══════════════════════════════════════════════════════════════════════════════
        🎯 SYSTEM EFFICIENCY ANALYSIS
        ═══════════════════════════════════════════════════════════════════════════════

        📈 Throughput Metrics:
        • Jobs per Hour: {(self.performance_metrics['completed_jobs'] / (self.current_time / 60)):.2f}
        • Effective Production Rate: {(self.performance_metrics['completed_jobs'] / len(self.jobs) * 100):.1f}%

        ⚖️ System Balance:
        • Unit Utilization Variance: {np.var(list(self.performance_metrics['unit_utilizations'].values())):.4f}
        • Load Balance Score: {(1 - np.std(list(self.performance_metrics['unit_utilizations'].values())) / np.mean(list(self.performance_metrics['unit_utilizations'].values()))):.3f}

        🚧 Bottleneck Analysis:"""

        # Identify bottlenecks
        utilizations = self.performance_metrics['unit_utilizations']
        avg_util = np.mean(list(utilizations.values()))
        bottlenecks = [unit_id for unit_id, util in utilizations.items() if util > avg_util * 1.2]
        underutilized = [unit_id for unit_id, util in utilizations.items() if util < avg_util * 0.8]

        if bottlenecks:
            report_content += f"""
        • Bottleneck Units: {bottlenecks} (high utilization)"""
        if underutilized:
            report_content += f"""
        • Underutilized Units: {underutilized} (low utilization)"""

        report_content += f"""

        ═══════════════════════════════════════════════════════════════════════════════
        💡 RECOMMENDATIONS AND INSIGHTS
        ═══════════════════════════════════════════════════════════════════════════════

        🔧 Optimization Opportunities:"""

        # Generate recommendations
        if self.performance_metrics['constraint_violations'] > 0:
            report_content += f"""
        • Address {self.performance_metrics['constraint_violations']} constraint violations"""

        if bottlenecks:
            report_content += f"""
        • Consider load balancing for units: {bottlenecks}"""

        if self.performance_metrics['average_wait_time'] > 30:
            report_content += f"""
        • Reduce waiting times through improved scheduling"""

        if len(underutilized) > 0:
            report_content += f"""
        • Optimize resource allocation for underutilized units"""

        report_content += f"""

        📁 Generated Files:
        • HTML Gantt Chart: production_gantt_chart.html
        • Image Gantt Chart: images/production_gantt_chart.jpg
        • Performance Metrics: images/performance_metrics.jpg
        • Utilization Analysis: images/utilization_analysis.jpg
        • Job Flow Analysis: images/job_flow_analysis.jpg
        • Gantt Statistics: data/gantt_statistics.json
        • Performance Data: data/performance_metrics.json

        ==========================================
        END OF SIMULATION REPORT
        ==========================================
        """

        # Save report
        report_path = os.path.join(self.output_dir, "reports", "simulation_performance_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # Save performance metrics data
        metrics_path = os.path.join(self.output_dir, "data", "performance_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.performance_metrics, f, ensure_ascii=False, indent=2, default=str)

        print(f"📋 Simulation report saved to: {report_path}")
        print(f"📊 Performance data saved to: {metrics_path}")


def main():
    """Main function for production simulation"""

    # Load layout data
    with open("Cell layout output/data/fixed_grid_layout_optimization_data.json", 'r') as f:
        layout_data = json.load(f)

    # Load clustering data
    with open("process_config_results/kpc_clusters.json", 'r') as f:
        clustering_result = json.load(f)
        clusters_data = clustering_result['clusters']

    # Production plan
    production_plan = {
        "K3473": 10,
        "K16842": 5,
        "K14386": 7,
        "K1786": 6,
        "K6286": 12,
        "K15022": 10,
        "K15126": 9
    }

    # Simulation configuration
    simulation_config = {
        "simulation_speed": 1.0,
        "transport_speed": 8.0,  # Faster transport
        "setup_time_factor": 0.15,
        "queue_capacity": 15,
        "random_seed": 42
    }

    # Initialize simulator
    simulator = MatrixProductionSimulator(
        neo4j_uri="bolt://localhost:7687",
        neo4j_auth=("neo4j", "dididaodao"),
        layout_data=layout_data,
        clusters_data=clusters_data,
        simulation_config=simulation_config,
        output_dir="Cell layout output"
    )

    # Initialize simulation environment
    simulator.initialize_simulation()

    # Create production plan
    jobs = simulator.create_production_plan(production_plan)

    # Schedule job arrivals
    simulator.schedule_job_arrival(jobs, arrival_pattern="uniform")

    # Run simulation (24 hours)
    simulator.run_simulation(duration=1440.0)

    # Visualize results
    simulator.visualize_simulation_results()

    return simulator


if __name__ == "__main__":
    main()