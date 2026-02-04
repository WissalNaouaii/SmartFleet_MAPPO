import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import random
import time
from collections import deque, defaultdict
import json
# ==================== CONFIGURATION ====================
@dataclass
class SimConfig:
    """Simulation configuration"""
    city_size: int = 1000  # meters
    n_depots: int = 2
    n_boutiques: int = 5
    n_restaurants: int = 5
    n_customer_zones: int = 10
    
    # Time
    sim_duration: int = 3600  # seconds
    timestep: float = 1.0  # seconds
    
    # Communication
    v2v_range: float = 300.0  # meters
    v2v_interval: float = 1.0  # seconds
    v5g_available_prob: float = 0.7  # 70% coverage
    v5g_query_cost: float = 0.5  # energy cost
    
    # Task spawning
    task_spawn_rate: float = 0.1  # tasks per second
    task_cancel_prob: float = 0.05  # 5% cancel rate
    
    # Environment dynamics
    road_block_prob: float = 0.02  # per timestep
    no_fly_zone_prob: float = 0.01

# ==================== ENUMS & DATA STRUCTURES ====================

class VehicleType(Enum):
    TRUCK = "truck"
    VAN = "van"
    DRONE = "drone"
    ROBOT = "robot"

class TaskStatus(Enum):
    PENDING = "pending"
    ANNOUNCED = "announced"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class MessageType(Enum):
    TASK_ANNOUNCEMENT = "task_announcement"
    BID = "bid"
    AWARD = "award"
    HANDOVER_REQUEST = "handover_request"
    HANDOVER_ACCEPT = "handover_accept"
    STATUS_UPDATE = "status_update"
    GOSSIP = "gossip"

@dataclass
class Position:
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

@dataclass
class Task:
    """Delivery task/parcel"""
    id: str
    pickup_loc: Position
    delivery_loc: Position
    created_time: float
    deadline: float  # absolute time
    priority: int  # 1=low, 3=normal, 5=urgent/medical
    weight: float  # kg
    volume: float  # cubic meters
    fragile: bool = False
    temperature_sensitive: bool = False
    allowed_vehicle_types: List[VehicleType] = field(default_factory=lambda: list(VehicleType))
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    pickup_time: Optional[float] = None
    delivery_time: Optional[float] = None

@dataclass
class VehicleSpec:
    """Vehicle specifications"""
    vehicle_type: VehicleType
    max_capacity_kg: float
    max_volume_m3: float
    max_battery_kwh: float
    max_speed_ms: float  # meters per second
    battery_consumption_rate: float  # kWh per meter
    comm_range: float
    can_access_no_fly: bool = False

# Vehicle specifications database
VEHICLE_SPECS = {
    VehicleType.TRUCK: VehicleSpec(
        vehicle_type=VehicleType.TRUCK,
        max_capacity_kg=500.0,
        max_volume_m3=20.0,
        max_battery_kwh=2000.0,
        max_speed_ms=15.0,
        battery_consumption_rate=0.02,
        comm_range=800.0,
        can_access_no_fly=False
    ),
    VehicleType.VAN: VehicleSpec(
        vehicle_type=VehicleType.VAN,
        max_capacity_kg=200.0,
        max_volume_m3=8.0,
        max_battery_kwh=150.0,
        max_speed_ms=18.0,
        battery_consumption_rate=0.03,
        comm_range=700.0,
        can_access_no_fly=False
    ),
    VehicleType.DRONE: VehicleSpec(
        vehicle_type=VehicleType.DRONE,
        max_capacity_kg=5.0,
        max_volume_m3=0.5,
        max_battery_kwh=200.0,
        max_speed_ms=25.0,
        battery_consumption_rate=0.05,
        comm_range=600.0,
        can_access_no_fly=True
    ),
    VehicleType.ROBOT: VehicleSpec(
        vehicle_type=VehicleType.ROBOT,
        max_capacity_kg=30.0,
        max_volume_m3=2.0,
        max_battery_kwh=100.0,
        max_speed_ms=8.0,
        battery_consumption_rate=0.02,
        comm_range=700.0,
        can_access_no_fly=False
    )
}

# ==================== RL AGENT ====================

class RLDeliveryAgent:
    """
    RL-based delivery agent with decentralized decision making
    
    State: local observations + neighbor info
    Action: discrete actions for task acceptance, movement, handovers
    Reward: multi-objective (on-time delivery, energy, fairness, cooperation)
    """
    
    def __init__(self, agent_id: str, vehicle_type: VehicleType, 
                 initial_position: Position, depot_position: Position):
        self.agent_id = agent_id
        self.vehicle_type = vehicle_type
        self.spec = VEHICLE_SPECS[vehicle_type]
        
        # Physical state
        self.position = initial_position
        self.depot_position = depot_position
        self.velocity = 0.0
        self.heading = 0.0  # radians
        
        # Resources
        self.battery = self.spec.max_battery_kwh
        self.current_load_kg = 0.0
        self.current_volume_m3 = 0.0
        
        # Task management
        self.assigned_tasks: List[Task] = []
        self.current_task: Optional[Task] = None
        self.task_history: List[Dict] = []
        
        # Communication
        self.neighbors: Dict[str, Dict] = {}
        self.known_tasks: Dict[str, Task] = {}
        self.last_comm_time = 0.0
        self.v5g_available = False
        
        # Performance metrics
        self.completed_count = 0
        self.on_time_count = 0
        self.failed_count = 0
        self.total_distance = 0.0
        self.total_energy = 0.0
        self.helped_neighbors = 0
        
        # RL state
        self.last_action = None
        self.episode_reward = 0.0
        
        # Navigation
        self.waypoints: List[Position] = []
        self.is_returning_to_depot = False
    
    # ==================== OBSERVATION SPACE ====================
    
    def get_observation(self, current_time: float, 
                       visible_tasks: List[Task],
                       nearby_agents: List['RLDeliveryAgent']) -> np.ndarray:
        """
        Construct observation vector for RL policy
        
        Observation structure (total 58 dims):
        - Self state (10): pos_x, pos_y, battery_norm, capacity_norm, volume_norm, 
                           speed, heading_sin, heading_cos, n_tasks, time_norm
        - Current task (6): rel_x, rel_y, deadline_remaining_norm, priority_norm, 
                            weight_norm, is_pickup
        - Top 5 visible tasks (30): [rel_x, rel_y, deadline_rem, priority, weight, distance]
        - Top 3 neighbors (12): [rel_x, rel_y, free_capacity, battery_level]
        """
        obs = np.zeros(58, dtype=np.float32)
        idx = 0
        
        # Self state (10)
        obs[idx:idx+2] = self.position.to_array() / 1000.0  # normalize to [0,1]
        obs[idx+2] = self.battery / self.spec.max_battery_kwh
        obs[idx+3] = 1.0 - (self.current_load_kg / self.spec.max_capacity_kg)
        obs[idx+4] = 1.0 - (self.current_volume_m3 / self.spec.max_volume_m3)
        obs[idx+5] = self.velocity / self.spec.max_speed_ms
        obs[idx+6] = np.sin(self.heading)
        obs[idx+7] = np.cos(self.heading)
        obs[idx+8] = min(len(self.assigned_tasks) / 5.0, 1.0)
        obs[idx+9] = (current_time % 3600) / 3600.0  # time of day
        idx += 10
        
        # Current task (6)
        if self.current_task:
            target = (self.current_task.pickup_loc if self.current_task.pickup_time is None 
                     else self.current_task.delivery_loc)
            rel_pos = (target.to_array() - self.position.to_array()) / 1000.0
            obs[idx:idx+2] = rel_pos
            obs[idx+2] = max(0, self.current_task.deadline - current_time) / 600.0
            obs[idx+3] = self.current_task.priority / 5.0
            obs[idx+4] = self.current_task.weight / 50.0
            obs[idx+5] = 1.0 if self.current_task.pickup_time is None else 0.0
        idx += 6
        
        # Top 5 visible tasks (30)
        sorted_tasks = sorted(visible_tasks, 
                             key=lambda t: self.position.distance_to(t.pickup_loc))[:5]
        for i in range(5):
            if i < len(sorted_tasks):
                t = sorted_tasks[i]
                rel_pos = (t.pickup_loc.to_array() - self.position.to_array()) / 1000.0
                obs[idx:idx+2] = rel_pos
                obs[idx+2] = max(0, t.deadline - current_time) / 600.0
                obs[idx+3] = t.priority / 5.0
                obs[idx+4] = t.weight / 50.0
                obs[idx+5] = min(self.position.distance_to(t.pickup_loc) / 500.0, 1.0)
            idx += 6
        
        # Top 3 neighbors (12)
        sorted_neighbors = sorted(nearby_agents, 
                                 key=lambda a: self.position.distance_to(a.position))[:3]
        for i in range(3):
            if i < len(sorted_neighbors):
                n = sorted_neighbors[i]
                rel_pos = (n.position.to_array() - self.position.to_array()) / 1000.0
                obs[idx:idx+2] = rel_pos
                obs[idx+2] = 1.0 - (n.current_load_kg / n.spec.max_capacity_kg)
                obs[idx+3] = n.battery / n.spec.max_battery_kwh
            idx += 4
        
        return obs
    
    # ==================== ACTION SPACE ====================
    
    @staticmethod
    def get_action_space():
        """
        Discrete action space (15 actions):
        0-4: ACCEPT_TASK[0-4] - accept one of top 5 visible tasks
        5-8: MOVE direction (NORTH, EAST, SOUTH, WEST)
        9: GO_TO_DEPOT
        10-12: REQUEST_HANDOVER to neighbor[0-2]
        13: QUERY_5G for routing
        14: DO_NOTHING / CONTINUE
        """
        return spaces.Discrete(15)
    
    def decode_action(self, action: int, visible_tasks: List[Task],
                     nearby_agents: List['RLDeliveryAgent']) -> Dict:
        """Decode discrete action into executable command"""
        
        if action < 5:
            # Accept task
            task_idx = action
            sorted_tasks = sorted(visible_tasks, 
                                 key=lambda t: self.position.distance_to(t.pickup_loc))
            if task_idx < len(sorted_tasks):
                return {'type': 'accept_task', 'task': sorted_tasks[task_idx]}
        
        elif action < 9:
            # Movement
            directions = {
                5: (0, 1),    # NORTH
                6: (1, 0),    # EAST
                7: (0, -1),   # SOUTH
                8: (-1, 0)    # WEST
            }
            dx, dy = directions[action]
            return {'type': 'move', 'direction': (dx, dy)}
        
        elif action == 9:
            return {'type': 'go_to_depot'}
        
        elif action < 13:
            # Request handover
            neighbor_idx = action - 10
            sorted_neighbors = sorted(nearby_agents, 
                                     key=lambda a: self.position.distance_to(a.position))
            if neighbor_idx < len(sorted_neighbors):
                return {'type': 'request_handover', 'neighbor': sorted_neighbors[neighbor_idx]}
        
        elif action == 13:
            return {'type': 'query_5g'}
        
        return {'type': 'continue'}
    
    # ==================== REWARD CALCULATION ====================
    
    def calculate_step_reward(self, action_result: Dict, current_time: float) -> float:
        """
        Multi-objective reward function:
        - On-time delivery: +10
        - Late delivery: +3
        - Missed deadline: -8
        - Energy efficiency: -0.01 * energy_used
        - Failure: -5
        - Cooperation (handover accepted): +1
        - 5G query cost: -0.5
        - Time penalty: -0.01 per step (encourage speed)
        """
        reward = -0.01  # base time penalty
        
        if action_result.get('delivered'):
            task = action_result['task']
            if current_time <= task.deadline:
                reward += 10.0  # on-time
                self.on_time_count += 1
            else:
                lateness = current_time - task.deadline
                if lateness < 300:  # within 5 min
                    reward += 3.0
                else:
                    reward -= 8.0
            self.completed_count += 1
        
        if action_result.get('pickup'):
            reward += 0.5  # small reward for pickup
        
        if action_result.get('task_cancelled'):
            reward -= 2.0
        
        if action_result.get('failed'):
            reward -= 5.0
            self.failed_count += 1
        
        if action_result.get('handover_accepted'):
            reward += 1.0
            self.helped_neighbors += 1
        
        if action_result.get('handover_helped'):
            reward += 2.0  # receiving help
        
        # Energy penalty
        energy_used = action_result.get('energy_used', 0)
        reward -= 0.01 * energy_used
        
        # 5G cost
        if action_result.get('used_5g'):
            reward -= 0.5
        
        # Battery critical warning
        if self.battery < self.spec.max_battery_kwh * 0.15:
            reward -= 0.5
        
        # Fairness bonus (if load is balanced)
        if action_result.get('fairness_bonus'):
            reward += 0.5
        
        self.episode_reward += reward
        return reward
    
    # ==================== EXECUTION ====================
    
    def execute_action(self, action: int, visible_tasks: List[Task],
                      nearby_agents: List['RLDeliveryAgent'], 
                      current_time: float, dt: float) -> Dict:
        """Execute decoded action and return result"""
        
        command = self.decode_action(action, visible_tasks, nearby_agents)
        result = {'energy_used': 0, 'distance_moved': 0}
        
        if command['type'] == 'accept_task':
            task = command['task']
            if self._can_accept_task(task, verbose=False):
                task.status = TaskStatus.ASSIGNED
                task.assigned_to = self.agent_id
                self.assigned_tasks.append(task)
                if self.current_task is None:
                    self.current_task = task
                result['accepted'] = True
        
        elif command['type'] == 'move':
            dx, dy = command['direction']
            distance = self.spec.max_speed_ms * dt
            new_x = self.position.x + dx * distance
            new_y = self.position.y + dy * distance
            # Keep in bounds
            new_x = max(0, min(1000, new_x))
            new_y = max(0, min(1000, new_y))
            
            moved_dist = np.sqrt((new_x - self.position.x)**2 + (new_y - self.position.y)**2)
            self.position.x = new_x
            self.position.y = new_y
            
            energy_used = moved_dist * self.spec.battery_consumption_rate
            self.battery = max(0, self.battery - energy_used)
            self.total_distance += moved_dist
            self.total_energy += energy_used
            
            result['energy_used'] = energy_used
            result['distance_moved'] = moved_dist
            
            # Check task completion
            if self.current_task:
                self._check_task_progress(current_time, result)
        
        elif command['type'] == 'go_to_depot':
            self.is_returning_to_depot = True
            target = self.depot_position
            dist = self.position.distance_to(target)
            if dist > 0:
                dx = (target.x - self.position.x) / dist
                dy = (target.y - self.position.y) / dist
                move_dist = min(self.spec.max_speed_ms * dt, dist)
                self.position.x += dx * move_dist
                self.position.y += dy * move_dist
                
                energy = move_dist * self.spec.battery_consumption_rate
                self.battery = max(0, self.battery - energy)
                self.total_distance += move_dist
                self.total_energy += energy
                result['energy_used'] = energy
                
                if self.position.distance_to(target) < 5.0:
                    # Reached depot - recharge
                    self.battery = self.spec.max_battery_kwh
                    self.is_returning_to_depot = False
                    result['recharged'] = True
        
        elif command['type'] == 'query_5g':
            if self.v5g_available:
                self.battery = max(0, self.battery - 0.5)
                result['used_5g'] = True
        
        elif command['type'] == 'request_handover':
            neighbor = command.get('neighbor')
            if neighbor and self.current_task:
                if neighbor._can_accept_task(self.current_task, verbose=False):
                    self.current_task.assigned_to = neighbor.agent_id
                    neighbor.assigned_tasks.append(self.current_task)
                    self.assigned_tasks.remove(self.current_task)
                    self.current_task = None
                    result['handover_accepted'] = True
        
        elif command['type'] == 'continue' and self.current_task:
            # Auto-navigate toward current task target
            if self.current_task.pickup_time is None:
                target = self.current_task.pickup_loc
            else:
                target = self.current_task.delivery_loc
            
            dist = self.position.distance_to(target)
            if dist > 0:
                dx = (target.x - self.position.x) / dist
                dy = (target.y - self.position.y) / dist
                
                move_dist = min(self.spec.max_speed_ms * dt, dist)
                self.position.x += dx * move_dist
                self.position.y += dy * move_dist
                
                energy_used = move_dist * self.spec.battery_consumption_rate
                self.battery = max(0, self.battery - energy_used)
                self.total_distance += move_dist
                self.total_energy += energy_used
                
                result['energy_used'] = energy_used
                result['distance_moved'] = move_dist
                
                self._check_task_progress(current_time, result)
        
        self.last_action = action
        return result
    
    def _can_accept_task(self, task: Task, verbose: bool = False) -> bool:
        """Check if agent can accept a task"""
        if self.vehicle_type not in task.allowed_vehicle_types:
            if verbose:
                print(f"[REJECT] {self.agent_id}: Wrong vehicle type")
            return False
        if self.current_load_kg + task.weight > self.spec.max_capacity_kg:
            if verbose:
                print(f"[REJECT] {self.agent_id}: Too heavy")
            return False
        if self.current_volume_m3 + task.volume > self.spec.max_volume_m3:
            if verbose:
                print(f"[REJECT] {self.agent_id}: Too big")
            return False
        
        # Check battery with safety margin
        est_distance = (self.position.distance_to(task.pickup_loc) + 
                    task.pickup_loc.distance_to(task.delivery_loc))
        required_battery = est_distance * self.spec.battery_consumption_rate * 0.5
        
        # Add buffer for return to depot
        depot_distance = task.delivery_loc.distance_to(self.depot_position)
        required_battery += depot_distance * self.spec.battery_consumption_rate * 0.3
        
        if self.battery < required_battery:
            if verbose:
                print(f"[REJECT] {self.agent_id}: Not enough battery")
            return False
        
        return True
    
    def _check_task_progress(self, current_time: float, result: Dict):
        """Check if current task progressed (pickup/delivery)"""
        if not self.current_task:
            return
        
        task = self.current_task
        
        # Check pickup
        if task.pickup_time is None:
            if self.position.distance_to(task.pickup_loc) < 5.0:
                task.pickup_time = current_time
                task.status = TaskStatus.IN_PROGRESS
                self.current_load_kg += task.weight
                self.current_volume_m3 += task.volume
                result['pickup'] = True
        
        # Check delivery
        elif task.delivery_time is None:
            if self.position.distance_to(task.delivery_loc) < 5.0:
                task.delivery_time = current_time
                task.status = TaskStatus.COMPLETED
                self.current_load_kg -= task.weight
                self.current_volume_m3 -= task.volume
                result['delivered'] = True
                result['task'] = task
                
                self.assigned_tasks.remove(task)
                self.task_history.append({
                    'task_id': task.id,
                    'on_time': current_time <= task.deadline,
                    'lateness': max(0, current_time - task.deadline)
                })
                
                # Get next task
                if self.assigned_tasks:
                    self.current_task = self.assigned_tasks[0]
                else:
                    self.current_task = None


# ==================== ENVIRONMENT ====================

class MultiAgentDeliveryEnv(gym.Env):
    """
    Multi-agent delivery environment with depot system,
    hybrid communication, and dynamic task spawning
    """
    
    def __init__(self, config: SimConfig = None, n_agents: int = 5):
        super().__init__()
        self.config = config or SimConfig()
        self.n_agents = n_agents
        
        # Spaces (per agent)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(58,), dtype=np.float32
        )
        self.action_space = RLDeliveryAgent.get_action_space()
        
        # Environment state
        self.agents: List[RLDeliveryAgent] = []
        self.depots: List[Position] = []
        self.tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.cancelled_tasks: List[Task] = []
        
        self.current_time = 0.0
        self.task_counter = 0
        
        # POIs
        self.boutiques: List[Position] = []
        self.restaurants: List[Position] = []
        self.customer_zones: List[Position] = []
        
        self.reset()
    
    def reset(self, seed=None):
        """Reset environment"""
        super().reset(seed=seed)
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        self.current_time = 0.0
        self.task_counter = 0
        self.tasks = []
        self.completed_tasks = []
        self.cancelled_tasks = []
        
        # Generate map
        self._generate_city()
        
        # Create agents
        self.agents = []
        vehicle_types = [VehicleType.TRUCK, VehicleType.VAN, 
                        VehicleType.DRONE, VehicleType.ROBOT, VehicleType.VAN]
        
        for i in range(self.n_agents):
            depot = random.choice(self.depots)
            vtype = vehicle_types[i % len(vehicle_types)]
            agent = RLDeliveryAgent(
                agent_id=f"AGENT_{i:02d}",
                vehicle_type=vtype,
                initial_position=Position(depot.x + random.uniform(-10, 10), 
                                        depot.y + random.uniform(-10, 10)),
                depot_position=depot
            )
            self.agents.append(agent)
        
        observations = [self._get_agent_observation(agent) for agent in self.agents]
        return observations, {}
    
    def _generate_city(self):
        """Generate city layout with depots, POIs"""
        city_size = self.config.city_size
        
        self.depots = [
            Position(city_size * 0.25, city_size * 0.25),
            Position(city_size * 0.75, city_size * 0.75)
        ]
        
        self.boutiques = [
            Position(random.uniform(0, city_size), random.uniform(0, city_size))
            for _ in range(self.config.n_boutiques)
        ]
        
        self.restaurants = [
            Position(random.uniform(0, city_size), random.uniform(0, city_size))
            for _ in range(self.config.n_restaurants)
        ]
        
        self.customer_zones = [
            Position(random.uniform(0, city_size), random.uniform(0, city_size))
            for _ in range(self.config.n_customer_zones)
        ]
    
    def _spawn_task(self):
        """Spawn a new delivery task"""
        task_type = random.choice(['boutique', 'restaurant', 'customer'])
        
        if task_type == 'boutique':
            pickup = random.choice(self.depots)
            delivery = random.choice(self.boutiques)
        elif task_type == 'restaurant':
            pickup = random.choice(self.restaurants)
            delivery = random.choice(self.customer_zones)
        else:
            pickup = random.choice(self.depots)
            delivery = random.choice(self.customer_zones)
        
        priority = random.choice([1, 3, 3, 5])  # weighted towards normal
        
        # Weight distribution matched to vehicle capabilities
        weight_type = random.random()
        if weight_type < 0.25:  # 25% light packages (for drones)
            weight = random.uniform(0.5, 4.5)
        elif weight_type < 0.50:  # 25% medium-light (for robots/drones)
            weight = random.uniform(4.5, 15.0)
        elif weight_type < 0.80:  # 30% medium (for vans/robots)
            weight = random.uniform(15.0, 50.0)
        else:  # 20% heavy (for trucks/vans)
            weight = random.uniform(50.0, 150.0)
        
        volume = weight * 0.01
        deadline = self.current_time + random.uniform(300, 1800)  # 5-30 min
        
        # Determine allowed vehicles based on weight
        allowed = []
        if weight <= 5:
            allowed.append(VehicleType.DRONE)
        if weight <= 30:
            allowed.append(VehicleType.ROBOT)
        if weight <= 200:
            allowed.append(VehicleType.VAN)
        allowed.append(VehicleType.TRUCK)  # Trucks can carry everything
        
        task = Task(
            id=f"TASK_{self.task_counter:04d}",
            pickup_loc=pickup,
            delivery_loc=delivery,
            created_time=self.current_time,
            deadline=deadline,
            priority=priority,
            weight=weight,
            volume=volume,
            fragile=random.random() < 0.2,
            temperature_sensitive=random.random() < 0.1,
            allowed_vehicle_types=allowed
        )
        
        self.tasks.append(task)
        self.task_counter += 1
    
    def step(self, actions: List[int]):
        """Execute one timestep"""
        dt = self.config.timestep
        self.current_time += dt
        
        # Spawn new tasks
        if random.random() < self.config.task_spawn_rate * dt:
            self._spawn_task()
        
        # Cancel some tasks
        for task in list(self.tasks):
            if (task.status == TaskStatus.PENDING and 
                random.random() < self.config.task_cancel_prob * dt):
                task.status = TaskStatus.CANCELLED
                self.tasks.remove(task)
                self.cancelled_tasks.append(task)
        
        # Update 5G availability
        for agent in self.agents:
            agent.v5g_available = random.random() < self.config.v5g_available_prob
        
        # Execute actions for all agents
        rewards = []
        dones = []
        infos = []
        
        for i, agent in enumerate(self.agents):
            visible_tasks = self._get_visible_tasks(agent)
            nearby_agents = self._get_nearby_agents(agent)
            
            action_result = agent.execute_action(
                actions[i], visible_tasks, nearby_agents, 
                self.current_time, dt
            )
            
            reward = agent.calculate_step_reward(action_result, self.current_time)
            rewards.append(reward)
            
            done = (agent.battery <= 0 or 
                   self.current_time >= self.config.sim_duration)
            dones.append(done)
            
            infos.append({
                'agent_id': agent.agent_id,
                'completed': agent.completed_count,
                'on_time_rate': (agent.on_time_count / agent.completed_count 
                               if agent.completed_count > 0 else 0)
            })
        
        # Move completed tasks
        for task in list(self.tasks):
            if task.status == TaskStatus.COMPLETED:
                self.tasks.remove(task)
                self.completed_tasks.append(task)
        
        observations = [self._get_agent_observation(agent) for agent in self.agents]
        done = all(dones) or self.current_time >= self.config.sim_duration
        
        return observations, rewards, done, False, infos
    
    def _get_visible_tasks(self, agent: RLDeliveryAgent) -> List[Task]:
        """Get tasks visible to agent"""
        visible = []
        for task in self.tasks:
            if task.status in [TaskStatus.PENDING, TaskStatus.ANNOUNCED]:
                dist_to_pickup = agent.position.distance_to(task.pickup_loc)
                if dist_to_pickup <= agent.spec.comm_range:
                    visible.append(task)
            elif task.assigned_to == agent.agent_id:
                visible.append(task)
        return visible
    
    def _get_nearby_agents(self, agent: RLDeliveryAgent) -> List[RLDeliveryAgent]:
        """Get agents within comm range"""
        nearby = []
        for other in self.agents:
            if other.agent_id != agent.agent_id:
                dist = agent.position.distance_to(other.position)
                if dist <= agent.spec.comm_range:
                    nearby.append(other)
        return nearby
    
    def _get_agent_observation(self, agent: RLDeliveryAgent) -> np.ndarray:
        """Get observation for specific agent"""
        visible_tasks = self._get_visible_tasks(agent)
        nearby_agents = self._get_nearby_agents(agent)
        return agent.get_observation(self.current_time, visible_tasks, nearby_agents)
    
    def get_metrics(self) -> Dict:
        """Calculate system-wide KPIs"""
        total_completed = len(self.completed_tasks)
        on_time = sum(1 for t in self.completed_tasks if t.delivery_time <= t.deadline)
        
        total_energy = sum(a.total_energy for a in self.agents)
        total_distance = sum(a.total_distance for a in self.agents)
        
        completions = [a.completed_count for a in self.agents]
        fairness = np.var(completions) if completions else 0
        
        lateness = [max(0, t.delivery_time - t.deadline) 
                   for t in self.completed_tasks if t.delivery_time]
        avg_lateness = np.mean(lateness) if lateness else 0
        
        return {
            'total_completed': total_completed,
            'on_time_deliveries': on_time,
            'on_time_rate': on_time / total_completed if total_completed > 0 else 0,
            'total_cancelled': len(self.cancelled_tasks),
            'total_pending': len([t for t in self.tasks if t.status == TaskStatus.PENDING]),
            'total_energy_kwh': total_energy,
            'energy_per_delivery': total_energy / total_completed if total_completed > 0 else 0,
            'total_distance_km': total_distance / 1000.0,
            'fairness_variance': fairness,
            'avg_lateness_sec': avg_lateness,
            'active_agents': sum(1 for a in self.agents if a.battery > 0),
        }


# ==================== TRAINING LOOP ====================

class RLTrainer:
    """Training loop for multi-agent RL"""
    
    def __init__(self, env: MultiAgentDeliveryEnv):
        self.env = env
        self.episode_rewards = []
        self.episode_metrics = []
    
    def train_random_baseline(self, n_episodes: int = 10):
        """Train with random actions (baseline)"""
        print("=" * 70)
        print("TRAINING RANDOM BASELINE POLICY")
        print("=" * 70)
        
        for episode in range(n_episodes):
            observations, _ = self.env.reset(seed=episode)
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < 1000:
                actions = [self.env.action_space.sample() 
                          for _ in range(self.env.n_agents)]
                
                observations, rewards, done, truncated, infos = self.env.step(actions)
                episode_reward += sum(rewards)
                step += 1
                
                if step % 100 == 0:
                    metrics = self.env.get_metrics()
                    print(f"  Step {step}: Completed={metrics['total_completed']}, "
                          f"OnTime={metrics['on_time_rate']:.2%}, "
                          f"Energy={metrics['total_energy_kwh']:.2f}kWh")
            
            metrics = self.env.get_metrics()
            self.episode_rewards.append(episode_reward)
            self.episode_metrics.append(metrics)
            
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Completed Deliveries: {metrics['total_completed']}")
            print(f"  On-Time Rate: {metrics['on_time_rate']:.2%}")
            print(f"  Energy/Delivery: {metrics['energy_per_delivery']:.3f} kWh")
            print("-" * 70)
        
        self._print_summary()
    
    def train_greedy_heuristic(self, n_episodes: int = 10):
        """Train with greedy heuristic policy"""
        print("=" * 70)
        print("TRAINING GREEDY HEURISTIC POLICY")
        print("=" * 70)
        
        for episode in range(n_episodes):
            observations, _ = self.env.reset(seed=episode)
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < 1000:
                actions = []
                
                for i, agent in enumerate(self.env.agents):
                    # Battery depleted - force depot return
                    if agent.battery <= 0:
                        actions.append(9)
                        continue
                    
                    # Low battery - return to depot
                    battery_percent = agent.battery / agent.spec.max_battery_kwh
                    if battery_percent < 0.35 or agent.is_returning_to_depot:
                        actions.append(9)
                        continue
                    
                    # Has task with sufficient battery - continue
                    if agent.current_task:
                        if agent.current_task.pickup_time is None:
                            target = agent.current_task.delivery_loc
                            est_dist = agent.position.distance_to(agent.current_task.pickup_loc) + \
                                      agent.current_task.pickup_loc.distance_to(target)
                        else:
                            target = agent.current_task.delivery_loc
                            est_dist = agent.position.distance_to(target)
                        
                        required = est_dist * agent.spec.battery_consumption_rate * 1.2
                        
                        if agent.battery >= required:
                            actions.append(14)  # Continue
                            continue
                        else:
                            # Abandon task and return to depot
                            agent.current_task.status = TaskStatus.PENDING
                            agent.current_task.assigned_to = None
                            agent.assigned_tasks.remove(agent.current_task)
                            agent.current_task = None
                            actions.append(9)
                            continue
                    
                    # Try to accept a new task
                    visible_tasks = self.env._get_visible_tasks(agent)
                    
                    if visible_tasks:
                        acceptable_tasks = [t for t in visible_tasks 
                                           if agent._can_accept_task(t, verbose=False)]
                        
                        if acceptable_tasks:
                            # Score tasks by distance and vehicle match
                            def score_task(task):
                                dist = agent.position.distance_to(task.pickup_loc)
                                
                                weight_score = 0
                                if agent.vehicle_type == VehicleType.DRONE:
                                    weight_score = -abs(task.weight - 2.5) * 10
                                elif agent.vehicle_type == VehicleType.ROBOT:
                                    weight_score = -abs(task.weight - 15) * 5
                                elif agent.vehicle_type == VehicleType.VAN:
                                    weight_score = -abs(task.weight - 50) * 2
                                else:  # TRUCK
                                    weight_score = task.weight * 0.5 if task.weight > 50 else -20
                                
                                return (dist / (task.priority + 1)) + weight_score
                            
                            best_task = min(acceptable_tasks, key=score_task)
                            task_idx = visible_tasks.index(best_task)
                            actions.append(min(task_idx, 4))
                        else:
                            actions.append(random.choice([5, 6, 7, 8]))
                    else:
                        actions.append(random.choice([5, 6, 7, 8]))
                
                observations, rewards, done, truncated, infos = self.env.step(actions)
                episode_reward += sum(rewards)
                step += 1
                
                if step % 100 == 0:
                    metrics = self.env.get_metrics()
                    active_agents = sum(1 for a in self.env.agents if a.battery > 0)
                    print(f"  Step {step}: Completed={metrics['total_completed']}, "
                          f"OnTime={metrics['on_time_rate']:.2%}, "
                          f"ActiveAgents={active_agents}/{len(self.env.agents)}")
            
            metrics = self.env.get_metrics()
            self.episode_rewards.append(episode_reward)
            self.episode_metrics.append(metrics)
            
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Completed Deliveries: {metrics['total_completed']}")
            print(f"  On-Time Rate: {metrics['on_time_rate']:.2%}")
            print(f"  Energy/Delivery: {metrics['energy_per_delivery']:.3f} kWh")
            print("-" * 70)
        
        self._print_summary()
    
    def _print_summary(self):
        """Print training summary"""
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        
        avg_reward = np.mean(self.episode_rewards)
        avg_completed = np.mean([m['total_completed'] for m in self.episode_metrics])
        avg_on_time_rate = np.mean([m['on_time_rate'] for m in self.episode_metrics])
        avg_energy = np.mean([m['energy_per_delivery'] for m in self.episode_metrics 
                             if m['energy_per_delivery'] > 0])
        
        print(f"Average Episode Reward: {avg_reward:.2f}")
        print(f"Average Completed Deliveries: {avg_completed:.1f}")
        print(f"Average On-Time Rate: {avg_on_time_rate:.2%}")
        print(f"Average Energy per Delivery: {avg_energy:.3f} kWh")
        print("=" * 70)
        
        print("\nPER-AGENT PERFORMANCE:")
        for agent in self.env.agents:
            on_time_rate = agent.on_time_count / max(agent.completed_count, 1)
            print(f"  {agent.agent_id} ({agent.vehicle_type.value}):")
            print(f"    Completed: {agent.completed_count}")
            print(f"    On-Time Rate: {on_time_rate:.2%}")
            print(f"    Total Distance: {agent.total_distance/1000:.2f} km")
            print(f"    Energy Used: {agent.total_energy:.2f} kWh")


# ==================== STABLE-BASELINES3 WRAPPER ====================

class StableBaselinesWrapper(gym.Env):
    """Wrapper for stable-baselines3 compatibility (single-agent view)"""
    
    def __init__(self, agent_id: int = 0, n_agents: int = 5):
        super().__init__()
        self.env = MultiAgentDeliveryEnv(n_agents=n_agents)
        self.agent_id = agent_id
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def reset(self, seed=None):
        observations, info = self.env.reset(seed=seed)
        return observations[self.agent_id], info
    
    def step(self, action):
        actions = [self.env.action_space.sample() for _ in range(self.env.n_agents)]
        actions[self.agent_id] = action
        
        observations, rewards, done, truncated, infos = self.env.step(actions)
        
        return (observations[self.agent_id], 
                rewards[self.agent_id], 
                done, 
                truncated,
                infos[self.agent_id])


# ==================== MAIN ====================

def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("SMARTFLEET: RL-BASED MULTI-AGENT DELIVERY SYSTEM")
    print("=" * 70)
    
    config = SimConfig(
        city_size=1000,
        sim_duration=600,
        task_spawn_rate=0.5,
    )
    
    env = MultiAgentDeliveryEnv(config=config, n_agents=5)
    
    print(f"\nEnvironment Configuration:")
    print(f"  City Size: {config.city_size}m x {config.city_size}m")
    print(f"  Number of Agents: {env.n_agents}")
    print(f"  Simulation Duration: {config.sim_duration}s")
    print(f"\nAgent Fleet:")
    for agent in env.agents:
        print(f"  {agent.agent_id}: {agent.vehicle_type.value} "
              f"(capacity: {agent.spec.max_capacity_kg}kg, "
              f"battery: {agent.spec.max_battery_kwh}kWh)")
    
    trainer = RLTrainer(env)
    
    print("\n" + "=" * 70)
    print("PHASE 1: RANDOM BASELINE")
    print("=" * 70)
    trainer.train_random_baseline(n_episodes=3)
    
    print("\n" + "=" * 70)
    print("PHASE 2: GREEDY HEURISTIC")
    print("=" * 70)
    trainer.episode_rewards = []
    trainer.episode_metrics = []
    trainer.train_greedy_heuristic(n_episodes=3)
    
    print("\n[Program completed successfully]")


if __name__ == "__main__":
    main()

