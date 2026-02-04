"""
================================================================================
SMARTFLEET: INTEGRATED RL MULTI-AGENT DELIVERY SYSTEM WITH SUMO
================================================================================

This is the integrated version that connects your RL agents to the real 
Tunis SUMO map. Agents now drive on actual roads with real traffic!

Original: delivery_agent.py (standalone simulation)
This version: delivery_agent_sumo.py (SUMO integrated)

Author: Maram Ouelhazi
Project: IEEE RAS x VTS SmartFleet Challenge
================================================================================
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import random
import time
import os
import sys
from collections import deque, defaultdict
import json
import xml.etree.ElementTree as ET

# ==================== SUMO SETUP ====================

# Set SUMO_HOME - UPDATE THIS PATH IF NEEDED!
SUMO_HOME = 'C:\\Program Files\\sumo-1.25.0'
os.environ['SUMO_HOME'] = SUMO_HOME

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    print("⚠ WARNING: SUMO_HOME not set. Set it to your SUMO installation path.")

try:
    import traci
    import sumolib
    SUMO_AVAILABLE = True
    print("✓ SUMO libraries loaded successfully")
except ImportError:
    SUMO_AVAILABLE = False
    print("⚠ WARNING: SUMO libraries not found. Running in simulation-only mode.")


# ==================== CONFIGURATION ====================

@dataclass
class SimConfig:
    """Simulation configuration"""
    # City/Map settings
    city_size: int = 1000  # meters (your coordinate system)
    
    # SUMO settings - NOW USING SMARTFLEET SCENARIO
    sumo_config_file: str = "smartfleet.sumocfg"
    sumo_net_file: str = "finished_map_cleaning.net.xml"
    sumo_add_file: str = "smartfleet_zones.add.xml"
    scenario_file: str = "smartfleet_scenario.json"  # New: verified locations
    use_sumo_gui: bool = True
    
    # Depot/Zone settings (loaded from scenario)
    n_depots: int = 2
    n_boutiques: int = 5
    n_restaurants: int = 5
    n_customer_zones: int = 10
    
    # Time settings
    sim_duration: int = 600  # seconds (10 minutes for testing)
    timestep: float = 1.0  # seconds per step
    
    # Communication settings
    v2v_range: float = 300.0  # meters
    v2v_interval: float = 1.0  # seconds
    v5g_available_prob: float = 0.7  # 70% coverage
    v5g_query_cost: float = 0.5  # energy cost
    
    # Task spawning
    task_spawn_rate: float = 0.15  # tasks per second
    task_cancel_prob: float = 0.02  # 2% cancel rate
    
    # Environment dynamics
    road_block_prob: float = 0.02
    no_fly_zone_prob: float = 0.01
    
    # Logging
    verbose: bool = True
    log_every_n_steps: int = 50


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


class AgentState(Enum):
    """Agent state machine for clear logging"""
    IDLE = "idle"
    MOVING_TO_PICKUP = "moving_to_pickup"
    PICKING_UP = "picking_up"
    MOVING_TO_DELIVERY = "moving_to_delivery"
    DELIVERING = "delivering"
    RETURNING_TO_DEPOT = "returning_to_depot"
    CHARGING = "charging"
    WAITING = "waiting"


@dataclass
class Position:
    """2D Position with utility methods"""
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def __str__(self):
        return f"({self.x:.1f}, {self.y:.1f})"


@dataclass
class Task:
    """Delivery task/parcel"""
    id: str
    pickup_loc: Position
    delivery_loc: Position
    created_time: float
    deadline: float
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
    
    # SUMO-specific
    pickup_edge: Optional[str] = None
    delivery_edge: Optional[str] = None


@dataclass
class VehicleSpec:
    """Vehicle specifications"""
    vehicle_type: VehicleType
    max_capacity_kg: float
    max_volume_m3: float
    max_battery_kwh: float
    max_speed_ms: float
    battery_consumption_rate: float  # kWh per meter
    comm_range: float
    can_access_no_fly: bool = False
    sumo_vtype: str = "car"  # SUMO vehicle type ID


# Vehicle specifications database with SUMO vehicle types
VEHICLE_SPECS = {
    VehicleType.TRUCK: VehicleSpec(
        vehicle_type=VehicleType.TRUCK,
        max_capacity_kg=500.0,
        max_volume_m3=20.0,
        max_battery_kwh=2000.0,
        max_speed_ms=15.0,
        battery_consumption_rate=0.02,
        comm_range=800.0,
        can_access_no_fly=False,
        sumo_vtype="truck"
    ),
    VehicleType.VAN: VehicleSpec(
        vehicle_type=VehicleType.VAN,
        max_capacity_kg=200.0,
        max_volume_m3=8.0,
        max_battery_kwh=150.0,
        max_speed_ms=18.0,
        battery_consumption_rate=0.03,
        comm_range=700.0,
        can_access_no_fly=False,
        sumo_vtype="van"
    ),
    VehicleType.DRONE: VehicleSpec(
        vehicle_type=VehicleType.DRONE,
        max_capacity_kg=5.0,
        max_volume_m3=0.5,
        max_battery_kwh=200.0,
        max_speed_ms=25.0,
        battery_consumption_rate=0.05,
        comm_range=600.0,
        can_access_no_fly=True,
        sumo_vtype="drone"  # We'll create this type
    ),
    VehicleType.ROBOT: VehicleSpec(
        vehicle_type=VehicleType.ROBOT,
        max_capacity_kg=30.0,
        max_volume_m3=2.0,
        max_battery_kwh=100.0,
        max_speed_ms=8.0,
        battery_consumption_rate=0.02,
        comm_range=700.0,
        can_access_no_fly=False,
        sumo_vtype="robot"  # We'll create this type
    )
}


# ==================== SUMO ZONE MANAGER ====================

class SUMOZoneManager:
    """
    Manages the mapping between your RL zones and SUMO TAZ zones.
    Handles coordinate conversion and edge finding.
    """
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.network = None
        self.sumo_bounds = None
        
        # Zone storage
        self.delivery_zones = []   # Green zones
        self.pickup_zones = []     # Purple zones  
        self.depot_zones = []      # Blue zones
        self.no_fly_zones = []     # Red zones
        
        # Edge caches
        self.zone_edges = {}  # zone_id -> list of edges
        self.valid_edges = []  # All drivable edges
        
    def load_network(self):
        """Load SUMO network for coordinate conversion"""
        if not SUMO_AVAILABLE:
            print("⚠ SUMO not available, using simulated coordinates")
            return False
            
        try:
            self.network = sumolib.net.readNet(self.config.sumo_net_file)
            bounds = self.network.getBoundary()
            self.sumo_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])
            
            print(f"✓ Network loaded: {self.config.sumo_net_file}")
            print(f"  Bounds: x=[{bounds[0]:.1f}, {bounds[2]:.1f}], y=[{bounds[1]:.1f}, {bounds[3]:.1f}]")
            
            # Cache valid edges
            for edge in self.network.getEdges():
                if not edge.getID().startswith(':'):  # Skip internal edges
                    self.valid_edges.append(edge.getID())
            
            print(f"  Valid edges: {len(self.valid_edges)}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading network: {e}")
            return False
    
    def load_zones(self):
        """Load zones from smartfleet_scenario.json (verified connected locations)"""
        try:
            # First try to load from scenario file (verified locations)
            if os.path.exists(self.config.scenario_file):
                with open(self.config.scenario_file, 'r') as f:
                    scenario = json.load(f)
                
                # Load verified zones
                for depot in scenario.get('depots', []):
                    zone_data = {
                        'id': depot['id'],
                        'color': '98,189,239',  # Blue
                        'edges': [depot['edge']],
                        'center': Position(depot['x'], depot['y'])
                    }
                    self.depot_zones.append(zone_data)
                    self.zone_edges[depot['id']] = [depot['edge']]
                
                for pickup in scenario.get('pickups', []):
                    zone_data = {
                        'id': pickup['id'],
                        'color': '226,39,246',  # Purple
                        'edges': [pickup['edge']],
                        'center': Position(pickup['x'], pickup['y'])
                    }
                    self.pickup_zones.append(zone_data)
                    self.zone_edges[pickup['id']] = [pickup['edge']]
                
                for delivery in scenario.get('deliveries', []):
                    zone_data = {
                        'id': delivery['id'],
                        'color': '98,246,40',  # Green
                        'edges': [delivery['edge']],
                        'center': Position(delivery['x'], delivery['y'])
                    }
                    self.delivery_zones.append(zone_data)
                    self.zone_edges[delivery['id']] = [delivery['edge']]
                
                # Store connected edges for spawning
                self.connected_edges = scenario.get('connected_edges', [])
                self.spawn_edges = scenario.get('spawn_edges', [])
                
                # Store bounds
                bounds = scenario.get('bounds', {})
                self.scenario_bounds = bounds
                
                print(f"✓ Scenario loaded from {self.config.scenario_file}")
                print(f"  Depot zones: {len(self.depot_zones)}")
                print(f"  Pickup zones: {len(self.pickup_zones)}")
                print(f"  Delivery zones: {len(self.delivery_zones)}")
                print(f"  Connected edges: {len(self.connected_edges)}")
                
                return True
            
            # Fallback to XML file
            return self._load_zones_from_xml()
            
        except Exception as e:
            print(f"✗ Error loading scenario: {e}")
            return self._load_zones_from_xml()
    
    def _load_zones_from_xml(self):
        """Fallback: Load TAZ zones from additional file"""
        try:
            tree = ET.parse(self.config.sumo_add_file)
            root = tree.getroot()
            
            for taz in root.findall('.//taz'):
                taz_id = taz.get('id')
                color = taz.get('color', '0,0,0')
                
                # Get source/sink edges
                sources = [s.get('id') for s in taz.findall('tazSource')]
                sinks = [s.get('id') for s in taz.findall('tazSink')]
                edges = list(set(sources + sinks))
                
                if not edges:
                    continue
                
                zone_data = {
                    'id': taz_id,
                    'color': color,
                    'edges': edges,
                    'center': self._calculate_zone_center(edges)
                }
                
                # Categorize by color
                if '98,246,40' in color or '98, 246, 40' in color:
                    self.delivery_zones.append(zone_data)
                elif '226,39,246' in color or '226, 39, 246' in color:
                    self.pickup_zones.append(zone_data)
                elif '98,189,239' in color or '98, 189, 239' in color:
                    self.depot_zones.append(zone_data)
                elif '246,99,22' in color or '218,34,40' in color:
                    self.no_fly_zones.append(zone_data)
                else:
                    self.delivery_zones.append(zone_data)
                
                self.zone_edges[taz_id] = edges
            
            print(f"✓ Zones loaded from {self.config.sumo_add_file}")
            print(f"  Delivery zones (green): {len(self.delivery_zones)}")
            print(f"  Pickup zones (purple): {len(self.pickup_zones)}")
            print(f"  Depot zones (blue): {len(self.depot_zones)}")
            print(f"  No-fly zones (red): {len(self.no_fly_zones)}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading zones from XML: {e}")
            return False
    
    def _calculate_zone_center(self, edges: List[str]) -> Position:
        """Calculate center position of a zone based on its edges"""
        if not self.network or not edges:
            return Position(500, 500)  # Default center
        
        x_coords = []
        y_coords = []
        
        for edge_id in edges:
            try:
                edge = self.network.getEdge(edge_id)
                shape = edge.getShape()
                if shape:
                    for point in shape:
                        x_coords.append(point[0])
                        y_coords.append(point[1])
            except:
                continue
        
        if x_coords and y_coords:
            sumo_x = np.mean(x_coords)
            sumo_y = np.mean(y_coords)
            return self.sumo_to_local(sumo_x, sumo_y)
        
        return Position(500, 500)
    
    def local_to_sumo(self, position: Position) -> Tuple[float, float]:
        """Convert local coordinates (0-1000) to SUMO coordinates"""
        if not self.sumo_bounds:
            return (position.x, position.y)
        
        min_x, min_y, max_x, max_y = self.sumo_bounds
        sumo_x = min_x + (position.x / self.config.city_size) * (max_x - min_x)
        sumo_y = min_y + (position.y / self.config.city_size) * (max_y - min_y)
        
        return (sumo_x, sumo_y)
    
    def sumo_to_local(self, sumo_x: float, sumo_y: float) -> Position:
        """Convert SUMO coordinates to local coordinates (0-1000)"""
        if not self.sumo_bounds:
            return Position(sumo_x, sumo_y)
        
        min_x, min_y, max_x, max_y = self.sumo_bounds
        local_x = ((sumo_x - min_x) / (max_x - min_x)) * self.config.city_size
        local_y = ((sumo_y - min_y) / (max_y - min_y)) * self.config.city_size
        
        # Clamp to bounds
        local_x = max(0, min(self.config.city_size, local_x))
        local_y = max(0, min(self.config.city_size, local_y))
        
        return Position(local_x, local_y)
    
    def find_nearest_edge(self, position: Position, radius: float = 200.0) -> Optional[str]:
        """Find the nearest SUMO edge to a position"""
        if not self.network:
            return None
        
        sumo_x, sumo_y = self.local_to_sumo(position)
        
        try:
            nearby = self.network.getNeighboringEdges(sumo_x, sumo_y, r=radius)
            if nearby:
                # Sort by distance and return closest non-internal edge
                nearby.sort(key=lambda x: x[1])
                for edge, dist in nearby:
                    if not edge.getID().startswith(':'):
                        return edge.getID()
        except:
            pass
        
        return None
    
    def get_edge_position(self, edge_id: str) -> Optional[Position]:
        """Get the center position of an edge in local coordinates"""
        if not self.network:
            return None
        
        try:
            edge = self.network.getEdge(edge_id)
            shape = edge.getShape()
            if shape:
                # Use middle point of edge
                mid_idx = len(shape) // 2
                sumo_x, sumo_y = shape[mid_idx]
                return self.sumo_to_local(sumo_x, sumo_y)
        except:
            pass
        
        return None
    
    def get_random_zone_position(self, zone_type: str) -> Tuple[Position, Optional[str]]:
        """Get a random position from a zone type, returns (position, edge_id)"""
        zones = {
            'delivery': self.delivery_zones,
            'pickup': self.pickup_zones,
            'depot': self.depot_zones,
            'restaurant': self.pickup_zones,  # Alias
            'customer': self.delivery_zones   # Alias
        }.get(zone_type, self.delivery_zones)
        
        if not zones:
            # Fallback to random position
            return Position(random.uniform(100, 900), random.uniform(100, 900)), None
        
        zone = random.choice(zones)
        
        if zone['edges']:
            edge_id = random.choice(zone['edges'])
            pos = self.get_edge_position(edge_id)
            if pos:
                return pos, edge_id
        
        return zone['center'], None


# ==================== RL DELIVERY AGENT (SUMO INTEGRATED) ====================

class RLDeliveryAgentSUMO:
    """
    RL-based delivery agent integrated with SUMO.
    
    Key differences from original:
    - Position updates come from SUMO vehicle
    - Movement is handled by SUMO routing
    - Battery consumption based on actual distance traveled
    - Real traffic affects travel time
    """
    
    def __init__(self, agent_id: str, vehicle_type: VehicleType, 
                 initial_position: Position, depot_position: Position,
                 zone_manager: SUMOZoneManager):
        
        self.agent_id = agent_id
        self.vehicle_type = vehicle_type
        self.spec = VEHICLE_SPECS[vehicle_type]
        self.zone_manager = zone_manager
        
        # Physical state
        self.position = initial_position
        self.depot_position = depot_position
        self.velocity = 0.0
        self.heading = 0.0
        
        # SUMO integration
        self.sumo_vehicle_id = f"agent_{agent_id}"
        self.current_edge = None
        self.target_edge = None
        self.is_spawned_in_sumo = False
        
        # Resources
        self.battery = self.spec.max_battery_kwh
        self.current_load_kg = 0.0
        self.current_volume_m3 = 0.0
        
        # State machine
        self.state = AgentState.IDLE
        self.previous_state = AgentState.IDLE
        
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
        
        # Step-by-step logging
        self.action_log: List[Dict] = []
        
        # RL state
        self.last_action = None
        self.last_action_name = "none"
        self.episode_reward = 0.0
        
        # Navigation
        self.is_returning_to_depot = False
        self.last_position = initial_position
    
    def get_state_description(self) -> str:
        """Get human-readable state description for logging"""
        base = f"{self.state.value}"
        
        if self.current_task:
            if self.state == AgentState.MOVING_TO_PICKUP:
                return f"Moving to pickup {self.current_task.id}"
            elif self.state == AgentState.MOVING_TO_DELIVERY:
                return f"Delivering {self.current_task.id}"
        
        if self.state == AgentState.RETURNING_TO_DEPOT:
            return f"Returning to depot (battery: {self.battery:.1f}kWh)"
        
        return base
    
    def log_action(self, step: int, action: int, action_name: str, result: Dict, current_time: float):
        """Log action for step-by-step visualization"""
        log_entry = {
            'step': step,
            'time': current_time,
            'agent_id': self.agent_id,
            'vehicle_type': self.vehicle_type.value,
            'position': str(self.position),
            'state': self.state.value,
            'action_id': action,
            'action_name': action_name,
            'battery_percent': (self.battery / self.spec.max_battery_kwh) * 100,
            'current_task': self.current_task.id if self.current_task else None,
            'tasks_queued': len(self.assigned_tasks),
            'completed_total': self.completed_count,
            'result': result
        }
        self.action_log.append(log_entry)
        return log_entry
    
    # ==================== OBSERVATION SPACE ====================
    
    def get_observation(self, current_time: float, 
                       visible_tasks: List[Task],
                       nearby_agents: List['RLDeliveryAgentSUMO']) -> np.ndarray:
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
        city_size = self.zone_manager.config.city_size
        idx = 0
        
        # Self state (10)
        obs[idx:idx+2] = self.position.to_array() / city_size
        obs[idx+2] = self.battery / self.spec.max_battery_kwh
        obs[idx+3] = 1.0 - (self.current_load_kg / self.spec.max_capacity_kg)
        obs[idx+4] = 1.0 - (self.current_volume_m3 / self.spec.max_volume_m3)
        obs[idx+5] = self.velocity / self.spec.max_speed_ms
        obs[idx+6] = np.sin(self.heading)
        obs[idx+7] = np.cos(self.heading)
        obs[idx+8] = min(len(self.assigned_tasks) / 5.0, 1.0)
        obs[idx+9] = (current_time % 3600) / 3600.0
        idx += 10
        
        # Current task (6)
        if self.current_task:
            target = (self.current_task.pickup_loc if self.current_task.pickup_time is None 
                     else self.current_task.delivery_loc)
            rel_pos = (target.to_array() - self.position.to_array()) / city_size
            obs[idx:idx+2] = np.clip(rel_pos, -1, 1)
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
                rel_pos = (t.pickup_loc.to_array() - self.position.to_array()) / city_size
                obs[idx:idx+2] = np.clip(rel_pos, -1, 1)
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
                rel_pos = (n.position.to_array() - self.position.to_array()) / city_size
                obs[idx:idx+2] = np.clip(rel_pos, -1, 1)
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
        5-8: MOVE direction (NORTH, EAST, SOUTH, WEST) - for exploration
        9: GO_TO_DEPOT - return for charging
        10-12: REQUEST_HANDOVER to neighbor[0-2]
        13: QUERY_5G for routing
        14: DO_NOTHING / CONTINUE current task
        """
        return spaces.Discrete(15)
    
    @staticmethod
    def get_action_name(action: int) -> str:
        """Get human-readable action name"""
        action_names = {
            0: "ACCEPT_TASK_0",
            1: "ACCEPT_TASK_1",
            2: "ACCEPT_TASK_2",
            3: "ACCEPT_TASK_3",
            4: "ACCEPT_TASK_4",
            5: "MOVE_NORTH",
            6: "MOVE_EAST",
            7: "MOVE_SOUTH",
            8: "MOVE_WEST",
            9: "GO_TO_DEPOT",
            10: "HANDOVER_TO_0",
            11: "HANDOVER_TO_1",
            12: "HANDOVER_TO_2",
            13: "QUERY_5G",
            14: "CONTINUE"
        }
        return action_names.get(action, f"UNKNOWN_{action}")
    
    def decode_action(self, action: int, visible_tasks: List[Task],
                     nearby_agents: List['RLDeliveryAgentSUMO']) -> Dict:
        """Decode discrete action into executable command"""
        
        if action < 5:
            # Accept task
            task_idx = action
            sorted_tasks = sorted(visible_tasks, 
                                 key=lambda t: self.position.distance_to(t.pickup_loc))
            if task_idx < len(sorted_tasks):
                return {'type': 'accept_task', 'task': sorted_tasks[task_idx]}
            return {'type': 'continue'}
        
        elif action < 9:
            # Movement (exploration)
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
            return {'type': 'continue'}
        
        elif action == 13:
            return {'type': 'query_5g'}
        
        return {'type': 'continue'}
    
    # ==================== REWARD CALCULATION ====================
    
    def calculate_step_reward(self, action_result: Dict, current_time: float) -> float:
        """
        Multi-objective reward function:
        - On-time delivery: +10
        - Late delivery: +3 (within 5 min) or -8 (very late)
        - Pickup: +0.5
        - Energy efficiency: -0.01 * energy_used
        - Battery critical: -0.5
        - Cooperation bonus: +1 to +2
        - Time penalty: -0.01 per step
        """
        reward = -0.01  # Base time penalty
        
        # Delivery rewards
        if action_result.get('delivered'):
            task = action_result['task']
            if current_time <= task.deadline:
                reward += 10.0
                self.on_time_count += 1
            else:
                lateness = current_time - task.deadline
                if lateness < 300:
                    reward += 3.0
                else:
                    reward -= 8.0
            self.completed_count += 1
        
        # Pickup reward
        if action_result.get('pickup'):
            reward += 0.5
        
        # Task acceptance
        if action_result.get('accepted'):
            reward += 0.2
        
        # Cancelled task penalty
        if action_result.get('task_cancelled'):
            reward -= 2.0
        
        # Failure penalty
        if action_result.get('failed'):
            reward -= 5.0
            self.failed_count += 1
        
        # Cooperation rewards
        if action_result.get('handover_accepted'):
            reward += 1.0
            self.helped_neighbors += 1
        
        if action_result.get('handover_helped'):
            reward += 2.0
        
        # Energy penalty
        energy_used = action_result.get('energy_used', 0)
        reward -= 0.01 * energy_used
        
        # 5G cost
        if action_result.get('used_5g'):
            reward -= 0.5
        
        # Battery critical warning
        if self.battery < self.spec.max_battery_kwh * 0.15:
            reward -= 0.5
        
        # Recharge bonus
        if action_result.get('recharged'):
            reward += 0.3
        
        self.episode_reward += reward
        return reward
    
    # ==================== TASK MANAGEMENT ====================
    
    def can_accept_task(self, task: Task, verbose: bool = False) -> bool:
        """Check if agent can accept a task"""
        # Vehicle type check
        if self.vehicle_type not in task.allowed_vehicle_types:
            if verbose:
                print(f"    [REJECT] {self.agent_id}: Wrong vehicle type for {task.id}")
            return False
        
        # Capacity check
        if self.current_load_kg + task.weight > self.spec.max_capacity_kg:
            if verbose:
                print(f"    [REJECT] {self.agent_id}: Too heavy for {task.id}")
            return False
        
        # Volume check
        if self.current_volume_m3 + task.volume > self.spec.max_volume_m3:
            if verbose:
                print(f"    [REJECT] {self.agent_id}: Too big for {task.id}")
            return False
        
        # Battery check
        est_distance = (self.position.distance_to(task.pickup_loc) + 
                       task.pickup_loc.distance_to(task.delivery_loc))
        depot_distance = task.delivery_loc.distance_to(self.depot_position)
        
        required_battery = (est_distance + depot_distance) * self.spec.battery_consumption_rate * 1.3
        
        if self.battery < required_battery:
            if verbose:
                print(f"    [REJECT] {self.agent_id}: Not enough battery for {task.id}")
            return False
        
        return True
    
    def _update_state(self):
        """Update agent state machine"""
        self.previous_state = self.state
        
        if self.is_returning_to_depot:
            self.state = AgentState.RETURNING_TO_DEPOT
        elif self.current_task:
            if self.current_task.pickup_time is None:
                # Check if at pickup
                if self.position.distance_to(self.current_task.pickup_loc) < 10.0:
                    self.state = AgentState.PICKING_UP
                else:
                    self.state = AgentState.MOVING_TO_PICKUP
            else:
                # Check if at delivery
                if self.position.distance_to(self.current_task.delivery_loc) < 10.0:
                    self.state = AgentState.DELIVERING
                else:
                    self.state = AgentState.MOVING_TO_DELIVERY
        else:
            self.state = AgentState.IDLE
    
    # ==================== EXECUTION ====================
    
    def execute_action(self, action: int, visible_tasks: List[Task],
                      nearby_agents: List['RLDeliveryAgentSUMO'], 
                      current_time: float, dt: float,
                      sumo_controller: 'SUMOController' = None) -> Dict:
        """Execute decoded action and return result"""
        
        command = self.decode_action(action, visible_tasks, nearby_agents)
        result = {'energy_used': 0, 'distance_moved': 0, 'action_type': command['type']}
        
        self.last_action = action
        self.last_action_name = self.get_action_name(action)
        
        # Store last position for distance calculation
        self.last_position = Position(self.position.x, self.position.y)
        
        if command['type'] == 'accept_task':
            task = command['task']
            if self.can_accept_task(task):
                task.status = TaskStatus.ASSIGNED
                task.assigned_to = self.agent_id
                self.assigned_tasks.append(task)
                if self.current_task is None:
                    self.current_task = task
                    # Set SUMO target to pickup location
                    if sumo_controller:
                        sumo_controller.set_vehicle_target(self, task.pickup_loc)
                result['accepted'] = True
                result['task_id'] = task.id
        
        elif command['type'] == 'move':
            dx, dy = command['direction']
            distance = self.spec.max_speed_ms * dt
            new_x = max(0, min(1000, self.position.x + dx * distance))
            new_y = max(0, min(1000, self.position.y + dy * distance))
            
            moved_dist = np.sqrt((new_x - self.position.x)**2 + (new_y - self.position.y)**2)
            
            # Update position (SUMO will override if connected)
            if not sumo_controller or not self.is_spawned_in_sumo:
                self.position.x = new_x
                self.position.y = new_y
            
            energy_used = moved_dist * self.spec.battery_consumption_rate
            self.battery = max(0, self.battery - energy_used)
            self.total_distance += moved_dist
            self.total_energy += energy_used
            
            result['energy_used'] = energy_used
            result['distance_moved'] = moved_dist
        
        elif command['type'] == 'go_to_depot':
            self.is_returning_to_depot = True
            
            if sumo_controller and self.is_spawned_in_sumo:
                sumo_controller.set_vehicle_target(self, self.depot_position)
            else:
                # Manual movement toward depot
                self._move_toward(self.depot_position, dt, result)
            
            # Check if reached depot
            if self.position.distance_to(self.depot_position) < 15.0:
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
                if neighbor.can_accept_task(self.current_task):
                    self.current_task.assigned_to = neighbor.agent_id
                    neighbor.assigned_tasks.append(self.current_task)
                    self.assigned_tasks.remove(self.current_task)
                    self.current_task = None
                    result['handover_accepted'] = True
        
        elif command['type'] == 'continue':
            if self.current_task:
                # Move toward current target
                if self.current_task.pickup_time is None:
                    target = self.current_task.pickup_loc
                else:
                    target = self.current_task.delivery_loc
                
                if sumo_controller and self.is_spawned_in_sumo:
                    # SUMO handles movement
                    pass
                else:
                    # Manual movement
                    self._move_toward(target, dt, result)
        
        # Check task progress
        if self.current_task:
            self._check_task_progress(current_time, result)
        
        # Update state machine
        self._update_state()
        
        return result
    
    def _move_toward(self, target: Position, dt: float, result: Dict):
        """Move toward a target position (used when SUMO not controlling)"""
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
            
            result['energy_used'] = result.get('energy_used', 0) + energy_used
            result['distance_moved'] = result.get('distance_moved', 0) + move_dist
    
    def _check_task_progress(self, current_time: float, result: Dict):
        """Check if current task progressed (pickup/delivery)"""
        if not self.current_task:
            return
        
        task = self.current_task
        
        # Check pickup
        if task.pickup_time is None:
            if self.position.distance_to(task.pickup_loc) < 15.0:
                task.pickup_time = current_time
                task.status = TaskStatus.IN_PROGRESS
                self.current_load_kg += task.weight
                self.current_volume_m3 += task.volume
                result['pickup'] = True
                result['task_id'] = task.id
        
        # Check delivery
        elif task.delivery_time is None:
            if self.position.distance_to(task.delivery_loc) < 15.0:
                task.delivery_time = current_time
                task.status = TaskStatus.COMPLETED
                self.current_load_kg = max(0, self.current_load_kg - task.weight)
                self.current_volume_m3 = max(0, self.current_volume_m3 - task.volume)
                result['delivered'] = True
                result['task'] = task
                
                self.assigned_tasks.remove(task)
                self.task_history.append({
                    'task_id': task.id,
                    'on_time': current_time <= task.deadline,
                    'lateness': max(0, current_time - task.deadline),
                    'delivery_time': current_time
                })
                
                # Get next task
                if self.assigned_tasks:
                    self.current_task = self.assigned_tasks[0]
                else:
                    self.current_task = None


# ==================== SUMO CONTROLLER ====================

class SUMOController:
    """
    Controls the SUMO simulation and syncs with RL agents.
    Handles vehicle spawning, routing, and position updates.
    """
    
    def __init__(self, config: SimConfig, zone_manager: SUMOZoneManager):
        self.config = config
        self.zone_manager = zone_manager
        self.is_connected = False
        self.step_count = 0
        
        # Vehicle type definitions for SUMO
        self.vehicle_types_defined = False
    
    def start(self):
        """Start SUMO simulation"""
        if not SUMO_AVAILABLE:
            print("⚠ SUMO not available, running in simulation mode")
            return False
        
        try:
            # Use full path to SUMO binary on Windows
            sumo_home = os.environ.get('SUMO_HOME', 'C:\\Program Files\\sumo-1.25.0')
            
            if self.config.use_sumo_gui:
                sumo_binary = os.path.join(sumo_home, 'bin', 'sumo-gui.exe')
            else:
                sumo_binary = os.path.join(sumo_home, 'bin', 'sumo.exe')
            
            # Check if binary exists
            if not os.path.exists(sumo_binary):
                print(f"✗ SUMO binary not found at: {sumo_binary}")
                print(f"  Please update SUMO_HOME in the script (line 27)")
                return False
            
            print(f"✓ Found SUMO at: {sumo_binary}")
            
            sumo_cmd = [sumo_binary, "-c", self.config.sumo_config_file,
                       "--start", "--quit-on-end", "false"]
            
            traci.start(sumo_cmd)
            self.is_connected = True
            
            # Define vehicle types
            self._define_vehicle_types()
            
            print(f"✓ SUMO started: {self.config.sumo_config_file}")
            return True
            
        except Exception as e:
            print(f"✗ Error starting SUMO: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _define_vehicle_types(self):
        """Define delivery vehicle types in SUMO (skip if already loaded from XML)"""
        if self.vehicle_types_defined:
            return
        
        try:
            # Check if types already exist (loaded from vehicle_types.add.xml)
            existing_types = traci.vehicletype.getIDList()
            
            if "delivery_truck" in existing_types:
                print("✓ Vehicle types loaded from XML file")
                self.vehicle_types_defined = True
                return
            
            # Types not in XML, define them via TraCI
            if "car" in existing_types:
                base_type = "car"
            elif "DEFAULT_VEHTYPE" in existing_types:
                base_type = "DEFAULT_VEHTYPE"
            else:
                base_type = existing_types[0] if existing_types else "DEFAULT_VEHTYPE"
            
            # Delivery truck
            traci.vehicletype.copy(base_type, "delivery_truck")
            traci.vehicletype.setLength("delivery_truck", 8.0)
            traci.vehicletype.setMaxSpeed("delivery_truck", 15.0)
            traci.vehicletype.setColor("delivery_truck", (255, 100, 100, 255))
            
            # Delivery van
            traci.vehicletype.copy(base_type, "delivery_van")
            traci.vehicletype.setLength("delivery_van", 6.0)
            traci.vehicletype.setMaxSpeed("delivery_van", 18.0)
            traci.vehicletype.setColor("delivery_van", (100, 100, 255, 255))
            
            # Drone (uses car model but faster)
            traci.vehicletype.copy(base_type, "delivery_drone")
            traci.vehicletype.setLength("delivery_drone", 2.0)
            traci.vehicletype.setMaxSpeed("delivery_drone", 25.0)
            traci.vehicletype.setColor("delivery_drone", (100, 255, 100, 255))
            
            # Robot (slow)
            traci.vehicletype.copy(base_type, "delivery_robot")
            traci.vehicletype.setLength("delivery_robot", 1.5)
            traci.vehicletype.setMaxSpeed("delivery_robot", 8.0)
            traci.vehicletype.setColor("delivery_robot", (255, 255, 100, 255))
            
            self.vehicle_types_defined = True
            print("✓ Delivery vehicle types defined via TraCI")
            
        except Exception as e:
            # Types might already exist, that's fine
            self.vehicle_types_defined = True
            print("✓ Using existing vehicle types")
    
    def spawn_agent(self, agent: RLDeliveryAgentSUMO) -> bool:
        """Spawn an agent as a SUMO vehicle on a verified connected edge"""
        if not self.is_connected:
            return False
        
        try:
            start_edge = None
            
            # PRIORITY 1: Use spawn_edges from scenario (verified connected)
            if hasattr(self.zone_manager, 'spawn_edges') and self.zone_manager.spawn_edges:
                start_edge = random.choice(self.zone_manager.spawn_edges)
            
            # PRIORITY 2: Use connected_edges from scenario
            elif hasattr(self.zone_manager, 'connected_edges') and self.zone_manager.connected_edges:
                start_edge = random.choice(self.zone_manager.connected_edges[:10])
            
            # PRIORITY 3: Use depot edges
            elif self.zone_manager.depot_zones:
                depot = random.choice(self.zone_manager.depot_zones)
                if depot.get('edges'):
                    start_edge = depot['edges'][0]
            
            # PRIORITY 4: Find edge near depot (fallback)
            if not start_edge:
                for radius in [200, 400, 600]:
                    candidate = self.zone_manager.find_nearest_edge(agent.depot_position, radius)
                    if candidate:
                        start_edge = candidate
                        break
            
            if not start_edge:
                print(f"✗ No valid edge for spawning {agent.agent_id}")
                return False
            
            # Create route with unique ID - try to find a valid multi-edge route
            route_id = f"route_{agent.agent_id}_{random.randint(1000, 9999)}"
            route_edges = [start_edge]
            
            # Try to find connected edges to create a longer route
            if hasattr(self.zone_manager, 'connected_edges') and self.zone_manager.connected_edges:
                # Add more edges to the route if possible
                for other_edge in self.zone_manager.connected_edges[:5]:
                    if other_edge != start_edge:
                        route_edges.append(other_edge)
                        break
            
            try:
                traci.route.add(route_id, [start_edge])  # Start with single edge
            except Exception as e:
                # Try another connected edge
                if hasattr(self.zone_manager, 'connected_edges') and self.zone_manager.connected_edges:
                    start_edge = random.choice(self.zone_manager.connected_edges)
                    route_id = f"route_{agent.agent_id}_{random.randint(10000, 99999)}"
                    traci.route.add(route_id, [start_edge])
                else:
                    print(f"✗ Route creation failed for {agent.agent_id}: {e}")
                    return False
            
            # Map vehicle type
            vtype_map = {
                VehicleType.TRUCK: "delivery_truck",
                VehicleType.VAN: "delivery_van",
                VehicleType.DRONE: "delivery_drone",  # Use drone type if available
                VehicleType.ROBOT: "delivery_robot"   # Use robot type if available
            }
            sumo_vtype = vtype_map.get(agent.vehicle_type, "delivery_van")
            
            # Check if type exists, fallback to delivery_van
            try:
                existing_types = traci.vehicletype.getIDList()
                if sumo_vtype not in existing_types:
                    sumo_vtype = "delivery_van" if "delivery_van" in existing_types else existing_types[0]
            except:
                sumo_vtype = "delivery_van"
            
            # Spawn vehicle
            traci.vehicle.add(
                agent.sumo_vehicle_id,
                route_id,
                typeID=sumo_vtype,
                depart="now"
            )
            
            agent.is_spawned_in_sumo = True
            agent.current_edge = start_edge
            agent.stuck_counter = 0  # Initialize stuck counter
            
            # Update agent position to match spawn edge
            edge_pos = self.zone_manager.get_edge_position(start_edge)
            if edge_pos:
                agent.position = edge_pos
            
            print(f"✓ Spawned {agent.agent_id} ({agent.vehicle_type.value}) on edge {start_edge}")
            return True
            
        except Exception as e:
            print(f"✗ Error spawning {agent.agent_id}: {e}")
            return False
    
    def update_agent_position(self, agent: RLDeliveryAgentSUMO):
        """Update agent position from SUMO"""
        if not self.is_connected or not agent.is_spawned_in_sumo:
            return
        
        try:
            if agent.sumo_vehicle_id in traci.vehicle.getIDList():
                sumo_x, sumo_y = traci.vehicle.getPosition(agent.sumo_vehicle_id)
                new_pos = self.zone_manager.sumo_to_local(sumo_x, sumo_y)
                
                # Calculate distance moved
                dist_moved = agent.position.distance_to(new_pos)
                
                # Initialize stuck counter if not exists
                if not hasattr(agent, 'stuck_counter'):
                    agent.stuck_counter = 0
                
                # Check if vehicle is actually moving
                if dist_moved > 0.5:  # Vehicle moved significantly
                    # Update position
                    agent.position = new_pos
                    agent.stuck_counter = 0  # Reset stuck counter
                    
                    # Update battery based on distance
                    energy = dist_moved * agent.spec.battery_consumption_rate
                    agent.battery = max(0, agent.battery - energy)
                    agent.total_distance += dist_moved
                    agent.total_energy += energy
                    
                    # Update speed
                    agent.velocity = traci.vehicle.getSpeed(agent.sumo_vehicle_id)
                    
                    # Update edge
                    agent.current_edge = traci.vehicle.getRoadID(agent.sumo_vehicle_id)
                else:
                    # Vehicle might be stuck - increment counter
                    agent.stuck_counter += 1
                    
                    # Only mark as fallback after many stuck steps (give time to start)
                    if agent.stuck_counter > 50:  # 50 steps grace period
                        agent.is_spawned_in_sumo = False
                        print(f"  ⚠ {agent.agent_id} stuck in SUMO, using fallback movement")
            else:
                # Vehicle not in SUMO anymore
                agent.is_spawned_in_sumo = False
                    
        except Exception as e:
            # If we can't get position, use fallback movement
            agent.is_spawned_in_sumo = False
    
    def set_vehicle_target(self, agent: RLDeliveryAgentSUMO, target: Position):
        """Set vehicle destination in SUMO using verified connected edges"""
        if not self.is_connected or not agent.is_spawned_in_sumo:
            return False
        
        try:
            # Find target edge - prefer connected edges
            target_edge = None
            
            # First try to find nearest edge within connected edges
            candidate = self.zone_manager.find_nearest_edge(target, radius=200)
            
            # Verify it's in connected edges
            if candidate:
                if hasattr(self.zone_manager, 'connected_edges'):
                    if candidate in self.zone_manager.connected_edges:
                        target_edge = candidate
                    else:
                        # Try to find a connected edge near the target
                        for conn_edge in self.zone_manager.connected_edges[:20]:
                            target_edge = conn_edge
                            break
                else:
                    target_edge = candidate
            
            # If not found, use any connected edge
            if not target_edge and hasattr(self.zone_manager, 'connected_edges'):
                if self.zone_manager.connected_edges:
                    target_edge = random.choice(self.zone_manager.connected_edges)
            
            if target_edge and agent.sumo_vehicle_id in traci.vehicle.getIDList():
                current_edge = traci.vehicle.getRoadID(agent.sumo_vehicle_id)
                
                # Skip internal edges (start with :)
                if current_edge.startswith(':'):
                    return True
                
                # Don't set same target
                if current_edge == target_edge:
                    return True
                
                try:
                    traci.vehicle.changeTarget(agent.sumo_vehicle_id, target_edge)
                    agent.target_edge = target_edge
                    return True
                except Exception as route_error:
                    # Routing failed - this edge pair can't be routed
                    # Try a different target edge from connected edges
                    if hasattr(self.zone_manager, 'connected_edges'):
                        for alt_edge in self.zone_manager.connected_edges[:10]:
                            if alt_edge != current_edge:
                                try:
                                    traci.vehicle.changeTarget(agent.sumo_vehicle_id, alt_edge)
                                    agent.target_edge = alt_edge
                                    return True
                                except:
                                    continue
                    return False
                
        except Exception as e:
            pass
        
        return False
    
    def step(self):
        """Advance SUMO simulation by one step"""
        if self.is_connected:
            try:
                traci.simulationStep()
                self.step_count += 1
            except:
                pass
    
    def get_simulation_time(self) -> float:
        """Get current SUMO simulation time"""
        if self.is_connected:
            try:
                return traci.simulation.getTime()
            except:
                pass
        return 0.0
    
    def close(self):
        """Close SUMO connection"""
        if self.is_connected:
            try:
                traci.close()
                print("✓ SUMO closed")
            except Exception as e:
                print(f"⚠ Warning closing SUMO: {e}")
            finally:
                self.is_connected = False
                self.vehicle_types_defined = False
        
        # Small delay to ensure SUMO fully closes
        import time
        time.sleep(0.5)


# ==================== INTEGRATED ENVIRONMENT ====================

class MultiAgentDeliveryEnvSUMO(gym.Env):
    """
    Multi-agent delivery environment integrated with SUMO.
    
    This combines your RL environment with real SUMO traffic simulation.
    Agents drive on actual roads in the Tunis map!
    """
    
    def __init__(self, config: SimConfig = None, n_agents: int = 5, use_sumo: bool = True):
        super().__init__()
        self.config = config or SimConfig()
        self.n_agents = n_agents
        self.use_sumo = use_sumo and SUMO_AVAILABLE
        
        # Spaces (per agent)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(58,), dtype=np.float32
        )
        self.action_space = RLDeliveryAgentSUMO.get_action_space()
        
        # Zone manager
        self.zone_manager = SUMOZoneManager(self.config)
        
        # SUMO controller
        self.sumo_controller = SUMOController(self.config, self.zone_manager) if self.use_sumo else None
        
        # Environment state
        self.agents: List[RLDeliveryAgentSUMO] = []
        self.tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.cancelled_tasks: List[Task] = []
        
        self.current_time = 0.0
        self.step_count = 0
        self.task_counter = 0
        
        # Logging
        self.step_logs: List[Dict] = []
    
    def reset(self, seed=None):
        """Reset environment"""
        super().reset(seed=seed)
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        print("\n" + "="*70)
        print("RESETTING ENVIRONMENT")
        print("="*70)
        
        # Close existing SUMO connection first
        if self.sumo_controller and self.sumo_controller.is_connected:
            self.sumo_controller.close()
        
        self.current_time = 0.0
        self.step_count = 0
        self.task_counter = 0
        self.tasks = []
        self.completed_tasks = []
        self.cancelled_tasks = []
        self.step_logs = []
        
        # Load zones
        self.zone_manager.load_network()
        self.zone_manager.load_zones()
        
        # Start SUMO (fresh connection)
        if self.use_sumo and self.sumo_controller:
            self.sumo_controller.start()
        
        # Create agents
        self._create_agents()
        
        # Spawn agents in SUMO
        if self.use_sumo and self.sumo_controller and self.sumo_controller.is_connected:
            for agent in self.agents:
                self.sumo_controller.spawn_agent(agent)
        
        # Initial observations
        observations = [self._get_agent_observation(agent) for agent in self.agents]
        
        print(f"\n✓ Environment reset complete")
        print(f"  Agents: {len(self.agents)}")
        print(f"  SUMO connected: {self.sumo_controller.is_connected if self.sumo_controller else False}")
        print("="*70 + "\n")
        
        return observations, {}
    
    def _create_agents(self):
        """Create delivery agents"""
        self.agents = []
        
        vehicle_types = [
            VehicleType.TRUCK, 
            VehicleType.VAN, 
            VehicleType.DRONE, 
            VehicleType.ROBOT, 
            VehicleType.VAN
        ]
        
        for i in range(self.n_agents):
            # Get depot position
            depot_pos, depot_edge = self.zone_manager.get_random_zone_position('depot')
            
            # Slight offset from depot
            initial_pos = Position(
                depot_pos.x + random.uniform(-20, 20),
                depot_pos.y + random.uniform(-20, 20)
            )
            
            vtype = vehicle_types[i % len(vehicle_types)]
            
            agent = RLDeliveryAgentSUMO(
                agent_id=f"AGENT_{i:02d}",
                vehicle_type=vtype,
                initial_position=initial_pos,
                depot_position=depot_pos,
                zone_manager=self.zone_manager
            )
            
            self.agents.append(agent)
            
            print(f"  Created {agent.agent_id}: {vtype.value} at {initial_pos}")
    
    def _spawn_task(self):
        """Spawn a new delivery task"""
        task_type = random.choice(['depot_to_customer', 'restaurant_to_customer', 'depot_to_boutique'])
        
        if task_type == 'depot_to_customer':
            pickup_pos, pickup_edge = self.zone_manager.get_random_zone_position('depot')
            delivery_pos, delivery_edge = self.zone_manager.get_random_zone_position('delivery')
        elif task_type == 'restaurant_to_customer':
            pickup_pos, pickup_edge = self.zone_manager.get_random_zone_position('pickup')
            delivery_pos, delivery_edge = self.zone_manager.get_random_zone_position('delivery')
        else:
            pickup_pos, pickup_edge = self.zone_manager.get_random_zone_position('depot')
            delivery_pos, delivery_edge = self.zone_manager.get_random_zone_position('delivery')
        
        # Task properties
        priority = random.choice([1, 3, 3, 5])
        
        # Weight distribution matching vehicle types
        weight_roll = random.random()
        if weight_roll < 0.30:
            weight = random.uniform(0.5, 4.5)  # Drone-capable
        elif weight_roll < 0.55:
            weight = random.uniform(4.5, 25.0)  # Robot-capable
        elif weight_roll < 0.85:
            weight = random.uniform(25.0, 150.0)  # Van-capable
        else:
            weight = random.uniform(150.0, 400.0)  # Truck only
        
        volume = weight * 0.01
        deadline = self.current_time + random.uniform(300, 1800)
        
        # Allowed vehicles based on weight
        allowed = []
        if weight <= 5:
            allowed.append(VehicleType.DRONE)
        if weight <= 30:
            allowed.append(VehicleType.ROBOT)
        if weight <= 200:
            allowed.append(VehicleType.VAN)
        allowed.append(VehicleType.TRUCK)
        
        task = Task(
            id=f"TASK_{self.task_counter:04d}",
            pickup_loc=pickup_pos,
            delivery_loc=delivery_pos,
            created_time=self.current_time,
            deadline=deadline,
            priority=priority,
            weight=weight,
            volume=volume,
            fragile=random.random() < 0.15,
            temperature_sensitive=random.random() < 0.1,
            allowed_vehicle_types=allowed,
            pickup_edge=pickup_edge,
            delivery_edge=delivery_edge
        )
        
        self.tasks.append(task)
        self.task_counter += 1
        
        if self.config.verbose:
            print(f"  [NEW TASK] {task.id}: {weight:.1f}kg, priority={priority}, "
                  f"deadline={deadline-self.current_time:.0f}s")
    
    def step(self, actions: List[int]):
        """Execute one timestep"""
        dt = self.config.timestep
        self.current_time += dt
        self.step_count += 1
        
        # Advance SUMO
        if self.use_sumo and self.sumo_controller:
            self.sumo_controller.step()
            
            # Update agent positions from SUMO
            for agent in self.agents:
                old_pos = Position(agent.position.x, agent.position.y)
                self.sumo_controller.update_agent_position(agent)
                
                # Check if agent is stuck (hasn't moved in SUMO)
                if agent.is_spawned_in_sumo:
                    dist_moved = old_pos.distance_to(agent.position)
                    if dist_moved < 0.5 and agent.current_task:
                        # Agent should be moving but isn't - increment stuck counter
                        if not hasattr(agent, 'stuck_counter'):
                            agent.stuck_counter = 0
                        agent.stuck_counter += 1
                        
                        # If stuck for too long, switch to fallback movement
                        if agent.stuck_counter > 30:  # Stuck for 30 steps
                            agent.is_spawned_in_sumo = False
                            agent.stuck_counter = 0
                            if self.config.verbose and self.step_count % 100 == 0:
                                print(f"  ⚠ {agent.agent_id} stuck in SUMO, using fallback movement")
                    else:
                        agent.stuck_counter = 0
        
        # Spawn new tasks
        if random.random() < self.config.task_spawn_rate * dt:
            self._spawn_task()
        
        # Random task cancellation
        for task in list(self.tasks):
            if (task.status == TaskStatus.PENDING and 
                random.random() < self.config.task_cancel_prob * dt):
                task.status = TaskStatus.CANCELLED
                self.tasks.remove(task)
                self.cancelled_tasks.append(task)
        
        # Update 5G availability
        for agent in self.agents:
            agent.v5g_available = random.random() < self.config.v5g_available_prob
        
        # Execute actions
        rewards = []
        dones = []
        infos = []
        step_log = {'step': self.step_count, 'time': self.current_time, 'agents': []}
        
        for i, agent in enumerate(self.agents):
            visible_tasks = self._get_visible_tasks(agent)
            nearby_agents = self._get_nearby_agents(agent)
            
            # Execute action
            action_result = agent.execute_action(
                actions[i], visible_tasks, nearby_agents,
                self.current_time, dt, self.sumo_controller
            )
            
            # Calculate reward
            reward = agent.calculate_step_reward(action_result, self.current_time)
            rewards.append(reward)
            
            # Log action
            action_name = agent.get_action_name(actions[i])
            log_entry = agent.log_action(self.step_count, actions[i], action_name, 
                                        action_result, self.current_time)
            step_log['agents'].append(log_entry)
            
            # Check done
            done = (agent.battery <= 0 or 
                   self.current_time >= self.config.sim_duration)
            dones.append(done)
            
            infos.append({
                'agent_id': agent.agent_id,
                'completed': agent.completed_count,
                'on_time_rate': (agent.on_time_count / agent.completed_count 
                               if agent.completed_count > 0 else 0),
                'state': agent.state.value
            })
        
        # Move completed tasks
        for task in list(self.tasks):
            if task.status == TaskStatus.COMPLETED:
                self.tasks.remove(task)
                self.completed_tasks.append(task)
        
        # Store step log
        self.step_logs.append(step_log)
        
        # Verbose logging
        if self.config.verbose and self.step_count % self.config.log_every_n_steps == 0:
            self._print_step_summary()
        
        # Get new observations
        observations = [self._get_agent_observation(agent) for agent in self.agents]
        
        # Global done
        done = all(dones) or self.current_time >= self.config.sim_duration
        
        return observations, rewards, done, False, infos
    
    def _print_step_summary(self):
        """Print step-by-step summary"""
        metrics = self.get_metrics()
        
        print(f"\n[Step {self.step_count}] Time: {self.current_time:.1f}s")
        print(f"  Completed: {metrics['total_completed']} | "
              f"On-Time: {metrics['on_time_rate']:.1%} | "
              f"Pending: {metrics['total_pending']}")
        
        for agent in self.agents:
            task_info = f"Task: {agent.current_task.id}" if agent.current_task else "No task"
            print(f"  {agent.agent_id} ({agent.vehicle_type.value}): "
                  f"{agent.state.value} | "
                  f"Pos: {agent.position} | "
                  f"Battery: {agent.battery:.1f}kWh ({agent.battery/agent.spec.max_battery_kwh*100:.0f}%) | "
                  f"{task_info}")
    
    def _get_visible_tasks(self, agent: RLDeliveryAgentSUMO) -> List[Task]:
        """Get tasks visible to agent"""
        visible = []
        for task in self.tasks:
            if task.status in [TaskStatus.PENDING, TaskStatus.ANNOUNCED]:
                dist = agent.position.distance_to(task.pickup_loc)
                if dist <= agent.spec.comm_range:
                    visible.append(task)
            elif task.assigned_to == agent.agent_id:
                visible.append(task)
        return visible
    
    def _get_nearby_agents(self, agent: RLDeliveryAgentSUMO) -> List[RLDeliveryAgentSUMO]:
        """Get agents within comm range"""
        nearby = []
        for other in self.agents:
            if other.agent_id != agent.agent_id:
                dist = agent.position.distance_to(other.position)
                if dist <= agent.spec.comm_range:
                    nearby.append(other)
        return nearby
    
    def _get_agent_observation(self, agent: RLDeliveryAgentSUMO) -> np.ndarray:
        """Get observation for specific agent"""
        visible_tasks = self._get_visible_tasks(agent)
        nearby_agents = self._get_nearby_agents(agent)
        return agent.get_observation(self.current_time, visible_tasks, nearby_agents)
    
    def get_metrics(self) -> Dict:
        """Calculate system-wide KPIs"""
        total_completed = len(self.completed_tasks)
        on_time = sum(1 for t in self.completed_tasks if t.delivery_time and t.delivery_time <= t.deadline)
        
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
    
    def export_logs(self, filename: str = "simulation_log.json"):
        """Export step-by-step logs for visualization"""
        export_data = {
            'config': {
                'city_size': self.config.city_size,
                'sim_duration': self.config.sim_duration,
                'n_agents': self.n_agents
            },
            'final_metrics': self.get_metrics(),
            'steps': self.step_logs,
            'agents_summary': [
                {
                    'id': a.agent_id,
                    'type': a.vehicle_type.value,
                    'completed': a.completed_count,
                    'on_time': a.on_time_count,
                    'total_distance_km': a.total_distance / 1000,
                    'total_energy_kwh': a.total_energy
                }
                for a in self.agents
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"✓ Logs exported to {filename}")
        return filename
    
    def close(self):
        """Clean up"""
        if self.sumo_controller:
            self.sumo_controller.close()


# ==================== TRAINING ====================

class RLTrainerSUMO:
    """Training loop for SUMO-integrated environment"""
    
    def __init__(self, env: MultiAgentDeliveryEnvSUMO):
        self.env = env
        self.episode_rewards = []
        self.episode_metrics = []
    
    def train_greedy(self, n_episodes: int = 5):
        """Train with greedy heuristic"""
        print("\n" + "="*70)
        print("TRAINING GREEDY HEURISTIC (SUMO INTEGRATED)")
        print("="*70)
        
        for episode in range(n_episodes):
            print(f"\n{'='*70}")
            print(f"EPISODE {episode + 1}/{n_episodes}")
            print(f"{'='*70}")
            
            observations, _ = self.env.reset(seed=episode)
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < 1000:
                actions = []
                
                for agent in self.env.agents:
                    action = self._greedy_action(agent)
                    actions.append(action)
                
                observations, rewards, done, truncated, infos = self.env.step(actions)
                episode_reward += sum(rewards)
                step += 1
            
            # Episode summary
            metrics = self.env.get_metrics()
            self.episode_rewards.append(episode_reward)
            self.episode_metrics.append(metrics)
            
            print(f"\n{'='*70}")
            print(f"EPISODE {episode + 1} COMPLETE")
            print(f"{'='*70}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Completed Deliveries: {metrics['total_completed']}")
            print(f"  On-Time Rate: {metrics['on_time_rate']:.2%}")
            print(f"  Energy/Delivery: {metrics['energy_per_delivery']:.3f} kWh")
            print(f"  Distance: {metrics['total_distance_km']:.2f} km")
            
            # Export logs
            self.env.export_logs(f"episode_{episode+1}_log.json")
        
        self._print_summary()
        self.env.close()
    
    def _greedy_action(self, agent: RLDeliveryAgentSUMO) -> int:
        """Greedy action selection"""
        # Battery critical - go to depot
        battery_percent = agent.battery / agent.spec.max_battery_kwh
        if battery_percent < 0.30 or agent.is_returning_to_depot:
            return 9  # GO_TO_DEPOT
        
        # Has task - continue
        if agent.current_task:
            return 14  # CONTINUE
        
        # Try to accept task
        visible_tasks = self.env._get_visible_tasks(agent)
        acceptable = [t for t in visible_tasks if agent.can_accept_task(t)]
        
        if acceptable:
            # Score tasks
            def score(task):
                dist = agent.position.distance_to(task.pickup_loc)
                return dist / (task.priority + 1)
            
            best = min(acceptable, key=score)
            idx = visible_tasks.index(best)
            return min(idx, 4)
        
        # Explore
        return random.choice([5, 6, 7, 8])
    
    def _print_summary(self):
        """Print training summary"""
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        
        avg_reward = np.mean(self.episode_rewards)
        avg_completed = np.mean([m['total_completed'] for m in self.episode_metrics])
        avg_on_time = np.mean([m['on_time_rate'] for m in self.episode_metrics])
        
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Completed: {avg_completed:.1f}")
        print(f"  Average On-Time Rate: {avg_on_time:.2%}")
        print("="*70)


# ==================== MAIN ====================

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("SMARTFLEET: SUMO-INTEGRATED MULTI-AGENT DELIVERY SYSTEM")
    print("="*70)
    
    # Configuration - Using verified smartfleet scenario
    config = SimConfig(
        sumo_config_file="smartfleet.sumocfg",
        sumo_net_file="finished_map_cleaning.net.xml",
        sumo_add_file="smartfleet_zones.add.xml",
        scenario_file="smartfleet_scenario.json",
        use_sumo_gui=True,
        sim_duration=600,
        task_spawn_rate=0.2,
        verbose=True,
        log_every_n_steps=50
    )
    
    # Create environment
    env = MultiAgentDeliveryEnvSUMO(config=config, n_agents=5, use_sumo=True)
    
    # Train
    trainer = RLTrainerSUMO(env)
    trainer.train_greedy(n_episodes=2)
    
    print("\n✓ Training complete!")
    print("  Check episode_*_log.json files for step-by-step visualization data")


if __name__ == "__main__":
    main()