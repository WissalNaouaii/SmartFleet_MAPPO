"""
================================================================================
SMARTFLEET - CREATE CUSTOM DELIVERY SCENARIO
================================================================================
This script helps you create a working delivery scenario by:
1. Finding the best-connected area in your network
2. Automatically selecting depot, pickup, and delivery locations
3. Testing that all routes work
4. Generating all necessary files

No SUMO expertise needed - just run this script!

================================================================================
"""

import xml.etree.ElementTree as ET
from collections import defaultdict, deque
import random
import json
import os

# ============ CONFIGURATION ============
NET_FILE = "finished_map_cleaning.net.xml"
SUMO_HOME = os.environ.get('SUMO_HOME', r'C:\Program Files\sumo-1.25.0')

# Scenario parameters
NUM_DEPOTS = 2
NUM_PICKUP_ZONES = 5
NUM_DELIVERY_ZONES = 10

print("=" * 70)
print("SMARTFLEET - CREATE CUSTOM DELIVERY SCENARIO")
print("=" * 70)

# ==================== STEP 1: LOAD NETWORK ====================

print("\n[STEP 1] Loading road network...")

tree = ET.parse(NET_FILE)
root = tree.getroot()

# Get network bounds
location = root.find('location')
if location is not None:
    net_offset = location.get('netOffset', '0,0').split(',')
    conv_boundary = location.get('convBoundary', '0,0,1000,1000').split(',')
    print(f"  Network bounds: {conv_boundary}")

# Extract all edges with their properties
edges = {}
edge_lanes = {}
junctions = {}

for junction in root.findall('.//junction'):
    junc_id = junction.get('id')
    if junc_id and not junc_id.startswith(':'):
        x = float(junction.get('x', 0))
        y = float(junction.get('y', 0))
        junctions[junc_id] = {'x': x, 'y': y}

for edge in root.findall('.//edge'):
    edge_id = edge.get('id')
    
    # Skip internal edges
    if not edge_id or edge_id.startswith(':'):
        continue
    
    from_node = edge.get('from')
    to_node = edge.get('to')
    function = edge.get('function', '')
    
    if function == 'internal':
        continue
    
    # Get lane info
    lanes = edge.findall('lane')
    if not lanes:
        continue
    
    lane = lanes[0]
    length = float(lane.get('length', 0))
    speed = float(lane.get('speed', 13.89))
    
    # Check if allows vehicles
    allow = lane.get('allow', '')
    disallow = lane.get('disallow', '')
    
    # Skip pedestrian-only or rail
    if 'pedestrian' in allow and 'passenger' not in allow:
        continue
    if 'rail' in allow:
        continue
    if 'passenger' in disallow:
        continue
    
    # Need minimum length
    if length < 10:
        continue
    
    # Get position from junction
    if from_node in junctions:
        pos = junctions[from_node]
    elif to_node in junctions:
        pos = junctions[to_node]
    else:
        continue
    
    edges[edge_id] = {
        'from': from_node,
        'to': to_node,
        'length': length,
        'speed': speed,
        'x': pos['x'],
        'y': pos['y']
    }

print(f"  Found {len(edges)} usable road edges")

# ==================== STEP 2: BUILD CONNECTION GRAPH ====================

print("\n[STEP 2] Building road connection graph...")

# Forward connections (from -> to)
forward_connections = defaultdict(set)
# Backward connections (to -> from) for bidirectional search
backward_connections = defaultdict(set)

for conn in root.findall('.//connection'):
    from_edge = conn.get('from')
    to_edge = conn.get('to')
    
    if from_edge in edges and to_edge in edges:
        forward_connections[from_edge].add(to_edge)
        backward_connections[to_edge].add(from_edge)

total_connections = sum(len(v) for v in forward_connections.values())
print(f"  Found {total_connections} road connections")

# ==================== STEP 3: FIND LARGEST STRONGLY CONNECTED COMPONENT ====================

print("\n[STEP 3] Finding best connected road cluster...")

def bfs_reachable(start, connections):
    """Find all edges reachable from start"""
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in connections.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
    return visited

def find_strongly_connected(edges, forward, backward):
    """Find edges that can reach each other (both directions)"""
    best_cluster = set()
    tested = set()
    
    for start_edge in edges:
        if start_edge in tested:
            continue
        
        # Find edges reachable FROM this edge
        reachable_forward = bfs_reachable(start_edge, forward)
        
        # Find edges that can REACH this edge
        reachable_backward = bfs_reachable(start_edge, backward)
        
        # Strongly connected = intersection (can go both ways)
        strongly_connected = reachable_forward & reachable_backward
        
        if len(strongly_connected) > len(best_cluster):
            best_cluster = strongly_connected
        
        tested.update(strongly_connected)
    
    return best_cluster

# Find the largest strongly connected component
best_cluster = find_strongly_connected(edges.keys(), forward_connections, backward_connections)

print(f"  Best connected cluster: {len(best_cluster)} edges ({100*len(best_cluster)/len(edges):.1f}%)")

if len(best_cluster) < 20:
    print("\n  ⚠ Warning: Small cluster found. Using all forward-reachable edges instead...")
    # Fall back to largest weakly connected component
    all_edges = set(edges.keys())
    unvisited = all_edges.copy()
    components = []
    
    while unvisited:
        start = next(iter(unvisited))
        component = bfs_reachable(start, forward_connections)
        components.append(component)
        unvisited -= component
    
    components.sort(key=len, reverse=True)
    best_cluster = components[0]
    print(f"  Using largest component: {len(best_cluster)} edges")

# Filter edges to only include best cluster
connected_edges = {eid: edges[eid] for eid in best_cluster if eid in edges}

print(f"\n  ✓ Selected {len(connected_edges)} connected edges for scenario")

# ==================== STEP 4: SELECT STRATEGIC LOCATIONS ====================

print("\n[STEP 4] Selecting strategic locations...")

# Get bounds of connected area
x_coords = [e['x'] for e in connected_edges.values()]
y_coords = [e['y'] for e in connected_edges.values()]

min_x, max_x = min(x_coords), max(x_coords)
min_y, max_y = min(y_coords), max(y_coords)
center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2

print(f"  Area bounds: X=[{min_x:.0f}, {max_x:.0f}], Y=[{min_y:.0f}, {max_y:.0f}]")
print(f"  Center: ({center_x:.0f}, {center_y:.0f})")

# Score edges by connectivity (more connections = better for depot/pickup)
edge_scores = {}
for eid in connected_edges:
    forward = len(forward_connections.get(eid, []))
    backward = len(backward_connections.get(eid, []))
    edge_scores[eid] = forward + backward

# Sort by score
sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)

# Select DEPOTS - most connected edges near center
print(f"\n  Selecting {NUM_DEPOTS} depot locations...")
depots = []
for eid, score in sorted_edges:
    if len(depots) >= NUM_DEPOTS:
        break
    edge = connected_edges[eid]
    # Prefer edges near center
    dist_to_center = ((edge['x'] - center_x)**2 + (edge['y'] - center_y)**2) ** 0.5
    if dist_to_center < (max_x - min_x) * 0.4:  # Within 40% of center
        depots.append({
            'id': f'DEPOT_{len(depots)+1}',
            'edge': eid,
            'x': edge['x'],
            'y': edge['y'],
            'score': score
        })
        print(f"    {depots[-1]['id']}: edge {eid} at ({edge['x']:.0f}, {edge['y']:.0f})")

# Select PICKUP zones - well-connected, spread out
print(f"\n  Selecting {NUM_PICKUP_ZONES} pickup locations...")
pickups = []
used_edges = set(d['edge'] for d in depots)

for eid, score in sorted_edges:
    if len(pickups) >= NUM_PICKUP_ZONES:
        break
    if eid in used_edges:
        continue
    
    edge = connected_edges[eid]
    
    # Check distance from existing pickups (want spread out)
    too_close = False
    for p in pickups:
        dist = ((edge['x'] - p['x'])**2 + (edge['y'] - p['y'])**2) ** 0.5
        if dist < (max_x - min_x) * 0.15:  # Min 15% apart
            too_close = True
            break
    
    if not too_close:
        pickups.append({
            'id': f'PICKUP_{len(pickups)+1}',
            'edge': eid,
            'x': edge['x'],
            'y': edge['y'],
            'score': score
        })
        used_edges.add(eid)
        print(f"    {pickups[-1]['id']}: edge {eid} at ({edge['x']:.0f}, {edge['y']:.0f})")

# Select DELIVERY zones - spread throughout the area
print(f"\n  Selecting {NUM_DELIVERY_ZONES} delivery locations...")
deliveries = []

# Divide area into grid and pick one edge per cell
grid_size = 3
cell_width = (max_x - min_x) / grid_size
cell_height = (max_y - min_y) / grid_size

for eid in connected_edges:
    if len(deliveries) >= NUM_DELIVERY_ZONES:
        break
    if eid in used_edges:
        continue
    
    edge = connected_edges[eid]
    
    # Check minimum distance from other deliveries
    too_close = False
    for d in deliveries:
        dist = ((edge['x'] - d['x'])**2 + (edge['y'] - d['y'])**2) ** 0.5
        if dist < (max_x - min_x) * 0.1:  # Min 10% apart
            too_close = True
            break
    
    if not too_close and edge_scores.get(eid, 0) >= 1:  # At least 1 connection
        deliveries.append({
            'id': f'DELIVERY_{len(deliveries)+1}',
            'edge': eid,
            'x': edge['x'],
            'y': edge['y'],
            'score': edge_scores.get(eid, 0)
        })
        used_edges.add(eid)

print(f"    Selected {len(deliveries)} delivery locations")

# ==================== STEP 5: VERIFY ROUTES ====================

print("\n[STEP 5] Verifying all routes work...")

def can_route(from_edge, to_edge, connections, max_depth=50):
    """Check if there's a path from from_edge to to_edge"""
    if from_edge == to_edge:
        return True
    
    visited = set()
    queue = deque([(from_edge, 0)])
    
    while queue:
        edge, depth = queue.popleft()
        if depth > max_depth:
            continue
        if edge in visited:
            continue
        visited.add(edge)
        
        for next_edge in connections.get(edge, []):
            if next_edge == to_edge:
                return True
            if next_edge not in visited:
                queue.append((next_edge, depth + 1))
    
    return False

# Test routes from depots to all other locations
route_tests = 0
route_success = 0

for depot in depots:
    for pickup in pickups:
        route_tests += 1
        if can_route(depot['edge'], pickup['edge'], forward_connections):
            route_success += 1
        
        route_tests += 1
        if can_route(pickup['edge'], depot['edge'], forward_connections):
            route_success += 1
    
    for delivery in deliveries:
        route_tests += 1
        if can_route(depot['edge'], delivery['edge'], forward_connections):
            route_success += 1
        
        route_tests += 1
        if can_route(delivery['edge'], depot['edge'], forward_connections):
            route_success += 1

for pickup in pickups:
    for delivery in deliveries:
        route_tests += 1
        if can_route(pickup['edge'], delivery['edge'], forward_connections):
            route_success += 1

print(f"  Route tests: {route_success}/{route_tests} successful ({100*route_success/route_tests:.1f}%)")

if route_success / route_tests < 0.8:
    print("\n  ⚠ Some routes may fail. Filtering to only verified locations...")
    # Keep only locations that can route to/from depots
    verified_pickups = []
    for p in pickups:
        can_reach = any(can_route(d['edge'], p['edge'], forward_connections) for d in depots)
        can_return = any(can_route(p['edge'], d['edge'], forward_connections) for d in depots)
        if can_reach and can_return:
            verified_pickups.append(p)
    pickups = verified_pickups
    
    verified_deliveries = []
    for d in deliveries:
        can_reach = any(can_route(dep['edge'], d['edge'], forward_connections) for dep in depots)
        can_return = any(can_route(d['edge'], dep['edge'], forward_connections) for dep in depots)
        if can_reach and can_return:
            verified_deliveries.append(d)
    deliveries = verified_deliveries
    
    print(f"  Verified pickups: {len(pickups)}")
    print(f"  Verified deliveries: {len(deliveries)}")

# ==================== STEP 6: CREATE TAZ FILE ====================

print("\n[STEP 6] Creating TAZ zones file...")

add_root = ET.Element('additional')

# Add depot zones (blue)
for depot in depots:
    taz = ET.SubElement(add_root, 'taz')
    taz.set('id', depot['id'])
    taz.set('color', '98,189,239')  # Blue
    
    source = ET.SubElement(taz, 'tazSource')
    source.set('id', depot['edge'])
    source.set('weight', '1.0')
    
    sink = ET.SubElement(taz, 'tazSink')
    sink.set('id', depot['edge'])
    sink.set('weight', '1.0')

# Add pickup zones (purple)
for pickup in pickups:
    taz = ET.SubElement(add_root, 'taz')
    taz.set('id', pickup['id'])
    taz.set('color', '226,39,246')  # Purple
    
    source = ET.SubElement(taz, 'tazSource')
    source.set('id', pickup['edge'])
    source.set('weight', '1.0')
    
    sink = ET.SubElement(taz, 'tazSink')
    sink.set('id', pickup['edge'])
    sink.set('weight', '1.0')

# Add delivery zones (green)
for delivery in deliveries:
    taz = ET.SubElement(add_root, 'taz')
    taz.set('id', delivery['id'])
    taz.set('color', '98,246,40')  # Green
    
    source = ET.SubElement(taz, 'tazSource')
    source.set('id', delivery['edge'])
    source.set('weight', '1.0')
    
    sink = ET.SubElement(taz, 'tazSink')
    sink.set('id', delivery['edge'])
    sink.set('weight', '1.0')

output_add = "smartfleet_zones.add.xml"
add_tree = ET.ElementTree(add_root)
add_tree.write(output_add, encoding='UTF-8', xml_declaration=True)
print(f"  ✓ Saved {output_add}")

# ==================== STEP 7: CREATE SUMOCFG ====================

print("\n[STEP 7] Creating SUMO configuration...")

output_cfg = "smartfleet.sumocfg"

cfg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{NET_FILE}"/>
        <additional-files value="{output_add}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="7200"/>
        <step-length value="1.0"/>
    </time>
    <processing>
        <time-to-teleport value="300"/>
        <ignore-route-errors value="true"/>
    </processing>
    <report>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
</configuration>
'''

with open(output_cfg, 'w') as f:
    f.write(cfg_content)
print(f"  ✓ Saved {output_cfg}")

# ==================== STEP 8: SAVE SCENARIO CONFIG ====================

print("\n[STEP 8] Saving scenario configuration...")

scenario_config = {
    'network_file': NET_FILE,
    'additional_file': output_add,
    'config_file': output_cfg,
    'bounds': {
        'min_x': min_x, 'max_x': max_x,
        'min_y': min_y, 'max_y': max_y,
        'center_x': center_x, 'center_y': center_y
    },
    'depots': depots,
    'pickups': pickups,
    'deliveries': deliveries,
    'connected_edges': list(connected_edges.keys()),
    'spawn_edges': [d['edge'] for d in depots]  # Best edges for spawning agents
}

with open('smartfleet_scenario.json', 'w') as f:
    json.dump(scenario_config, f, indent=2)
print(f"  ✓ Saved smartfleet_scenario.json")

# ==================== SUMMARY ====================

print("\n" + "=" * 70)
print("SCENARIO CREATED SUCCESSFULLY!")
print("=" * 70)

print(f"""
📍 LOCATIONS:
   • Depots: {len(depots)} (agent starting/charging points)
   • Pickups: {len(pickups)} (restaurants, warehouses)
   • Deliveries: {len(deliveries)} (customer locations)
   • Connected roads: {len(connected_edges)}

📁 FILES CREATED:
   • {output_add} - Zone definitions for SUMO
   • {output_cfg} - SUMO configuration
   • smartfleet_scenario.json - Full scenario data

🧪 TO TEST:
   sumo-gui -c {output_cfg}

📍 DEPOT LOCATIONS:""")

for d in depots:
    print(f"   • {d['id']}: ({d['x']:.0f}, {d['y']:.0f}) - edge {d['edge']}")

print(f"""
🚀 NEXT STEP:
   Run the delivery simulation with these verified locations!
""")

print("=" * 70)
