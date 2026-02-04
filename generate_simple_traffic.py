"""
FINAL WORKING Traffic Generator for Tunis SUMO Map
- Checks edge connectivity
- Filters out restricted edges (pedestrian, bus-only, etc.)
- Sorts vehicles by departure time
- Validates vehicle types can use the edges
"""
import xml.etree.ElementTree as ET
import random
from collections import defaultdict

# Configuration
NET_FILE = "finished_map_cleaning.net.xml"
OUTPUT_FILE = "tunis_final_traffic.rou.xml"
NUM_VEHICLES = 80  # Reduced to avoid restrictions
SIM_DURATION = 3600

print("="*70)
print("GENERATING FINAL VALIDATED TUNIS TRAFFIC")
print("="*70)

# Parse network
print(f"\n✓ Reading network from {NET_FILE}...")
tree = ET.parse(NET_FILE)
root = tree.getroot()

# Get edges and check which vehicle types are allowed
print("✓ Analyzing edges and lane restrictions...")
valid_edges = []
edge_connections = defaultdict(list)

for edge in root.findall('.//edge'):
    edge_id = edge.get('id')
    
    # Skip internal edges
    if not edge_id or edge_id.startswith(':'):
        continue
    
    # Check lanes
    lanes = edge.findall('lane')
    if not lanes:
        continue
    
    # Check if edge allows passenger vehicles (not pedestrian/bicycle only)
    edge_allows_cars = False
    for lane in lanes:
        allowed = lane.get('allow', '')
        disallowed = lane.get('disallow', '')
        
        # If 'allow' is specified, check if passenger vehicles are allowed
        if allowed:
            if 'passenger' in allowed or 'private' in allowed or not allowed:
                edge_allows_cars = True
        # If 'disallow' is specified, check if passenger vehicles are NOT disallowed
        elif disallowed:
            if 'passenger' not in disallowed and 'private' not in disallowed:
                edge_allows_cars = True
        else:
            # No restrictions = allowed
            edge_allows_cars = True
        
        if edge_allows_cars:
            break
    
    # Only use edges that allow cars/vans
    if edge_allows_cars:
        # Check edge is long enough (at least 15m)
        length = float(lanes[0].get('length', 0))
        if length >= 15.0:
            valid_edges.append(edge_id)

print(f"✓ Found {len(valid_edges)} vehicle-accessible edges")

# Build connectivity map
print("✓ Building road connectivity map...")
for connection in root.findall('.//connection'):
    from_edge = connection.get('from')
    to_edge = connection.get('to')
    
    # Only include connections between valid edges
    if (from_edge in valid_edges and to_edge in valid_edges):
        edge_connections[from_edge].append(to_edge)

# Get edges with outgoing connections
start_edges = [edge for edge in valid_edges if len(edge_connections.get(edge, [])) > 0]
print(f"✓ Found {len(start_edges)} edges suitable as start points")

if len(start_edges) < 10:
    print("✗ ERROR: Not enough connected edges found!")
    exit(1)

# Generate vehicles (unsorted first)
print(f"\n✓ Generating {NUM_VEHICLES} vehicles with validated routes...")

vehicles = []  # Store vehicles to sort later

vehicles_created = 0
attempts = 0
max_attempts = NUM_VEHICLES * 5

while vehicles_created < NUM_VEHICLES and attempts < max_attempts:
    attempts += 1
    
    # Pick a random starting edge
    from_edge = random.choice(start_edges)
    
    # Build a connected route (2-5 edges long)
    route_edges = [from_edge]
    current_edge = from_edge
    
    for step in range(random.randint(1, 4)):
        connections = edge_connections.get(current_edge, [])
        if connections:
            next_edge = random.choice(connections)
            route_edges.append(next_edge)
            current_edge = next_edge
        else:
            break
    
    # Need at least 2 edges for a valid route
    if len(route_edges) < 2:
        continue
    
    # Random departure time
    depart = random.uniform(0, SIM_DURATION * 0.8)
    
    # Vehicle type (mostly cars, some vans)
    vtype = 'van' if vehicles_created % 6 == 0 else 'car'
    
    # Create vehicle data
    veh_id = f"veh_{vehicles_created}"
    route_str = ' '.join(route_edges)
    
    vehicles.append({
        'id': veh_id,
        'type': vtype,
        'depart': depart,
        'route': route_str
    })
    
    vehicles_created += 1
    
    if vehicles_created % 20 == 0:
        print(f"  Created {vehicles_created}/{NUM_VEHICLES}...")

# Sort vehicles by departure time (CRITICAL!)
print(f"\n✓ Sorting {len(vehicles)} vehicles by departure time...")
vehicles.sort(key=lambda v: v['depart'])

# Write sorted vehicles to file
print(f"✓ Writing sorted routes to {OUTPUT_FILE}...")

with open(OUTPUT_FILE, 'w') as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<routes>\n')
    
    # Vehicle types
    f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" color="1,1,0"/>\n')
    f.write('    <vType id="van" accel="2.0" decel="4.0" sigma="0.3" length="6" maxSpeed="40" color="0,0,1"/>\n\n')
    
    # Write vehicles (now sorted)
    for veh in vehicles:
        f.write(f'    <vehicle id="{veh["id"]}" type="{veh["type"]}" depart="{veh["depart"]:.2f}">\n')
        f.write(f'        <route edges="{veh["route"]}"/>\n')
        f.write(f'    </vehicle>\n')
    
    f.write('</routes>\n')

print(f"\n✓ SUCCESS! Created {len(vehicles)} vehicles with:")
print(f"  - Validated connectivity")
print(f"  - No restricted edges")
print(f"  - Sorted by departure time")
print(f"  - Saved to {OUTPUT_FILE}")

print(f"\n{'='*70}")
print("NEXT STEPS:")
print("="*70)
print("1. Update tunis_delivery.sumocfg:")
print(f"   <route-files value=\"{OUTPUT_FILE}\"/>")
print("\n2. Test in SUMO:")
print("   sumo-gui -c tunis_delivery.sumocfg")
print("\n3. Click Play and watch vehicles move!")
print("="*70)