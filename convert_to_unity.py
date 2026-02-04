#!/usr/bin/env python3
"""
Convert SmartFleet episode log to Unity-compatible format
"""
import json
import re
import sys

def parse_position(pos_str):
    """Convert '(308.6, 230.7)' to [308.6, 230.7]"""
    if isinstance(pos_str, list):
        return pos_str
    if isinstance(pos_str, str):
        # Extract numbers from string like "(308.6, 230.7)"
        numbers = re.findall(r'[-+]?\d*\.?\d+', pos_str)
        if len(numbers) >= 2:
            return [float(numbers[0]), float(numbers[1])]
    return [0.0, 0.0]

def convert_log(input_path, output_path):
    """Convert episode log to Unity format"""
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Create Unity-compatible format
    unity_log = {
        "episode": 1,
        "steps": []
    }
    
    for step_data in data.get("steps", []):
        unity_step = {
            "step": step_data.get("step", 0),
            "simulation_time": step_data.get("time", 0.0),
            "completed_deliveries": 0,
            "on_time_rate": 1.0,
            "agents": [],
            "events": []
        }
        
        # Convert agents
        for agent in step_data.get("agents", []):
            unity_agent = {
                "id": agent.get("agent_id", ""),
                "vehicle_type": agent.get("vehicle_type", "van"),
                "position": parse_position(agent.get("position", "(0,0)")),
                "battery_percent": agent.get("battery_percent", 100.0),
                "state": agent.get("state", "idle"),
                "current_task": agent.get("current_task") or ""
            }
            unity_step["agents"].append(unity_agent)
            
            # Track completed deliveries
            unity_step["completed_deliveries"] = max(
                unity_step["completed_deliveries"],
                agent.get("completed_total", 0)
            )
        
        unity_log["steps"].append(unity_step)
    
    # Add final metrics if available
    if "final_metrics" in data:
        unity_log["final_metrics"] = data["final_metrics"]
    
    # Write output
    with open(output_path, 'w') as f:
        json.dump(unity_log, f, indent=2)
    
    print(f"✓ Converted {len(unity_log['steps'])} steps")
    print(f"✓ Output saved to: {output_path}")
    
    # Show sample
    if unity_log["steps"]:
        sample = unity_log["steps"][0]
        print(f"\nSample step 1:")
        print(f"  Time: {sample['simulation_time']}s")
        print(f"  Agents: {len(sample['agents'])}")
        if sample['agents']:
            agent = sample['agents'][0]
            print(f"  First agent: {agent['id']} at {agent['position']}")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    else:
        input_file = "episode_1_log.json"
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = "episode_1_unity.json"
    
    convert_log(input_file, output_file)
