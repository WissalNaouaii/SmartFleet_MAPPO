#!/usr/bin/env python3
"""
SmartFleet Episode Log Converter
Converts the Python simulation output to Unity-compatible JSON format

Usage:
    python convert_episode_for_unity.py episode_1_log.json episode_unity.json
"""
import json
import re
import sys
import os

def parse_position_string(pos_str):
    """
    Convert position string "(308.6, 230.7)" to [x, z] array
    Also applies offset to center the map around origin
    """
    if isinstance(pos_str, list):
        return pos_str
    
    if isinstance(pos_str, str):
        # Extract numbers from string like "(308.6, 230.7)"
        numbers = re.findall(r'[-+]?\d*\.?\d+', pos_str)
        if len(numbers) >= 2:
            x = float(numbers[0])
            z = float(numbers[1])
            return [x, z]
    
    return [0.0, 0.0]


def convert_episode_log(input_path, output_path):
    """Convert SmartFleet episode log to Unity format"""
    
    print(f"Loading {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize Unity-compatible structure
    unity_data = {
        "episode": 1,
        "config": data.get("config", {
            "city_size": 500,
            "sim_duration": 600,
            "n_agents": 5
        }),
        "final_metrics": data.get("final_metrics", {}),
        "steps": []
    }
    
    print(f"Processing {len(data.get('steps', []))} steps...")
    
    # Track max completed for progress
    max_completed = 0
    
    for step_data in data.get("steps", []):
        unity_step = {
            "step": step_data.get("step", 0),
            "time": step_data.get("time", 0.0),
            "agents": []
        }
        
        for agent in step_data.get("agents", []):
            # Parse position from string format
            pos = parse_position_string(agent.get("position", "(0, 0)"))
            
            unity_agent = {
                "id": agent.get("agent_id", "AGENT_00"),
                "vehicle_type": agent.get("vehicle_type", "van"),
                "position": pos,
                "battery_percent": agent.get("battery_percent", 100.0),
                "state": agent.get("state", "idle"),
                "current_task": agent.get("current_task") or ""
            }
            
            unity_step["agents"].append(unity_agent)
            
            # Track completed deliveries
            completed = agent.get("completed_total", 0)
            if completed > max_completed:
                max_completed = completed
        
        unity_data["steps"].append(unity_step)
    
    # Write output
    print(f"Writing {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unity_data, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("CONVERSION COMPLETE")
    print("=" * 50)
    print(f"✓ Steps converted: {len(unity_data['steps'])}")
    print(f"✓ Agents: {len(unity_data['steps'][0]['agents']) if unity_data['steps'] else 0}")
    print(f"✓ Max deliveries: {max_completed}")
    
    if unity_data.get('final_metrics'):
        fm = unity_data['final_metrics']
        print(f"✓ Total completed: {fm.get('total_completed', 'N/A')}")
        print(f"✓ On-time rate: {fm.get('on_time_rate', 'N/A')}")
        print(f"✓ Energy used: {fm.get('total_energy_kwh', 'N/A'):.1f} kWh")
    
    print(f"\n📁 Output saved to: {output_path}")
    print(f"\n📋 Next steps:")
    print(f"   1. Copy {output_path} to Unity project:")
    print(f"      → Assets/StreamingAssets/episode_1_log.json")
    print(f"   2. Press Play in Unity to see visualization!")
    
    return unity_data


def main():
    if len(sys.argv) < 2:
        print("SmartFleet Episode Log Converter")
        print("=" * 40)
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <input.json> [output.json]")
        print("\nExample:")
        print(f"  python {sys.argv[0]} episode_1_log.json episode_unity.json")
        print("\nThis converts your SmartFleet simulation log to Unity format.")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Default output name
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}_unity.json"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    convert_episode_log(input_file, output_file)


if __name__ == "__main__":
    main()
