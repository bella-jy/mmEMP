import os

def load_visual_data(folder):
    visual_data = {}
    for filename in os.listdir(folder):
        timestamp = int(os.path.splitext(filename)[0])
        with open(os.path.join(folder, filename), 'rb') as f:
            # visual
            visual_data[timestamp] = f.read()
    return visual_data

def load_radar_data(folder):
    radar_data = {}
    for filename in os.listdir(folder):
        timestamp = int(os.path.splitext(filename)[0])
        with open(os.path.join(folder, filename), 'r') as f:
            # radar
            radar_data[timestamp] = f.read()
    return radar_data

def load_imu_data(folder):
    imu_data = {}
    for filename in os.listdir(folder):
        timestamp = int(os.path.splitext(filename)[0])
        with open(os.path.join(folder, filename), 'r') as f:
            # IMU
            imu_data[timestamp] = f.read()
    return imu_data

# 加载三种传感器的数据
visual_folder = '\path1'
radar_folder = '\path2'
imu_folder = '\path3'

visual_data = load_visual_data(visual_folder)
radar_data = load_radar_data(radar_folder)
imu_data = load_imu_data(imu_folder)

synced_data = {}

for timestamp in visual_data.keys():
    if timestamp in radar_data and timestamp in imu_data:
        synced_data[timestamp] = {
            'visual': visual_data[timestamp],
            'radar': radar_data[timestamp],
            'imu': imu_data[timestamp]
        }

output_folder = 'synced_data'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for timestamp, data in synced_data.items():
    with open(os.path.join(output_folder, f"{timestamp}.txt"), 'w') as f:
        f.write(f"Visual data: {data['visual']}\n")
        f.write(f"Radar data: {data['radar']}\n")
        f.write(f"IMU data: {data['imu']}\n")

print("finish.")
