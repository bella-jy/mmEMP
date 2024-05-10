import numpy as np

def read_point_cloud(file_path):
    """
    read
    """
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            point = [float(coord) for coord in data]
            points.append(point)
    return np.array(points)

def compute_average_velocity(points, t):
    """
    velocity
    """
    displacement = np.linalg.norm(points[1:] - points[:-1], axis=1)
    average_velocity = np.mean(displacement) / t
    return average_velocity

def filter_points(points, threshold):
    """
    threshold
    """
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    filtered_points = [points[0]]
    for i in range(1, len(points)):
        if distances[i - 1] <= threshold:
            filtered_points.append(points[i])
    return np.array(filtered_points)

def main():
    # read
    static_points_frame1 = read_point_cloud("static_points_frame1.txt")
    dynamic_points_frame1 = read_point_cloud("dynamic_points_frame1.txt")
    static_points_frame2 = read_point_cloud("static_points_frame2.txt")
    dynamic_points_frame2 = read_point_cloud("dynamic_points_frame2.txt")

    # t
    t = float(input("t: "))
    # T
    T = np.array([[float(x) for x in input().split(',')] for _ in range(3)])
    # dynamic d
    d = float(input("d: "))
    # static d1
    d1 = float(input("d1: "))

    # static velocity
    v1 = compute_average_velocity(static_points_frame1, t)
    # dynamic velocity
    v2 = compute_average_velocity(dynamic_points_frame1, t)

    # threshold
    static_threshold = 0.5 * v1 * t
    dynamic_threshold = 0.5 * v2 * t

    # convert
    transformed_static_points_frame2 = np.dot(static_points_frame2, T)
    transformed_dynamic_points_frame2 = np.dot(dynamic_points_frame2, T)

    # filter
    filtered_static_points = filter_points(transformed_static_points_frame2, static_threshold)
    filtered_dynamic_points = filter_points(transformed_dynamic_points_frame2, dynamic_threshold)

    print("static point number:", len(filtered_static_points))
    print("dynamic point number:", len(filtered_dynamic_points))

if __name__ == "__main__":
    main()
