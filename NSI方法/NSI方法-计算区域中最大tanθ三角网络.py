import pandas as pd
import numpy as np
from pyproj import Proj, transform
from itertools import combinations

# 设置WGS84坐标系和UTM坐标系
wgs84 = Proj(init='epsg:4326')  # WGS84坐标系 (经纬度)
utm = Proj(init='epsg:32633')  # UTM坐标系（以米为单位，假设使用UTM Zone 33N）

# 读取上传的CSV文件
file_path = r"NSI方法-计算区域中最大tanθ三角网络-示例.csv"  # 文件路径
data = pd.read_csv(file_path)

# 提取前三列数据：经纬度、高度和沉降速率
points_data = data[['wkt_geom', 'HEIGHT', 'VEL']].dropna()

# 提取wkt_geom中的经纬度信息并剔除重复的点
points = []
for _, row in points_data.iterrows():
    geom = row['wkt_geom']
    # 提取经纬度和高度
    lat_lon = geom.split('((')[1].split('))')[0].split(' ')
    lat = float(lat_lon[1])
    lon = float(lat_lon[0])
    height = row['HEIGHT']
    vel = row['VEL']  # 沉降速率

    # 去除重复点
    point = {"lat": lat, "lon": lon, "height": height, "vel": vel}
    if point not in points:
        points.append(point)


# 经纬度转换为地面坐标（米）的函数
def convert_to_xy(lat, lon):
    # 使用pyproj转换经纬度为xy坐标（米为单位）
    x, y = transform(wgs84, utm, lon, lat)
    return x, y


# 将点转换为地面坐标
for point in points:
    point['xy'] = convert_to_xy(point['lat'], point['lon'])


# 计算三角形边长的函数
def calculate_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# 生成符合边长大于 X 米条件的点对
valid_edges = []
for i in range(len(points)):
    for j in range(i + 1, len(points)):
        x1, y1 = points[i]['xy']
        x2, y2 = points[j]['xy']
        length = calculate_length(x1, y1, x2, y2)

        if length >= 90 and length <= 180:  # 只保留边长位于X1和X2之间的点对
            valid_edges.append((i, j))

# 生成所有符合边长条件的三角形，并根据边长比例判断夹角
triangles = []
for (i, j) in valid_edges:  # 只遍历有效的点对
    for (k, l) in valid_edges:
        if j == k:  # 确保每组三点是独立的
            i, k, l = sorted([i, j, l])  # 排序使得三角形的点按顺序组合
            x1, y1 = points[i]['xy']
            x2, y2 = points[j]['xy']
            x3, y3 = points[l]['xy']

            # 计算三角形的边长
            a = calculate_length(x1, y1, x2, y2)
            b = calculate_length(x2, y2, x3, y3)
            c = calculate_length(x1, y1, x3, y3)

            # 边长比例筛选条件：确保边长比不超过某个阈值，以避免角度过小或过大
            # 比如，若 a, b, c 三条边的比例过大，则说明某个角度非常接近 0 或 180 度
            if ((a / b) < 1.732 and (a / c) < 1.732) and ((b / a) < 1.732 and (b / c) < 1.732) and ((c / a) < 1.732 and (c / b) < 1.732) :
                print(
                    f"三角形 {i + 1}-{j + 1}-{l + 1} 边长: {a/3}, {b/3}, {c/3}")  # 打印每个三角形的边长

                # 筛选条件其实是夹角，减少计算量
                triangles.append({
                    "triangle": (points[i], points[j], points[l]),
                    "edges": (a, b, c)
                })

# 如果没有三角形，输出点的数量
if len(triangles) == 0:
    print(f"数据中总共有 {len(points)} 个点，无法形成符合条件的三角形。")
else:
    print(f"生成了 {len(triangles)} 个三角形。")


# 计算新旧三角形的夹角tan(θ)的函数
def calculate_tan_theta(triangle):
    (p1, p2, p3) = triangle["triangle"]
    x1, y1 = p1['xy']
    x2, y2 = p2['xy']
    x3, y3 = p3['xy']
    h1, h2, h3 = p1['height'], p2['height'], p3['height']
    vel1, vel2, vel3 = p1['vel'], p2['vel'], p3['vel']  # 使用沉降速率来计算

    # 原始三角形的矢量（xy坐标平面）
    v1 = np.array([x2 - x1, y2 - y1, 0])
    v2 = np.array([x3 - x2, y3 - y2, 0])
    v3 = np.array([x1 - x3, y1 - y3, 0])

    # 新的三角形矢量（包括沉降，即Z轴变化）
    v1_z = np.array([v1[0], v1[1], (vel2 - vel1) / 1000])  # 沉降转换为米
    v2_z = np.array([v2[0], v2[1], (vel3 - vel2) / 1000])  # 沉降转换为米
    v3_z = np.array([v3[0], v3[1], (vel1 - vel3) / 1000])  # 沉降转换为米

    # 计算夹角的tan(θ)，使用矢量叉积公式
    tan_theta1 = np.linalg.norm(np.cross(v1, v1_z)) / np.dot(v1, v1_z)
    tan_theta2 = np.linalg.norm(np.cross(v2, v2_z)) / np.dot(v2, v2_z)
    tan_theta3 = np.linalg.norm(np.cross(v3, v3_z)) / np.dot(v3, v3_z)

    # 返回最大tan(θ)
    return max(tan_theta1, tan_theta2, tan_theta3)


# 计算所有三角形的tan(θ)并找到最大值的三角形
max_tan_theta_triangle = None
max_tan_theta_value = -1

for triangle in triangles:
    tan_theta_value = calculate_tan_theta(triangle)
    if tan_theta_value > max_tan_theta_value:
        max_tan_theta_value = tan_theta_value
        max_tan_theta_triangle = triangle

# 输出最大tan(θ)的三角形及其tan(θ)
if max_tan_theta_triangle is not None:
    print("最大tan(θ)三角形:", max_tan_theta_triangle)
    print("最大tan(θ)值:", max_tan_theta_value)
else:
    print("没有找到符合条件的三角形。")