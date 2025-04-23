import pandas as pd
import numpy as np

# 读取上传的CSV文件
file_path = r"NSI方法-计算时序tanθ-示例.csv"
data = pd.read_csv(file_path)

# 提取前三列数据：经纬度、高度
points_data = data[['wkt_geom', 'HEIGHT', 'VEL']].dropna()

# 提取形变数据（从D20190106开始的列）
deformation_columns = [col for col in data.columns if col.startswith('D')]
deformation_data = data[deformation_columns].values

# 提取wkt_geom中的经纬度信息
points = []
for _, row in points_data.iterrows():
    geom = row['wkt_geom']
    # 提取经纬度和高度
    lat_lon = geom.split('((')[1].split('))')[0].split(' ')
    lat = float(lat_lon[1])
    lon = float(lat_lon[0])
    height = row['HEIGHT']
    points.append({"lat": lat, "lon": lon, "height": height})

# 将形变数据按日期存入字典
deformation_data_dict = {}
for i, col in enumerate(deformation_columns):
    deformation_data_dict[col] = deformation_data[:, i].tolist()


# 经纬度到XYZ转换的函数
def latlon_to_xyz(lat, lon, height):
    R = 6371000  # 地球半径，单位米
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # 计算XYZ坐标
    x = (R + height) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (R + height) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (R + height) * np.sin(lat_rad)

    return x, y, z


# 计算法向量的函数
def calculate_plane_normal(xyz_points):
    # 计算两个向量
    vector1 = np.array(xyz_points[1]) - np.array(xyz_points[0])
    vector2 = np.array(xyz_points[2]) - np.array(xyz_points[0])

    # 计算法向量 (叉积)
    normal = np.cross(vector1, vector2)
    return normal


# 计算角度的tan值
def calculate_angle_tan(normal1, normal2):
    # 计算法向量之间的夹角
    dot_product = np.dot(normal1, normal2)
    norm1 = np.linalg.norm(normal1)
    norm2 = np.linalg.norm(normal2)
    cos_theta = dot_product / (norm1 * norm2)

    # tan值计算
    tan_theta = np.sqrt(1 - cos_theta ** 2) / cos_theta if cos_theta != 0 else np.inf
    return tan_theta


# 获取第4列的标题作为基准日期
base_date = data.columns[3]

# 读取基准日期的形变数据
base_deformation = deformation_data_dict[base_date]

# 计算基准日期的XYZ坐标和法向量
deformation_data_corrected_base = {key: [val * 0.001 for val in deformation_data_dict[key]] for key in
                                   deformation_data_dict}
xyz_coordinates_base = [latlon_to_xyz(p['lat'], p['lon'], p['height'] + base_deformation[i]) for i, p in
                        enumerate(points)]
normal_base = calculate_plane_normal(xyz_coordinates_base)

# 计算基准日期与其他日期的夹角tan值
tan_values_base = []

for date, deformation in deformation_data_corrected_base.items():
    if date != base_date:
        # 计算与基准平面（base_date）相比的形变
        deformation_difference = [deformation[i] - deformation_data_corrected_base[base_date][i] for i in range(3)]

        # 计算新的XYZ坐标
        deformed_xyz = [latlon_to_xyz(p['lat'], p['lon'], p['height'] + deformation_difference[i]) for i, p in
                        enumerate(points)]

        # 计算当前日期平面法向量
        normal_date = calculate_plane_normal(deformed_xyz)

        # 计算当前日期平面与基准平面之间的夹角tan值
        tan_value = calculate_angle_tan(normal_base, normal_date)
        tan_values_base.append((date, tan_value))

# 输出基准日期和其他日期的夹角tan值
print(f"{base_date}: 0")
for date, tan_value in tan_values_base:
    print(f"{date}: {tan_value:.6f}")