import base64
import io
import os
import re
from geopy.distance import geodesic
import math
import time
from PIL import Image
import csv
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from django.test import TestCase
from matplotlib import pyplot as plt
import os
import re
import time
from qgis.PyQt.QtCore import QSize
from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsPointXY,
    QgsRectangle
)
import math
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtWidgets import QApplication


def timeit_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 更精确的时间测量
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to complete.")
        return result

    return wrapper


def convert_value(x):
    """
    将值从[0, 50]区间线性转换到[1, 5]区间。

    参数:
    x -- 输入的原始值，应在[0, 50]区间内。

    返回:
    转换后的值，范围在[1, 5]区间内。
    """
    if x < 0 or x > 50:
        raise ValueError("x 必须在 [0, 50] 区间内")

    # 应用线性转换公式
    return (x / 50) * 4 + 1


def is_distance_less_than_5(lon1, lat1, lon2, lat2):
    """
    判断两点之间的距离是否小于5米

    :param lon1: 第一个点的经度
    :param lat1: 第一个点的纬度
    :param lon2: 第二个点的经度
    :param lat2: 第二个点的纬度
    :return: 如果距离小于5米，返回True；否则返回False
    """
    distance = haversine_distance(lon1, lat1, lon2, lat2)
    return distance < 5


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    计算两个经纬度之间的球面距离

    :param lon1: 第一个点的经度
    :param lat1: 第一个点的纬度
    :param lon2: 第二个点的经度
    :param lat2: 第二个点的纬度
    :return: 两点之间的距离（米）
    """
    # 将角度转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine公式
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 地球半径（米）
    earth_radius = 6378137.0

    # 计算距离
    distance = earth_radius * c
    return distance


# 大地坐标系资料WGS - 84 长半径a = 6378137 短半径b = 6356752.3142 扁率f = 1 / 298.2572236
# / ** 长半径a = 6378137 * /
a = 6378137
# / ** 短半径b = 6356752.3142 * /
b = 6356752.3142
# / ** 扁率f = 1 / 298.2572236 * /
f = 1 / 298.2572236


def computerThatLonLat(lon, lat, brng, dist):
    """
    根据当前经纬度、行进距离和方向角计算新的经纬度
    :param lon: 当前经度（度）
    :param lat: 当前纬度（度）
    :param dist: 行进距离（米）
    :param brng: 方向角（度）
    :return: 新的经纬度 (new_lat, new_lon)
    """
    global a
    global b
    global f

    alpha1 = rad(brng)
    sinAlpha1 = math.sin(alpha1)
    cosAlpha1 = math.cos(alpha1)

    tanU1 = (1 - f) * math.tan(rad(lat))
    cosU1 = 1 / math.sqrt((1 + tanU1 * tanU1))
    sinU1 = tanU1 * cosU1
    sigma1 = math.atan2(tanU1, cosAlpha1)
    sinAlpha = cosU1 * sinAlpha1
    cosSqAlpha = 1 - sinAlpha * sinAlpha
    uSq = cosSqAlpha * (a * a - b * b) / (b * b)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))

    cos2SigmaM = 0
    sinSigma = 0
    cosSigma = 0
    sigma = dist / (b * A)
    sigmaP = 2 * math.pi
    while math.fabs(sigma - sigmaP) > 1e-12:
        cos2SigmaM = math.cos(2 * sigma1 + sigma)
        sinSigma = math.sin(sigma)
        cosSigma = math.cos(sigma)
        deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (
                cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM) - B / 6 * cos2SigmaM * (
                -3 + 4 * sinSigma * sinSigma) * (-3 + 4 * cos2SigmaM * cos2SigmaM)))
        sigmaP = sigma
        sigma = dist / (b * A) + deltaSigma

    tmp = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1

    lat2 = math.atan2(sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1,
                      (1 - f) * math.sqrt(sinAlpha * sinAlpha + tmp * tmp))

    lambda_ = math.atan2(sinSigma * sinAlpha1, cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1)

    lambda_ = math.atan2(sinSigma * sinAlpha1, cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1)

    C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))

    L = lambda_ - (1 - C) * f * sinAlpha * (
            sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)))
    revAz = math.atan2(sinAlpha, -tmp)
    # print(revAz)
    # print(f"{lon + deg(L)} , {deg(lat2)}")
    new_lon = lon + deg(L)
    new_lat = deg(lat2)
    return new_lon, new_lat


def rad(d):
    """
    度换成弧度

    :param d: 度
    :return 弧度
    """
    return d * math.pi / 180.0


def deg(x):
    """
    弧度换成度

    :param x: 弧度
    :return 度
    """
    return x * 180 / math.pi


def extract_LatAndLon():
    """提取文件名中的经纬度"""

    # 设置图片文件夹路径和输出CSV路径
    image_folder = "./static/images"
    output_csv = "./static/coordinates.csv"

    # 创建CSV文件并写入表头
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ImageName", "Longitude", "Latitude", "Address", "RoadNumber"])  # 可根据需要添加其他字段

        # 遍历图片文件
        for filename in os.listdir(image_folder):
            if filename.endswith(".jpg"):
                parts = filename.split('_')
                if len(parts) >= 6:
                    lon = parts[1]  # 经度
                    lat = parts[2]  # 纬度
                    road_number = parts[5]  # 道路编号
                    address = parts[6]  # 地址（如“西四南大街”）
                    writer.writerow([filename, lon, lat, address, road_number])

    print("经纬度提取完成！")


def calculate_bearing(point_a, point_b):
    """
    计算从 point_a 到 point_b 的方位角（单位：度）
    point_a / point_b: (lat, lon)
    """
    lat1, lon1 = math.radians(point_a[0]), math.radians(point_a[1])
    lat2, lon2 = math.radians(point_b[0]), math.radians(point_b[1])

    d_lon = math.radians(point_b[1] - point_a[1])

    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(d_lon))

    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


@timeit_decorator
def find_closest_image_in_direction(current_lat, current_lon, heading, image_dir, state=True):
    """
    查找位于当前位置前进方向上，且距离在 50~150 米之间的图片。

    :param state:
    :param current_lat: 当前纬度
    :param current_lon: 当前经度
    :param heading: 当前方向（0~360 度）
    :param image_dir: 图片所在目录
    :return: 符合条件的图片文件名及详细信息
    """
    if not state:
        for filename in os.listdir(image_dir):
            lon = float(filename.split('_')[1])
            lat = float(filename.split('_')[2])
            pic_dir = filename.split('_')[3]
            pic_diff = abs((float(pic_dir) - heading + 180) % 360 - 180)
            if is_distance_less_than_5(lon1=current_lon, lat1=current_lat, lon2=lon, lat2=lat) and pic_diff <= 30:
                closest_image = {
                    'filename': filename,
                    'pic_diff': pic_diff
                }
                print(closest_image)
                return closest_image

        return None

    closest_image = None
    min_distance = float('inf')
    min_angle_diff = float('inf')
    min_pic_diff = float('inf')

    # 正则表达式匹配文件名中的经纬度
    pattern = r'panorama_(-?\d+\.\d+)_(-?\d+\.\d+)_.*\.jpg'

    for filename in os.listdir(image_dir):
        match = re.match(pattern, filename)
        if match:
            lon = float(match.group(1))
            lat = float(match.group(2))

            distance_m = geodesic((current_lat, current_lon), (lat, lon)).meters

            # 检查距离是否在指定范围内
            if 50 <= distance_m <= 100:
                bearing = calculate_bearing((current_lat, current_lon), (lat, lon))
                angle_diff = abs((bearing - heading + 180) % 360 - 180)
                pic_dir = filename.split('_')[3]
                pic_diff = abs((float(pic_dir) - heading + 180) % 360 - 180)

                # 检查角度差异是否在允许范围内
                if angle_diff <= 30:
                    # 更新最近的图片
                    if distance_m < min_distance or (distance_m == min_distance and angle_diff < min_angle_diff) or (
                            distance_m == min_distance and angle_diff == min_angle_diff and pic_diff < min_pic_diff):
                        closest_image = {
                            'filename': filename,
                            'distance': distance_m,
                            'angle_diff': angle_diff,
                            'pic_diff': pic_diff
                        }
                        min_distance = distance_m
                        min_angle_diff = angle_diff
                        min_pic_diff = pic_diff
    print(closest_image)
    return closest_image


def count_and_save_road_numbers_with_address(input_csv, output_csv):
    road_number_data = {}

    # 读取CSV文件并收集RoadNumber及其对应的Address
    with open(input_csv, mode='r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            road_number = row['RoadNumber']
            address = row['Address']
            if road_number not in road_number_data:
                road_number_data[road_number] = {
                    'count': 0,
                    'address': address
                }
            road_number_data[road_number]['count'] += 1

    # 统计每个RoadNumber的出现次数
    counter = Counter({road_number: data['count'] for road_number, data in road_number_data.items()})

    # 写入结果到新文件
    with open(output_csv, mode='w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['RoadNumber', 'Count', 'Address'])

        for road_number, count in counter.most_common():
            address = road_number_data[road_number]['address']
            writer.writerow([road_number, count, address])


def search_images_in_folder(folder_path, road_numbers_list, output_file):
    """
    在指定文件夹中搜索图片，并根据RoadNumber筛选符合条件的图片。

    :param folder_path: 存储图片的文件夹路径
    :param road_numbers_list: 包含RoadNumbers的列表
    :param output_file: 输出结果的txt文件路径
    """
    matched_files = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 确保只处理 .jpg 文件
        if filename.endswith('.jpg'):
            if len(filename.split('_')) <= 5:
                continue
            road_number = filename.split('_')[5]
            if road_number in road_numbers_list:
                matched_files.append(filename)

    # 将匹配的文件名写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for matched_file in matched_files:
            f_out.write(f"{matched_file}\n")


def filter_csv_by_ids(input_csv, output_csv, id_list):
    """
    过滤CSV文件中的行，如果 pic_1_id 或 pic_2_id 包含 id_list 中的任意一个ID，
    则将该行写入新的CSV文件。

    :param input_csv: 输入CSV文件路径
    :param output_csv: 输出CSV文件路径
    :param id_list: 要匹配的ID列表，例如 ['244777931', '33616161']
    """
    with open(input_csv, mode='r', newline='', encoding='utf-8') as infile, \
            open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)

        # 写入表头
        writer.writeheader()

        # 遍历每一行数据
        for row in reader:
            pic1 = row['pic_1_id']
            pic2 = row['pic_2_id']

            # 检查是否包含任何一个 ID
            if any(id_str in pic1 or id_str in pic2 for id_str in id_list):
                writer.writerow(row)

    print(f"筛选完成，结果已保存至 {output_csv}")


def get_birdwatchpic(pic_name: str):
    """
    该函数用于生成俯视图。
    :param pic_name: 图片名称 panorama_116.41293178071_39.9471880214627_270_1727101578_25091600_鼓楼东大街_secondary.jpg
    :return:
    """
    # 俯视图片名称
    name_no_ext = os.path.splitext(pic_name)[0]
    pic_brid = f"DjangoWeb/static/birdwatchpics/{name_no_ext}_snapshot.png"

    # pano名称
    heading = pic_name.split('_')[3]
    pic_pano = 'DjangoWeb/static/pano/pano_' + heading + '.png'

    # 打开两张图片
    img_a = Image.open(pic_brid)
    img_b = Image.open(pic_pano)  # PNG 支持透明背景

    # 获取尺寸
    width_a, height_a = img_a.size
    width_b, height_b = img_b.size

    # 计算居中位置
    x = (width_a - width_b) // 2
    y = (height_a - height_b) // 2

    # 粘贴图B到图A中间，支持透明背景
    img_a.paste(img_b, (x, y), mask=img_b if img_b.mode == 'RGBA' else None)

    img_a = crop_center_keep_ratio(img_a)

    # 使用BytesIO创建内存中的字节流
    buffered = io.BytesIO()
    # 将处理后的图像保存到字节流中
    img_a.save(buffered, format="PNG")
    # 重置字节流的位置到开始处
    buffered.seek(0)

    # 读取字节流中的数据并进行base64编码
    image_encoded = base64.b64encode(buffered.read()).decode('utf-8')
    return f'data:image/png;base64,{image_encoded}'


def crop_center_keep_ratio(img_a):
    """
    对给定的图片img_a进行等比例裁剪，裁剪后的宽度为原宽度的3/4，
    并保持原始宽高比，裁剪从图片中心开始。

    参数:
        img_a: 使用PIL.Image.open()打开的图像对象

    返回:
        裁剪并调整大小后的图像对象
    """
    # 原始尺寸
    original_width, original_height = img_a.size

    # 新的宽度为原来的3/4
    new_width = int(original_width * 2 / 4)

    # 根据新的宽度计算高度以保持比例不变
    new_height = int(new_width * (original_height / original_width))

    # 计算左、上、右、下的坐标
    left = (original_width - new_width) / 2
    top = (original_height - new_height) / 2
    right = (original_width + new_width) / 2
    bottom = (original_height + new_height) / 2

    # 裁剪图片
    img_cropped = img_a.crop((int(left), int(top), int(right), int(bottom)))

    return img_cropped


def draw_star_chart():
    """
    该函数用于绘制9个维度特征的星形图。
    :return: non
    """
    # 1. 设置中文字体（推荐使用较专业的字体）
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']  # 黑体，适用于中文环境

    # 2. 定义维度和数据
    # labels = ['活力指数', '喧闹指数', '整洁度', '美观度', '宽敞度', '历史感指数', '商业化指数', '绿化指数',
    #           '文化艺术指数']
    labels = ['Vibrancy', 'Noisiness', 'Cleanliness', 'Spaciousness', 'Aesthetics', 'Historical',
              'Commercialization', 'Greenery', 'Cultural-Artistic']
    data = [4.8, 3.5, 4.2, 3.6, 4.3, 2.0, 4.4, 2.8, 3.9]  # 示例数据（-2 到 2 之间）
    print(data)
    # 3. 雷达图角度设置
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    data += data[:1]  # 闭合图形
    angles += angles[:1]

    # 4. 图形初始化
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 5. 设置坐标标签
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=24)

    # 6. 设置雷达网格
    ax.set_rlabel_position(0)
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', ''], fontsize=18)
    ax.yaxis.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.xaxis.grid(color='gray', linestyle='--', linewidth=0.8)

    # 7. 绘制雷达图（填充+线条）
    ax.plot(angles, data, color='#007ACC', linewidth=2.5)
    ax.fill(angles, data, color='#007ACC', alpha=0.25)

    # 8. 标题与细节优化
    ax.set_title("Multi-Perception Radar Chart of Streets", size=20, pad=20)

    plt.tight_layout()

    plt.show()


import base64
import csv
from io import BytesIO


# 常量
x_pi = math.pi * 3000.0 / 180.0


def bd09_to_gcj02(bd_lon, bd_lat):
    """
    把百度经纬度(BD-09)转换为高德/Google中国(火星坐标 GCJ-02)
    :param bd_lon: 百度经度
    :param bd_lat: 百度纬度
    :return: (gcj_lon, gcj_lat)
    """
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    gg_lon = z * math.cos(theta)
    gg_lat = z * math.sin(theta)
    return gg_lon, gg_lat


from qgis.PyQt.QtCore import QEventLoop, QTimer


def wait(ms: int):
    """
    阻塞当前函数，但不阻塞 Qt 事件循环，
    并处理所有 pending 的 GUI 事件。
    :param ms: 等待毫秒数
    """
    loop = QEventLoop()
    # 定时退出事件循环
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()  # 进入循环，直到 QTimer 触发 quit


def snapshot_from_filename(
        file_path: str,
        buffer_size: float = 500,
        img_size: tuple = (800, 600),
        output_dir: str = None,
        point_layer_name: str = "gaode_coor"
) -> bool:
    """
    根据文件名提取 lon/lat（BD-09），转为 GCJ-02，选中对应要素，
    以该点为中心截图。
    """
    # 1. 提取文件名中的百度经纬度
    basename = os.path.basename(file_path)
    m = re.match(r'panorama_(?P<lon>-?[\d\.]+)_(?P<lat>-?[\d\.]+)_.*\.jpg', basename)
    if not m:
        print(f"[ERROR] 文件名不符合预期格式: {basename}")
        return False

    lon_bd, lat_bd = float(m.group('lon')), float(m.group('lat'))
    # 转为高德(火星)坐标
    lon, lat = bd09_to_gcj02(lon_bd, lat_bd)

    # 2. 获取项目和画布
    project = QgsProject.instance()
    canvas = iface.mapCanvas()

    # 3. 将经纬度转换到项目 CRS
    src_crs = QgsCoordinateReferenceSystem('EPSG:4326')
    dest_crs = project.crs()
    xform = QgsCoordinateTransform(src_crs, dest_crs, project)
    pt = xform.transform(QgsPointXY(lon, lat))

    # 4. 选中该点对应的要素
    try:
        point_layer = project.mapLayersByName(point_layer_name)[0]
        point_layer.removeSelection()
        expr = f"\"Longitude\" = {lon_bd} AND \"Latitude\" = {lat_bd}"
        req = QgsFeatureRequest().setFilterExpression(expr)
        ids = [feat.id() for feat in point_layer.getFeatures(req)]
        point_layer.selectByIds(ids)
        print(f"选中要素ID: {ids}")
    except Exception as e:
        print(f"[WARN] 通过属性选中要素失败: {e}")

    # 5. 设置地图范围并刷新
    half = buffer_size / 2.0
    rect = QgsRectangle(pt.x() - half, pt.y() - half, pt.x() + half, pt.y() + half)
    canvas.setExtent(rect)
    canvas.refresh()

    # 6. 等待重绘
    QApplication.processEvents()
    wait(1000)

    # 7. 调整画布尺寸
    canvas.resize(QSize(*img_size))

    # 8. 确定输出路径
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    os.makedirs(output_dir, exist_ok=True)
    name_no_ext = os.path.splitext(basename)[0]
    out_name = f"{name_no_ext}_snapshot.png"
    out_path = os.path.join(output_dir, out_name)

    # 9. 保存截图
    success = canvas.saveAsImage(out_path)
    if success:
        print(f"[OK] 已保存截图: {out_path}")
    else:
        print(f"[ERROR] 保存失败: {out_path}")
    return success


if __name__ == "__main__":

    pass
