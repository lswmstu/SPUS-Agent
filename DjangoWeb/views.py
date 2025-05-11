import base64
import time
from io import BytesIO

import numpy as np
import requests
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
# Create your views here.


import csv
import json
import os
import trueskill
from PIL import Image
from pathlib import Path

from matplotlib import pyplot as plt

from DjangoWeb.others import convert_value, computerThatLonLat, is_distance_less_than_5, \
    find_closest_image_in_direction, get_birdwatchpic, timeit_decorator
from predictCoze import settings
import csv
from collections import defaultdict
import csv
import os
from collections import defaultdict

"""评分"""


def compare_and_update_skills(player1, player2, winner_is_player1=True):
    """
    比较新图片和现有图片的技能水平并更新。
    :param player1: 新图片的技能评分
    :param player2: 现有图片的技能评分
    :param winner_is_player1: 如果新图片胜出为True，否则为False
    :return: 更新后的新图片技能评分，更新后的现有图片技能评分
    """
    # 进行对比并更新技能值
    if winner_is_player1:
        player1, player2 = trueskill.rate_1vs1(player1, player2)  # 新图片胜
    else:
        player2, player1 = trueskill.rate_1vs1(player2, player1)  # 现有图片胜

    return player1, player2


def trueskill_score(env, image_skills, player1, player2, result):
    """
    该函数用于计算比分，image_skills
    :param result: 比赛结果，1/2
    :param env: TrueSkill 环境
    :param image_skills: 记录的已有图片的技能水平
    :param player1: 玩家1的名称
    :param player2: 玩家2的名称
    :return: 无返回值，最后的image_skills就是想要的结果
    """

    if image_skills.get(player1) is None:
        image_skills.update({player1: env.create_rating()})
    if image_skills.get(player2) is None:
        image_skills.update({player2: env.create_rating()})

    if result == '1':
        image_skills[player1], image_skills[player2] = compare_and_update_skills(image_skills[player1],
                                                                                 image_skills[player2],
                                                                                 winner_is_player1=True)
    if result == '2':
        image_skills[player1], image_skills[player2] = compare_and_update_skills(image_skills[player1],
                                                                                 image_skills[player2],
                                                                                 winner_is_player1=False)


def calculate_rating_newimages(rating_data_storages, env, image_skills, new_image_name, new_image_rating_record,
                               whether_save=False):
    """
    该函数用于计算 新图片的分数，基本思想是先初始化样例图片的分数，然后根据只属于该新图片的对比文件来进行对比打分。
    :param new_image_rating_record:
    :param whether_save:
    :param rating_data_storages: 图片一级对比对
    :param env: trueskills的环境
    :param image_skills:存储trueskills的各个玩家评分
    :param new_image_name:新图片的名称
    :return:一张图片的各个维度的评分
    """
    # 新图片和模板图片进行比较。

    # 逐行读取文件内容
    for index, row in enumerate(rating_data_storages, start=1):
        pic_1_id = row[0]
        pic_2_id = row[1]
        features = row[2]
        result = row[3]
        if new_image_name not in [pic_1_id, pic_2_id]:
            continue
        player1, player2 = f'{pic_1_id}-{feature_map[features]}_skill', f'{pic_2_id}-{feature_map[features]}_skill'
        trueskill_score(env=env, image_skills=image_skills, player1=player1, player2=player2, result=result)

    result = []
    result.append({"pic_id": new_image_name,
                   "bustling_skill": None,
                   "lively_skill": None,
                   "clean_skill": None,
                   "beautiful_skill": None,
                   "spacious_skill": None,
                   "modern_skill": None,
                   "commercialization_skill": None,
                   "plants_skill": None,
                   "cultureandart_skill": None, })
    for player_name, skill in image_skills.items():
        if new_image_name in player_name:
            # 使用split()方法按照"-"对文件名进行分割
            split_result = player_name.split("-")
            feature = split_result[1]
            result[0][feature] = {
                "mu": skill.mu,
                "sigma": skill.sigma
            }

    # print("---------------------新图片评分-------------------------")
    # print(result)
    # print("------------------------------------------------------")
    if whether_save:
        append_json_to_file(json_data=result, file_path=new_image_rating_record)

    return result[0]


def image_skills_initialization(image_skills, results_pics, feature):
    """
    该函数用于初始化 已有的trueskill分数
    :param image_skills:存储trueskill玩家
    :param results_pics:需要初始化的数据
    :param feature:特征
    :return:
    """
    for results_pic in results_pics:
        player = results_pic['pic_id'] + '-' + feature
        player_value = trueskill.Rating(mu=results_pic[feature]['mu'], sigma=results_pic[feature]['sigma'])  # 恢复玩家的评分
        if image_skills.get(player) is None:
            image_skills.update({player: player_value})
        # 打印加载的评分
        # print(player,player_value)


""""""

"""其他"""


def list_files_in_directory(directory_path, location, heading):
    """
    判断该文件夹中是否有所需资源。

    :param location: 经纬度
    :param directory_path: 目录的路径
    :param heading: 方向
    :return: 资源名称
    """
    global current_location, current_quantity_found
    try:
        # 检查目录是否存在
        if not os.path.exists(directory_path):
            print(f"目录 {directory_path} 不存在。")
            return []

        # 获取目录中的所有文件和文件夹
        items = os.listdir(directory_path)
        lon, lat = map(float, location.split(','))
        # 筛选出文件（排除文件夹）
        # files = [item for item in items if os.path.isfile(os.path.join(directory_path, item))]

        for item in items:
            # print(item)
            parts = item.split('_')
            lo = float(parts[1])
            la = float(parts[2])
            h = int(parts[3])
            # print(la, lo, h)
            if is_distance_less_than_5(lon1=lon, lat1=lat, lon2=lo, lat2=la):
                if h == heading:
                    print(f"找到了{item}")
                    current_location = f"{lo},{la}"
                    current_quantity_found = current_quantity_found + 1
                    return item
            else:
                continue
        return None
    except Exception as e:
        print(f"读取目录时出错: {e}")
        return None


def read_csv(csv_file):
    """

    :param csv_file:
    :return: datas_reader,row_count
    """
    try:
        with open(csv_file, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # 打印表头（如果有的话）
            headers = next(reader, None)
            # if headers:
            #     print(f"Headers: {headers}")

            # 这两行代码用于获取行数，用于进度条
            datas_reader = list(reader)
            row_count = len(datas_reader)

    except FileNotFoundError:
        print("The file was not found.")
    except csv.Error as e:
        print(f"CSV error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return datas_reader, row_count


def calculate_nearest_points(intervals, point_indexs, current_dimension_rating):
    """
    计算 current_dimension_rating 和 intervals 中各元素的距离，并返回二级分界点中最接近的两个区间。
    :param point_indexs: 包含索引值的列表。
    :param intervals: 包含间隔值的列表。
    :param current_dimension_rating: 当前维度评分值。
    :return: 返回二级分界点中最接近的两个区间。
    """
    first_intervals_index = [point_index for index, point_index in enumerate(point_indexs) if index % 5 == 0]  # 一级索引
    first_intervals = [interval for index, interval in enumerate(intervals) if index % 5 == 0]  # 一级评分
    # 计算每个间隔值与当前维度评分之间的距离
    distances = [(abs(current_dimension_rating - interval), index * 5) for index, interval in
                 enumerate(first_intervals)]

    # 按距离从小到大排序
    sorted_distances = sorted(distances)

    # 获取第一近和第三近的点的索引
    first_nearest_index = sorted_distances[0][1]
    second_nearest_index = sorted_distances[1][1] if len(sorted_distances) >= 2 else None
    third_nearest_index = sorted_distances[2][1] if len(sorted_distances) >= 3 else None

    first_index, second_index = min(first_nearest_index, second_nearest_index), max(first_nearest_index,
                                                                                    second_nearest_index)
    first_level_indexs = point_indexs[first_index + 1:second_index]
    first_index, second_index = min(first_nearest_index, third_nearest_index), max(first_nearest_index,
                                                                                   third_nearest_index)
    second_level_indexs = point_indexs[first_index + 1:second_index]

    # 使用列表推导式去除 b 中的所有 a 元素
    second_level_indexs = [item for item in second_level_indexs if item not in first_level_indexs]
    second_level_indexs = [item for item in second_level_indexs if item not in first_intervals_index]

    return first_level_indexs, second_level_indexs


# 全局缓存结构
CACHE = defaultdict(dict)  # 结构: {sorted_ids: {feature: row_data}}
FILE_MTIME = None  # 记录文件最后修改时间


def build_cache(file_path):
    global CACHE, FILE_MTIME
    CACHE.clear()

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            id1, id2, feature = row[0], row[1], row[2]
            # 创建顺序无关的复合键
            sorted_ids = tuple(sorted((id1, id2)))
            CACHE[sorted_ids][feature] = row

    # 记录文件修改时间
    FILE_MTIME = os.path.getmtime(file_path)


build_cache(file_path='DjangoWeb/static/image_comparison_results_new_images_coze.csv')


def whethertorepeat_cache(pic_1_id, pic_2_id, feature, file_path):
    global CACHE, FILE_MTIME

    # 检查文件是否更新
    current_mtime = os.path.getmtime(file_path)
    if not FILE_MTIME or current_mtime > FILE_MTIME:
        build_cache(file_path)

    # 生成查询键
    query_key = tuple(sorted((pic_1_id, pic_2_id)))

    # 执行快速查询
    if feature_dict := CACHE.get(query_key):
        if row := feature_dict.get(feature):
            return True, row
    return False, None


def whethertorepeat(pic_1_id, pic_2_id, feature, file_path):
    """
    该函数用于判定，当前组合是否已经有数据了
    :param pic_1_id: 图片名称
    :param pic_2_id: 图片名称
    :param feature: 特征
    :param file_path: CSV文件路径
    :return: true已经存在，false不存在
    """

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        # 逐行读取文件内容
        for row in reader:
            pic_id_already = [row[0], row[1]]
            feature_already = row[2]
            if pic_1_id in pic_id_already and pic_2_id in pic_id_already and feature == feature_already:
                # save_to_csv(file_path='E:\临时桌面\python\djangoProject\myapp\static\images_results\\image_comparison_results_new_images_30pics_methed2.csv',
                #             headers=['pic_1_id', 'pic_2_id', 'features', 'result', 'model'],
                #             data=[[row[0], row[1], row[2], row[3], row[4]]])
                return True, row

        return False, None


def qwen_vl_pius_chat(image_url1, image_url2, text, model, processor):
    """
    该函数用于和qwen_vl_pius进行对话，API方法，输入两张图片的地址以及文字，大模型进行回应。
    :param image_url1: 图片地址
    :param image_url2: 图片地址
    :param text: 文本
    :return:
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_url1},
                {"image": image_url2},
                {"text": text}
            ]
        }
    ]
    start_time = time.time()  # 获取当前时间
    # model='qwen-vl-plus'
    # model = "qwen-vl-plus-latest"
    model = 'qwen2-vl-7b-instruct'
    response = dashscope.MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key='',
        model=model,
        messages=messages,
        max_tokens=3,
        result_format='message',
        # stream=True,
        # incremental_output=True,
    )
    end_time = time.time()  # 获取当前时间
    # print(response)
    print(image_url1)
    print(image_url2)
    print(text)
    if response.status_code == HTTPStatus.OK:
        model_reply = response.output['choices'][0]['message']['content'][0]['text']
        print(
            "Model Response:",
            f"${model_reply}$",
            response.usage,
            model,
            f"耗时:{end_time - start_time}秒")  # 打印模型的回复
        return response, model_reply, response.usage, model
    else:
        print("Error occurred:", response.message)  # 如果有错误发生，打印错误信息


def append_json_to_file(json_data, file_path):
    """
    将提供的 JSON 数据追加到指定的 JSON 文件中，不覆盖现有内容。

    :param json_data: 来自 response.json() 的字典或列表等可序列化为 JSON 的数据结构
    :param file_path: 存储 JSON 数据的目标文件路径
    """

    # 确保文件路径存在，如果不存在则创建
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 如果文件存在，则读取已有数据
        if path.exists():
            with open(path, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
        else:
            # 如果文件不存在，则初始化为空列表或空字典
            existing_data = [] if isinstance(json_data, list) else {}

        # 根据 json_data 的类型进行不同的处理
        if isinstance(json_data, list):
            if isinstance(existing_data, list):
                existing_data.extend(json_data)
            else:
                raise TypeError("无法将列表追加到非列表的数据类型")
        elif isinstance(json_data, dict):
            if isinstance(existing_data, dict):
                existing_data.update(json_data)
            else:
                raise TypeError("无法将字典合并到非字典的数据类型")
        else:
            raise TypeError("只支持列表或字典类型的 JSON 数据")

        # 写入更新后的数据到文件中
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)
            print(f"数据已成功保存到 {file_path}")

    except json.JSONDecodeError:
        print(f"警告：文件 {file_path} 包含无效的 JSON 数据。")
    except Exception as e:
        print(f"发生错误: {e}")


def is_valid_image(file_path):
    """
    判断给定路径是否是有效的图片文件。

    :param file_path: 图片文件的路径
    :return: 如果文件是有效图片返回 True，否则返回 False
    """
    if not os.path.isfile(file_path):
        print(f"{file_path} 不是一个文件")
        return False

    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证文件是否为图片
        return True
    except (IOError, SyntaxError) as e:
        print(f"{file_path} 不是一个有效的图片文件: {e}")
        return False


""""""

"""格式转换"""


def csv_to_json_features(csv_file):
    """
    将points_with_features_detail_5.csv文件转换成对应的json文件。
    :param csv_file:'./static/images_results_500/sorted_results_csv/points_with_features_detail_5.csv'
    :return:
    """

    datas_reader, row_count = read_csv(csv_file=csv_file)

    # 构建成json格式
    json_data = []

    for row in datas_reader:
        # print(row)
        data = {'feature': row[0],
                'intervals': [float(item) for item in row[1].split(',')],
                'point_indexs': [int(item) for item in row[2].split(',')],
                'demarc_pic_ids': row[3].split(',')}

        json_data.append(data)

    # print(json_data)
    return json_data


def csv_to_json_skill(csv_file):
    """
    将sorted_beautiful_skill.csv文件转换成对应的json文件。
    :param csv_file:./static/images_results_500/sorted_results_csv/sorted_beautiful_skill.csv
    :return:
    """
    # 文件名的处理
    # 提取文件名部分（去掉路径和扩展名）
    file_name = csv_file.split('/')[-1].split('.')[0]

    # 获取 "beautiful_skill"
    feature_skill = '_'.join(file_name.split('_')[1:])

    datas_reader, row_count = read_csv(csv_file=csv_file)

    # 构建成json格式
    json_data = []

    for row in datas_reader:
        data = {"pic_id": row[0],
                feature_skill: {
                    "mu": float(row[1]),
                    "sigma": float(row[2])
                }}

        json_data.append(data)

        # print(json_data)
    return json_data


""""""

"""预测"""


def qianwen_chat_and_save_newimage(model, processor, image_path, pic_name1, pic_name2, feature,
                                   image_comparison_results_path, whether_model=False):
    """
    该函数用于和大模型聊天并且保存结果,但是区别于qianwen_chat_and_save_all()，本函数用于新图片的判定，那个函数用于采样的图片。
    :param whether_model: 是否使用模型
    :param feature:本次对比的维度，但是此时的是‘bustling_skill’。后面会有处理
    :param image_path:存放图片的路径
    :param pic_name1:图片名称，新图片
    :param pic_name2:图片名称
    :param model:
    :param processor:
    :param image_comparison_results_path:
    :return:
    """
    global total_number_of_requests, repetition_frequency

    total_number_of_requests += 1
    image_url1 = os.path.join(image_path, pic_name1)
    image_url2 = os.path.join(image_path, pic_name2)

    # 遍历字典找到对应的键
    feature = feature.split('_')[0]
    for key, value in feature_map.items():
        if value == feature:
            feature = key
            break

    text = f"比较这两张图片，告诉我哪个更{feature}？解释：1代表第一张图片更{feature}，2代表第二张图片更{feature}，0代表两者相同。输出：只输出0或1或2"

    # 该函数用于判定，是否该信息已经获取，避免信息重复获取
    whetherfind, data = whethertorepeat_cache(pic_1_id=pic_name1, pic_2_id=pic_name2, feature=feature,
                                              file_path=image_comparison_results_path)
    if whetherfind:
        repetition_frequency += 1
        # print('find')
        return data
    if (not is_valid_image(image_url1)) or (not is_valid_image(image_url2)):
        print('无效图片')

    model_reply, model_name = '', ''
    if whether_model:
        model_reply, model_name = qwen_vl_plus_chat(image_url1=image_url1, image_url2=image_url2, text=text,
                                                    model=model, processor=processor)
    print('-' * 20)
    return [pic_name1, pic_name2, feature, model_reply, model_name]


@timeit_decorator
def predicting_new_images_2(new_image_name, image_path, model, processor, image_comparison_results_path,
                            points_with_features, sorted_results, new_image_rating_record, grade=5, whether_model=True,
                            whether_save=False):
    """
    该函数用于新图片的对比，此为方法2 在一级分界线处选2张图片，计算一级评分，然后在最接近的位置选取两张，第二接近的位置选一张。然后对比。
    :param image_path: 图片路径
    :param grade: 分级数 可以是5/3
    :param whether_model: 是否使用模型
    :param whether_save: 是否保存
    :param new_image_rating_record: 记录新图片的评分
    :param sorted_results: 排好序的datas
    :param points_with_features: 用于记录各个维度的比较点的位置
    :param new_image_name: 新图片的名称
    :param model:model
    :param processor:processor
    :param image_comparison_results_path:对比结果的文件路径
    :return:
    """
    global first_level_rating_data_storage, second_level_rating_data_storage
    first_level_rating_data_storage, second_level_rating_data_storage = [], []  # 初始化新图片的一级和二级的对比对数据

    image_skills = {}
    # 初始化已有图片的分数。
    # 创建 TrueSkill 环境
    env = trueskill.TrueSkill()

    # datas = read_json(json_file=points_with_features)
    datas = csv_to_json_features(csv_file=points_with_features)

    features = ['bustling_skill', 'lively_skill', 'clean_skill', 'beautiful_skill', 'spacious_skill', 'modern_skill',
                'commercialization_skill', 'plants_skill', 'cultureandart_skill']

    for feature in features:
        # 把要比较的图片选取出来，4个点，每个点2张。
        results_pics = []  # 存储用于比较的图片
        nearest_indices = []  # 分级点索引
        for data in datas:
            if data['feature'] == feature:
                nearest_indices = [item for index, item in enumerate(data['point_indexs']) if index % 5 == 0]
                break

        sorted_results_path = sorted_results + 'sorted_' + feature + '.csv'
        datas_sorted = csv_to_json_skill(csv_file=sorted_results_path)

        for index in nearest_indices[1:-1]:
            if index >= 0 and index + 1 < len(datas_sorted):
                result = [index, index + 1]
                if grade == 3 and index - 1 >= 0 and index + 2 < len(datas_sorted):
                    result = [index - 1, index, index + 1, index + 2]
                for i in result:
                    results_pics.append(datas_sorted[i])
            else:
                print(f"index={index},越界！！！")

        # 已经获取到了4x2张图片
        # print('每次获取到的4x2张图片:', results_pics)

        for results_pic in results_pics:
            pic_name2 = results_pic['pic_id']
            res_data = qianwen_chat_and_save_newimage(image_path=image_path, pic_name1=new_image_name,
                                                      pic_name2=pic_name2, model=model, processor=processor,
                                                      image_comparison_results_path=image_comparison_results_path,
                                                      feature=feature, whether_model=whether_model)
            # 存储一级对比数据，无结果
            first_level_rating_data_storage.append(res_data)

        # 初始化trueskill
        image_skills_initialization(image_skills=image_skills, results_pics=results_pics, feature=feature)

    # print(image_skills)

    # 计算一级评分
    current_image_rating = calculate_rating_newimages(rating_data_storages=first_level_rating_data_storage, env=env,
                                                      image_skills=image_skills, new_image_name=new_image_name,
                                                      whether_save=whether_save,
                                                      new_image_rating_record=new_image_rating_record)

    # 存储一级对比数据，无结果
    second_level_rating_data_storage = []
    # 现在开始第二级的比较
    for feature in features:

        current_dimension_rating = current_image_rating[feature]['mu']  # 当前维度的评分

        first_level_indexs, second_level_indexs = [], []
        for data in datas:
            if data['feature'] == feature:
                first_level_indexs, second_level_indexs = calculate_nearest_points(
                    intervals=data['intervals'], point_indexs=data['point_indexs'],
                    current_dimension_rating=current_dimension_rating)
                break

        # 把要比较的图片选取出来，4个点，每个点2张。
        results_pics = []  # 存储用于比较的图片

        # sorted_results_path = sorted_results + 'sorted_' + feature + '.json'
        # datas_sorted = read_json(json_file=sorted_results_path)
        sorted_results_path = sorted_results + 'sorted_' + feature + '.csv'
        datas_sorted = csv_to_json_skill(csv_file=sorted_results_path)

        for j, level_indexs in enumerate([first_level_indexs, second_level_indexs]):
            if j == 1:
                for index in level_indexs:
                    results_pics.append(datas_sorted[index])
            else:
                for index in level_indexs:
                    if index >= 0 and index + 1 < len(datas_sorted):
                        result = [index, index + 1]
                        for i in result:
                            results_pics.append(datas_sorted[i])
                    else:
                        print(f"index={index},越界！！！")

        # 已经获取到了4x2+4张图片
        # print('每次获取到的4x2+4张图片:', results_pics)

        for results_pic in results_pics:
            pic_name2 = results_pic['pic_id']
            res_data = qianwen_chat_and_save_newimage(image_path=image_path, pic_name1=new_image_name,
                                                      pic_name2=pic_name2,
                                                      model=model, processor=processor,
                                                      image_comparison_results_path=image_comparison_results_path,
                                                      feature=feature, whether_model=whether_model)
            # 存储二级对比数据，无结果
            second_level_rating_data_storage.append(res_data)

        # 初始化trueskill
        image_skills_initialization(image_skills=image_skills, results_pics=results_pics, feature=feature)

    # print(image_skills)

    # 计算评分
    current_image_rating = calculate_rating_newimages(rating_data_storages=second_level_rating_data_storage, env=env,
                                                      image_skills=image_skills, new_image_name=new_image_name,
                                                      whether_save=whether_save,
                                                      new_image_rating_record=new_image_rating_record)
    print(current_image_rating)
    return current_image_rating


total_number_of_requests, repetition_frequency = 0, 0
first_level_rating_data_storage = []  # 用于记录一级评分存储的图片对比对。
second_level_rating_data_storage = []  # 用于记录二级级评分存储的图片对比对。
third_level_rating_data_storage = []  # 用于记录三级级评分存储的图片对比对。
feature_map = {
    '繁华': 'bustling',
    '热闹': 'lively',
    '整洁': 'clean',
    '美丽': 'beautiful',
    '宽敞': 'spacious',
    '现代化': 'modern',
    '商业化': 'commercialization',
    '植物多': 'plants',
    '文化和文艺': 'cultureandart'
}
image_rating = {
    "pic_id": "panorama_116.422742520108_39.9256155475493_315_1728303434_59174507_\u540c\u798f\u5939\u9053_residential.jpg",
    "bustling_skill": {"mu": 21.019827192951414, "sigma": 2.0597451520721997},
    "lively_skill": {"mu": 31.676883132215043, "sigma": 1.9649053751622192},
    "clean_skill": {"mu": 6.811975912446798, "sigma": 2.3888784262061553},
    "beautiful_skill": {"mu": 7.220173675572495, "sigma": 2.615487278133916},
    "spacious_skill": {"mu": 6.171966195418639, "sigma": 2.659942625763254},
    "modern_skill": {"mu": 14.119323991358653, "sigma": 2.0577079933749927},
    "commercialization_skill": {"mu": 20.705200874568135, "sigma": 2.105901288275674},
    "plants_skill": {"mu": 17.439656727375205, "sigma": 2.022038642236732},
    "cultureandart_skill": {"mu": 8.32563303692594, "sigma": 2.7251073291977703}}


@timeit_decorator
def predict(request):
    """
    该函数用于预测。
    ngrok authtoken 2wNvXXE4KNjG8VMYRo5wnhM17Iv_2xHUdAxf4iw4Snj9Qh4ha
    ngrok http 8000
    """
    global image_rating, current_location, current_heading
    new_image_name = '{"pic_name": "panorama_116.447558551199_39.9169267921825_0_1733044559_244777931_日坛路_tertiary.jpg"}'
    default_value = new_image_name
    if request.method == 'GET':
        my_param_str = request.GET.get('name', default_value)
        # 转换为字典
        my_param_dict = json.loads(my_param_str)
        print(my_param_dict)
        new_image_name = my_param_dict['pic_name']

    image_comparison_results_path = 'DjangoWeb/static/image_comparison_results_new_images_coze.csv'
    points_with_features = 'DjangoWeb/static/sorted_results_csv/points_with_features_detail_5.csv'
    sorted_results = 'DjangoWeb/static/sorted_results_csv/'
    image_path = 'DjangoWeb/static/image_test'
    model = ''
    processor = ''
    new_image_rating_record = ''
    grade = 5
    whether_model = False
    whether_save = False

    image_rating = predicting_new_images_2(new_image_name=new_image_name, image_path=image_path, model=model,
                                           processor=processor,
                                           image_comparison_results_path=image_comparison_results_path,
                                           points_with_features=points_with_features, sorted_results=sorted_results,
                                           new_image_rating_record=new_image_rating_record, grade=5,
                                           whether_model=False,
                                           whether_save=False)

    # return HttpResponse('Hello Django' + image_rating)
    parts = new_image_name.split('_')
    lon = parts[1]  # 经度
    lat = parts[2]  # 纬度
    current_location = f"{lon},{lat}"  # 最新的坐标应该选取图片位置的
    current_heading = int(parts[3])
    print(f"当前位置:{current_location},当前方向:{current_heading}")
    return JsonResponse(image_rating)


@timeit_decorator
def predict_move(request):
    """该函数用于行走时，同时现实雷达图"""
    global image_rating
    new_image_name = 'panorama_116.408206332921_39.9447493168769_270_1727444965_30802727_黑芝麻胡同_service.jpg'
    # 获取图片名称
    if name_current_display_image != '':
        new_image_name = name_current_display_image

    image_comparison_results_path = 'DjangoWeb/static/image_comparison_results_new_images_coze.csv'
    points_with_features = 'DjangoWeb/static/sorted_results_csv/points_with_features_detail_5.csv'
    sorted_results = 'DjangoWeb/static/sorted_results_csv/'
    image_path = 'DjangoWeb/static/image_test'
    whether_model = False

    image_rating = predicting_new_images_2(new_image_name=new_image_name, image_path=image_path, model='', processor='',
                                           image_comparison_results_path=image_comparison_results_path,
                                           points_with_features=points_with_features, sorted_results=sorted_results,
                                           new_image_rating_record='', grade=5, whether_model=whether_model,
                                           whether_save=False)
    return JsonResponse(image_rating)


""""""


@timeit_decorator
def draw_star_chart(request):
    """
    该函数用于绘制9个维度特征的星形图。
    :return: non
    """

    bustling_skill = convert_value(image_rating['bustling_skill']['mu'])
    lively_skill = convert_value(image_rating['lively_skill']['mu'])
    clean_skill = convert_value(image_rating['clean_skill']['mu'])
    beautiful_skill = convert_value(image_rating['beautiful_skill']['mu'])
    spacious_skill = convert_value(image_rating['spacious_skill']['mu'])
    modern_skill = convert_value(image_rating['modern_skill']['mu'])
    commercialization_skill = convert_value(image_rating['commercialization_skill']['mu'])
    plants_skill = convert_value(image_rating['plants_skill']['mu'])
    cultureandart_skill = convert_value(image_rating['cultureandart_skill']['mu'])

    # 1. 设置中文字体（推荐使用较专业的字体）
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体，适用于中文环境

    # 2. 定义维度和数据
    # labels = ['活力指数', '喧闹指数', '整洁度', '美观度', '宽敞度', '历史感指数', '商业化指数', '绿化指数',
    #           '文化艺术指数']
    labels = ['Vibrancy', 'Noisiness', 'Cleanliness', 'Aesthetics', 'Spaciousness', 'Historical',
              'Commercialization', 'Greenery', 'Cultural-Artistic']
    data = [bustling_skill, lively_skill, clean_skill, beautiful_skill, spacious_skill, modern_skill,
            commercialization_skill, plants_skill, cultureandart_skill]  # 示例数据（-2 到 2 之间）
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
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=18)

    # 6. 设置雷达网格
    ax.set_rlabel_position(0)
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=15)
    ax.yaxis.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.xaxis.grid(color='gray', linestyle='--', linewidth=0.8)

    # 7. 绘制雷达图（填充+线条）
    ax.plot(angles, data, color='#007ACC', linewidth=2.5)
    ax.fill(angles, data, color='#007ACC', alpha=0.25)

    # 8. 标题与细节优化
    ax.set_title("Multi-Perception Radar Chart of Streets", size=20, pad=20)

    plt.tight_layout()

    # 将图像保存到内存缓冲区
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    # 转换为 base64 字符串
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    # print(image_base64)

    return HttpResponse(f"data:image/png;base64,{image_base64}")


@timeit_decorator
def get_location_map(request):
    return HttpResponse(get_birdwatchpic(pic_name=name_current_display_image))


"""行走"""
API_KEYS = ['']
API_KEY = API_KEYS[0]

BASE_URL = 'http://api.map.baidu.com/panorama/v2'
LOCATION = ''
WIDTH = 1024
HEIGHT = 512

# 初始全局位置变量
current_location = LOCATION
current_heading = 0

current_quantity_found = 0.0  # 当前可以在图库找到的街景图片的数量,为了防止后面是除数0，所以都设置为了1
current_quantity_downloads = 0.0  # 当前可以从百度云下载的街景图片的数量
current_quantity_failures = 0.0  # 当前从百度云下载失败的街景图片的数量
total_quantity_requests = 1.0  # 当前总的查找图片的次数


def get_panorama(location, heading=0, pitch=0, fov=120, osm_id='null', name='null', highway='null'):
    global current_quantity_downloads, total_quantity_requests
    total_quantity_requests = total_quantity_requests + 1
    params = {
        'ak': API_KEY,
        'width': WIDTH,
        'height': HEIGHT,
        'location': location,
        'heading': heading,  # 水平方向
        'pitch': pitch,  # 垂直方向
        'fov': fov  # 视角
    }
    # print(params)
    # 在本地文件中查询有没有请求的资源
    pic_name = list_files_in_directory(directory_path=os.path.join(settings.STATIC_URL, 'images'), location=location,
                                       heading=heading)
    if pic_name is not None:
        return pic_name

    # 访问百度接口
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        timestamp = int(time.time())  # 获取当前时间戳
        lon, lat = map(float, location.split(','))
        pic_name = f"panorama_{lon}_{lat}_{heading}_{timestamp}_{osm_id}_{name}_{highway}.jpg"
        # input_file_path = os.path.join(settings.BASE_DIR, settings.STATIC_URL, 'images', pic_name)
        file_path = os.path.join(settings.STATIC_URL, 'images', pic_name)
        print(file_path)

        with open(file_path, 'wb') as file:
            file.write(response.content)
            time.sleep(1)
            file.close()

        if is_valid_image(file_path):
            print(f"图片已保存 {pic_name}")
            current_quantity_downloads = current_quantity_downloads + 1
            return pic_name
        return None

    else:
        print(response.status_code)
        return None


@timeit_decorator
def panoramicMapWalking(request):
    global current_location, name_current_display_image, programInitialization

    panorama_image = get_panorama(current_location, current_heading)
    if panorama_image is None:
        return HttpResponse('图片不存在', status=404)

    image_path = os.path.join(settings.STATIC_URL, 'images', panorama_image)

    # 确认文件存在
    if not os.path.isfile(image_path):
        return HttpResponse('图片不存在', status=404)

    name_current_display_image = panorama_image  # 当前展示图片名称。
    programInitialization = True  # 初始化显示。

    # 打开并读取图片文件
    with open(image_path, 'rb') as img:
        # 对图片进行Base64编码
        image_encoded = base64.b64encode(img.read()).decode('utf-8')

        # 返回编码后的图片数据
        return HttpResponse(f'data:image/png;base64,{image_encoded}')


programInitialization = False

name_current_display_image = ''


@timeit_decorator
def panoramicMapWalkingMove(request):
    global current_location
    global current_heading
    global programInitialization, name_current_display_image

    # 在行走前，先判定是否初始化。
    if programInitialization:

        direction = request.GET.get('direction')
        lon, lat = map(float, current_location.split(','))

        if direction == 'forward':
            current_heading = current_heading
        elif direction == 'backward':
            current_heading = ((current_heading + 360) - 180) % 360
        elif direction == 'left':
            current_heading = ((current_heading + 360) - 90) % 360
        elif direction == 'right':
            current_heading = ((current_heading + 360) + 90) % 360

        if direction == 'forward':
            closest_image = find_closest_image_in_direction(current_lat=lat, current_lon=lon, heading=current_heading,
                                                            image_dir='DjangoWeb/static/image_test')
        else:
            closest_image = find_closest_image_in_direction(current_lat=lat, current_lon=lon, heading=current_heading,
                                                            image_dir='DjangoWeb/static/image_test', state=False)

        if closest_image is None:
            return HttpResponse('图片不存在', status=404)

        parts = closest_image['filename'].split('_')
        if len(parts) >= 6:
            lon = parts[1]  # 经度
            lat = parts[2]  # 纬度
            current_location = f"{lon},{lat}"  # 最新的坐标应该选取图片位置的

        image_path = os.path.join(settings.STATIC_URL, 'images', closest_image['filename'])
        print(f"当前位置:{current_location},当前方向:{current_heading}")
        # 确认文件存在
        if not os.path.isfile(image_path):
            return HttpResponse('图片不存在', status=404)

        name_current_display_image = closest_image['filename']  # 当前展示图片名称。
        # 打开并读取图片文件
        with open(image_path, 'rb') as img:
            # 对图片进行Base64编码
            image_encoded = base64.b64encode(img.read()).decode('utf-8')

            # 返回编码后的图片数据
            return HttpResponse(f'data:image/png;base64,{image_encoded}')
    else:
        return panoramicMapWalking(request)


""""""
if __name__ == '__main__':
    pass
