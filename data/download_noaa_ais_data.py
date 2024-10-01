# 文件名：download_noaa_ais_data.py

import os
import requests
from datetime import datetime
import zipfile
import glob
import pandas as pd

def download_noaa_ais_data(year, month, download_dir):
    """
    下载指定年份和月份的 NOAA AIS 数据。

    参数：
    - year: 年份，例如 2020
    - month: 月份，1-12
    - download_dir: 数据下载保存的目录
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # NOAA AIS 数据集链接模板
    base_url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/"
    file_template = "AIS_{year}_{month:02d}_1.zip"

    filename = file_template.format(year=year, month=month)
    file_url = base_url + filename
    file_path = os.path.join(download_dir, filename)

    if not os.path.exists(file_path):
        print(f"正在下载 {filename} ...")
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"{filename} 下载完成。")
        else:
            print(f"无法下载 {filename}，HTTP 状态码: {response.status_code}")
    else:
        print(f"{filename} 已存在，跳过下载。")
    return file_path

def extract_zip_file(zip_file_path, extract_to_dir):
    """
    解压缩 ZIP 文件。

    参数：
    - zip_file_path: ZIP 文件路径。
    - extract_to_dir: 解压缩后的文件保存目录。
    """
    if not os.path.exists(extract_to_dir):
        os.makedirs(extract_to_dir)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)
    print(f"{zip_file_path} 解压缩完成。")

def load_ais_data(data_dir):
    """
    读取指定目录下的所有 AIS CSV 数据。

    参数：
    - data_dir: AIS CSV 文件所在目录。

    返回：
    - data: 合并后的 DataFrame。
    """
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []
    for filename in all_files:
        print(f"正在读取 {filename} ...")
        df = pd.read_csv(filename)
        df_list.append(df)
    data = pd.concat(df_list, ignore_index=True)
    print("数据读取完成。")
    return data

if __name__ == "__main__":
    # 设置下载年份和月份
    year = 2020
    month = 1  # 1 表示一月

    # 设置数据保存目录
    download_dir = './data/NOAA_AIS/'
    extract_dir = os.path.join(download_dir, f"{year}_{month:02d}")

    # 下载 NOAA AIS 数据
    zip_file_path = download_noaa_ais_data(year, month, download_dir)

    # 解压缩数据
    extract_zip_file(zip_file_path, extract_dir)

    # 读取并合并数据（可选）
    ais_data = load_ais_data(extract_dir)

    # 显示数据前五行（可选）
    print(ais_data.head())

    # 您可以根据需要对数据进行进一步的处理和分析
