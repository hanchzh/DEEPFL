import os
import requests
from requests.auth import HTTPBasicAuth
import sys
import json

# 设置 Earthdata 登录凭证
username = 'your_earthdata_username'  # 请替换为您的 Earthdata 用户名
password = 'your_earthdata_password'  # 请替换为您的 Earthdata 密码

# 定义搜索参数
short_name = 'SENTINEL-1A_SAR_GRD'  # Sentinel-1A SAR 地面检测数据
version = '1'  # 数据版本，可能需要根据实际情况调整

# 时间范围（ISO 8601 格式）
time_start = '2020-01-01T00:00:00Z'
time_end = '2020-01-02T23:59:59Z'

# 定义感兴趣的区域（使用经纬度范围）
bounding_box = '-80.0,20.0,-70.0,30.0'  # [西经, 南纬, 东经, 北纬]

# 设置本地下载目录
download_dir = './data/NASA_Maritime/'
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

def build_cmr_query_url(short_name, version, time_start, time_end, bounding_box):
    """
    构建 CMR API 请求并获取查询结果。
    """
    cmr_url = 'https://cmr.earthdata.nasa.gov/search/granules'
    params = {
        'short_name': short_name,
        'version': version,
        'temporal': f'{time_start},{time_end}',
        'bounding_box': bounding_box,
        'page_size': 2000,
        'page_num': 1,
        'provider': 'ASF',
        'sort_key': '-start_date',
        'echo_compatible': 'true',
        'format': 'json'
    }
    response = requests.get(cmr_url, params=params)
    if response.status_code != 200:
        print(f"CMR 查询失败，HTTP 状态码: {response.status_code}")
        sys.exit(1)
    results = response.json()
    return results

def get_download_urls(results):
    """
    从查询结果中提取下载链接。
    """
    download_urls = []
    if 'feed' in results and 'entry' in results['feed']:
        for entry in results['feed']['entry']:
            for link in entry.get('links', []):
                if 'data#' in link.get('rel', '') and link.get('href', '').startswith('https'):
                    download_urls.append(link['href'])
    return download_urls

def download_data(download_urls, download_dir):
    """
    下载数据文件并保存到指定目录。
    """
    session = requests.Session()
    session.auth = (username, password)
    for url in download_urls:
        filename = os.path.basename(url)
        file_path = os.path.join(download_dir, filename)
        if not os.path.exists(file_path):
            print(f"正在下载 {filename} ...")
            response = session.get(url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"{filename} 下载完成。")
            else:
                print(f"无法下载 {filename}，HTTP 状态码: {response.status_code}")
        else:
            print(f"{filename} 已存在，跳过下载。")

if __name__ == "__main__":
    # 构建查询并获取结果
    results = build_cmr_query_url(short_name, version, time_start, time_end, bounding_box)
    # 提取下载链接
    download_urls = get_download_urls(results)
    print(f"找到 {len(download_urls)} 个数据文件。")
    # 下载数据
    if download_urls:
        download_data(download_urls, download_dir)
    else:
        print("未找到符合条件的数据文件。")
