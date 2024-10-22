import GPUtil
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path = '/home/pandas/snap/code/RapidOcr/ocr/font/SimHei.ttf'  # 替换为你的simhei.ttf文件的实际路径
font_prop = FontProperties(fname=font_path)
def get_gpu_memory_info():
    gpus = GPUtil.getGPUs()
    gpu_memory_info = []
    for gpu in gpus:
        used_memory = gpu.memoryUsed  # 显存使用量
        total_memory = gpu.memoryTotal  # 显存总量
        gpu_memory_info.append({'used': used_memory, 'total': total_memory})
    return gpu_memory_info
def create_pie_chart():
    """
        创建饼图的函数。

        参数:
        data -- 饼图的数据，一个包含数值的列表。
        labels -- 与数据对应的标签列表。
        title -- 饼图的标题。
        """
    # 如果没有提供标签，则使用数据索引作为标签
    gpu_memory_info = get_gpu_memory_info()
    used_gpu = gpu_memory_info[0]['used']
    total_gpu = gpu_memory_info[0]['total']

    used_gpu = used_gpu / total_gpu
    no_used_gpu = 1-used_gpu
    data = [used_gpu, no_used_gpu]
    labels = ['In use', 'Unused']
    if labels is None:
        labels = list(range(len(data)))

    # 定义颜色和突出显示的部分
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'purple']
    explode = [0.1] + [0] * (len(data) - 1)  # 突出显示第一个切片

    # 绘制饼图
    plt.figure(figsize=(8, 6))  # 设置图形大小
    plt.pie(data, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    # 确保饼图是圆形的
    plt.axis('equal')

    # 添加标题
    plt.title('gpu使用情况', fontproperties=font_prop)

    # 显示图表
    # plt.show()
    return plt