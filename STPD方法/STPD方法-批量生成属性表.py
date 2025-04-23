import csv
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import os
from STPD import STPD
from TPTR import TPTR

# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # CSV文件路径，包含时间和位移数据
    csvpath = r"STPD方法-示例.csv"

    # 读取CSV文件
    with open(csvpath, 'r') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        header = next(reader)  # 提取标题行
        t = []  # 时间
        displacements = {col: [] for col in header[1:]}  # 创建字典，用于存储每一列的位移数据

        for row in reader:
            try:
                # 检查行数据是否包含有效的时间和位移数据
                time_value = float(row[0])  # 尝试将时间值转换为浮动数
                t.append(time_value)  # 如果转换成功，将时间值添加到时间列表中

                # 处理位移列
                for i, col in enumerate(header[1:], 1):  # 从1开始，跳过“时间”列
                    displacement_value = float(row[i])  # 尝试将每个位移值转换为浮动数
                    displacements[col].append(displacement_value)
            except ValueError:
                # 如果出现ValueError（如数据为空或非数字），跳过当前行
                continue

    t = np.array(t)

    # 循环处理每一列位移数据
    for col, f in displacements.items():
        f = np.array(f)  # 转换为numpy数组

        # 确保时间和位移数据长度一致
        if len(t) != len(f):
            print(f"警告：时间和{col}列的长度不匹配，跳过此列。")
            continue  # 如果长度不匹配，跳过此列

        # 动态设置STPD的size参数，确保它在有效范围内
        size = min(31, len(f))  # 设置size为31或当前序列长度中较小的一个

        # 对每一列位移数据应用STPD和TPTR方法
        TPs = STPD(t, f, size=size, step=12, SNR=1, NDRI=0.3, dir_th=0, tp_th=1, margin=12, alpha=0.01)
        stats, y = TPTR(t, f, TPs)

        # 将TP结果保存到结果文件夹中的文本文件
        result_folder = './Result_TP'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        with open(f'{result_folder}/Result-{col}.txt', "w") as log:
            log.write("TP         slope        direction         SNR           |NDRI|\n")
            [log.write(f"{int(stats[k][0]):<6}{stats[k][1]:>10}{stats[k][2]:>15}{stats[k][3]:>15}{stats[k][4]:>15}\n")
             for k in range(len(TPs))]

        # 为当前位移列绘制并保存结果图像
        output_folder = './Displacement_Images'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plt.plot(t, f, '-ok', label='Time Series', linewidth=1, markersize=3)
        plt.plot(t, y, 'b', label='Linear trend')
        if len(TPs) > 0:
            plt.plot(t[np.array(TPs)], y[np.array(TPs)], 'db', label='Turning point')

        # 在图表中移除“Displacement-1”标题
        for k in range(len(TPs)):
            C = f"SNR = {np.round(stats[k][3], 2)}\nNDRI = {np.round(stats[k][4], 2)}"
            plt.text(0.975, 0.95, C, fontsize=10, ha='right', va='top', transform=plt.gca().transAxes,
                     bbox=dict(facecolor='b', alpha=0.1))  # 使用归一化坐标（1, 1）表示右上角

        plt.xlabel('Time (year)')
        plt.ylabel('Displacement (mm)')
        plt.grid(True)

        # 自定义x轴刻度（年份）
        plt.xticks([0, 1, 2], ['2019', '2020', '2021'])
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
                   mode="expand", borderaxespad=0)

        fig = plt.gcf()
        fig.set_size_inches(8, 4)

        # 保存当前位移列的图像
        plt.savefig(f"{output_folder}/Displacement-{col}.png", dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，防止循环中绘图重叠

    print("所有图表和结果已保存。")
