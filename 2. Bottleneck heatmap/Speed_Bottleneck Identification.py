#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw speed heatmaps with "low speed = red, high speed = green" for each day.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 全局绘图设置
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rc('font', family='Times New Roman')

# =========================
# 读取与预处理
# =========================
# 1) 读取 CSV（首列为时间索引）
csv_path = 'Speed.csv'
df_speed = pd.read_csv(csv_path, index_col=0, header=0)

# 2) 处理时间索引：去掉结尾 :00（若存在），再解析为 datetime
#    例如 "2017-04-28 11:30:00" -> "2017-04-28 11:30"
if isinstance(df_speed.index, pd.Index) and df_speed.index.dtype == 'object':
    df_speed.index = df_speed.index.str.replace(r':00$', '', regex=True)

# 转为时间类型（严格按 '%Y-%m-%d %H:%M' 解析）
df_speed.index = pd.to_datetime(df_speed.index, format='%Y-%m-%d %H:%M', errors='raise')

# 按时间排序（防止乱序）
df_speed = df_speed.sort_index()

# 将列名统一转成字符串，便于匹配 Abs
df_speed.columns = [str(c) for c in df_speed.columns]

# =========================
# 核心函数：画热力图（低速=红，高速=绿）
# =========================
def drow_speed_profile(df_speed, from_detector_Abs, to_detector_Abs, start_time, end_time):
    """
    根据速度数据绘制按天分割的拥堵热力图（低速=红，高速=绿）
    参数
    ----
    df_speed: DataFrame，索引为 datetime（5min 间隔），列为 detector Abs（字符串）
    from_detector_Abs, to_detector_Abs: 起止 Abs 名称（或能转换为相同字符串的数值）
    start_time, end_time: 字符串 'HH:MM'，当天内的起止时刻（5min 粒度）
    """
    Abs = list(df_speed.columns)

    # 找到起止 detector 的列索引
    from_detector_index = Abs.index(str(from_detector_Abs))
    to_detector_index   = Abs.index(str(to_detector_Abs))
    print('from_detector_index:', from_detector_index)
    print('to_detector_index  :', to_detector_index)

    # 统计天数
    total_number_of_days = pd.Series(df_speed.index.date).nunique()

    # 以 5min 间隔生成横轴标签
    time_intervals = pd.date_range(start_time, end_time, freq="5min").strftime('%H:%M')
    number_of_time_intervals = len(time_intervals)
    print('number_of_time_intervals:', number_of_time_intervals)

    # 计算当天内起止行索引（一天 288 个 5min）
    start_time_index = len(pd.date_range('00:00', start_time, freq="5min").strftime('%H:%M'))
    end_time_index   = len(pd.date_range('00:00', end_time,   freq="5min").strftime('%H:%M'))
    print('start_time_index:', start_time_index)
    print('end_time_index  :', end_time_index)

    # 列选择（支持 from > to 的降序区间）
    if from_detector_index <= to_detector_index:
        col_sel = slice(from_detector_index, to_detector_index)  # iloc 右开
    else:
        # 如果 from > to，按降序取列
        col_sel = list(range(from_detector_index, to_detector_index, -1))

    # 逐日绘图
    for d in range(1, total_number_of_days + 1):
        # 计算当日的行区间（iloc 右开）
        row_start = 288 * (d - 1) + start_time_index - 1
        row_end   = 288 * (d - 1) + end_time_index

        df_speed_temp = df_speed.iloc[row_start:row_end, col_sel].copy()

        # 保护：若切片为空，跳过
        if df_speed_temp.empty:
            print(f'[WARN] day #{d} empty slice, skip.')
            continue

        # 横轴显示 HH:MM
        df_speed_temp.index = time_intervals

        # 标题信息
        day_full = str(df_speed.iloc[row_start:row_end].index[0].date())
        day_of_week = df_speed.iloc[row_start:row_end].index[0].strftime("%A")

        # 绘图
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            df_speed_temp.T,
            vmin=20, vmax=70, center=45,   # 可根据你的速度范围调整
            cmap="RdYlGn",                 # 低值→红，中值→黄，高值→绿
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "Speed (mph)"}
        )

        # 每隔 12 个刻度（1 小时）显示一个标签
        ax.set_xticks(range(0, len(df_speed_temp.index), 12))
        ax.set_xticklabels(df_speed_temp.index[::12], fontsize=10, rotation=45)

        ax.set_ylabel('Vehicle detector locations (Abs)', fontsize=18)
        ax.set_title(f'Speed profile at I-405', fontsize=22)

        # 输出文件：按天保存，避免覆盖
        out_name = f'Speed profile.png'
        fig.savefig(out_name, dpi=500, bbox_inches='tight')
        plt.close(fig)
        print(f'[OK] Saved: {out_name}')

# =========================
# 示例调用
# =========================
if __name__ == "__main__":
    # 根据你的列名（Abs）传入起止 Abs；时间段为当天 'HH:MM'
    drow_speed_profile(
        df_speed=df_speed,
        from_detector_Abs=14.59,
        to_detector_Abs=9.42,
        start_time='11:30',
        end_time='19:40'
    )
