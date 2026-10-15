import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 物理常数
g = 9.8  # 重力加速度
smoke_radius = 10  # 烟幕半径
smoke_sink_speed = 3  # 烟幕下沉速度
missile_speed = 300  # 导弹速度

# 初始位置
FY1_initial = np.array([17800, 0, 1800])
M1_initial = np.array([20000, 0, 2000])
target_point = np.array([0, 200, 0])  # 真目标
fake_target = np.array([0, 0, 0])  # 假目标


def calculate_missile_direction():
    """计算导弹飞行方向（指向假目标）"""
    direction = fake_target - M1_initial
    return direction / np.linalg.norm(direction)


missile_direction = calculate_missile_direction()


def missile_position(t):
    return M1_initial + missile_speed * t * missile_direction


def calculate_smoke_burst_single(v_u, theta_deg, t1, t2):
    theta = np.deg2rad(theta_deg)
    uav_velocity = v_u * np.array([np.cos(theta), np.sin(theta), 0])
    drop_position = FY1_initial + uav_velocity * t1
    horizontal_displacement = uav_velocity * t2
    vertical_displacement = 0.5 * g * t2 ** 2
    burst_position = drop_position + horizontal_displacement - np.array([0, 0, vertical_displacement])
    burst_time = t1 + t2
    return burst_position, burst_time


def calculate_cover_quality_improved(M, S, P):
    try:
        MS = S - M
        MP = P - M
        distance_MS = np.linalg.norm(MS)

        if distance_MS < smoke_radius:
            return 1.0 - (distance_MS / smoke_radius) ** 2

        if distance_MS < 1e-12 or np.linalg.norm(MP) < 1e-12:
            return 0.0

        cos_beta = np.dot(MP, MS) / (np.linalg.norm(MP) * distance_MS)
        critical_angle = np.arcsin(smoke_radius / distance_MS)
        cos_critical = np.cos(critical_angle)

        if cos_beta >= cos_critical:
            relative_position = (cos_beta - cos_critical) / (1 - cos_critical + 1e-10)
            quality = np.sqrt(relative_position)
        else:
            falloff = 1.0 - (cos_critical - cos_beta) / (cos_critical + 1.0)
            quality = max(0, 0.2 * falloff ** 3)

        return min(1.0, max(0.0, quality))
    except:
        return 0.0


def calculate_single_smoke_cover_time(v_u, theta_deg, t1, t2, quality_threshold=0.1):
    burst_pos, burst_time = calculate_smoke_burst_single(v_u, theta_deg, t1, t2)

    total_time = 0
    dt = 0.01
    t_current = burst_time
    max_simulation_time = 30  # 最大仿真时间

    while t_current < burst_time + max_simulation_time:
        M_t = missile_position(t_current)
        sink_distance = smoke_sink_speed * (t_current - burst_time)
        S_t = burst_pos - np.array([0, 0, sink_distance])

        if S_t[2] < 0:  # 烟幕已经沉到地面以下
            break

        quality = calculate_cover_quality_improved(M_t, S_t, target_point)

        if quality > quality_threshold:
            total_time += dt

        t_current += dt

    return total_time


def check_smoke_constraints(t1_params):
    t1_sorted = np.sort(t1_params)
    penalty = 0

    for i in range(len(t1_sorted) - 1):
        time_gap = t1_sorted[i + 1] - t1_sorted[i]
        if time_gap < 1.0:
            penalty += 1000 * (1.0 - time_gap) ** 2

    return penalty


def check_physical_constraints(params):
    v_u, theta_deg, t1_1, t2_1, t1_2, t2_2, t1_3, t2_3 = params
    penalty = 0

    for t2 in [t2_1, t2_2, t2_3]:
        if t2 > 8:  # 如果起爆时间超过8秒，惩罚
            penalty += 100 * (t2 - 8) ** 2

    # 检查投放时间顺序
    t1_params = np.array([t1_1, t1_2, t1_3])
    t1_sorted = np.sort(t1_params)
    for i in range(len(t1_sorted) - 1):
        if t1_sorted[i + 1] - t1_sorted[i] < 1.0:
            penalty += 500

    return penalty


def objective_function_all_params(params):
    v_u, theta_deg, t1_1, t2_1, t1_2, t2_2, t1_3, t2_3 = params

    # 检查参数边界
    if not (70 <= v_u <= 140):
        return 10000 + abs(v_u - 70 if v_u < 70 else v_u - 140) * 100

    # 检查物理约束
    physical_penalty = check_physical_constraints(params)

    # 如果严重违反约束，直接返回大惩罚值
    if physical_penalty > 100:
        return physical_penalty

    bursts = []
    for t1, t2 in [(t1_1, t2_1), (t1_2, t2_2), (t1_3, t2_3)]:
        if t1 < 0 or t1 > 30 or t2 < 0 or t2 > 20:
            return 10000  # 参数越界

        burst_pos, burst_time = calculate_smoke_burst_single(v_u, theta_deg, t1, t2)
        bursts.append((burst_pos, burst_time))

    total_integral = 0
    max_dt = 0.02
    min_dt = 0.005
    t_start = min(burst_time for _, burst_time in bursts)
    t_end = max(burst_time for _, burst_time in bursts) + 30
    t_current = t_start
    prev_quality = 0
    iteration_count = 0
    max_iterations = 3000

    while t_current < t_end and iteration_count < max_iterations:
        dt = max(min_dt, min(max_dt, 0.1 / (abs(prev_quality - 0.5) + 0.1)))
        M_t = missile_position(t_current)
        max_quality = 0

        for burst_pos, burst_time in bursts:
            sink_distance = smoke_sink_speed * (t_current - burst_time)
            S_t = burst_pos - np.array([0, 0, sink_distance])
            if S_t[2] < 0:
                continue
            quality = calculate_cover_quality_improved(M_t, S_t, target_point)
            max_quality = max(max_quality, quality)

        total_integral += max_quality * dt
        prev_quality = max_quality
        t_current += dt
        iteration_count += 1

    return -total_integral + physical_penalty


def efficient_optimization():
    bounds = [
        (100, 140),  # v_u
        (170, 190),  # theta_deg
        (0, 10), (1, 6),
        (0, 10), (1, 6),
        (0, 10), (1, 6)
    ]

    initial_guess = [120.0, 180.0, 1.5, 3.0, 2.5, 3.5, 3.5, 4.0]

    result_de = differential_evolution(
        objective_function_all_params,
        bounds,
        strategy='best1bin',
        maxiter=50,
        popsize=15,
        tol=1e-4,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=True,
        polish=False
    )

    print(f"差分进化结果: {-result_de.fun:.6f}")

    result_local = minimize(
        objective_function_all_params,
        result_de.x,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 100,
            'ftol': 1e-6,
            'gtol': 1e-6,
            'disp': True
        }
    )

    best_params = result_local.x
    best_value = -result_local.fun

    # 解析最优参数
    v_u, theta_deg = best_params[0], best_params[1]
    smoke_params = best_params[2:].reshape(3, 2)
    t1_params = smoke_params[:, 0]

    # 检查约束
    time_gaps = np.diff(np.sort(t1_params))
    min_gap = np.min(time_gaps) if len(time_gaps) > 0 else 0

    print("\n" + "=" * 50)
    print("=== 最优策略 ===")
    print(f"无人机速度: {v_u:.6f} m/s")
    print(f"无人机航向: {theta_deg:.6f}°")
    print(f"总遮蔽积分: {best_value:.6f}")
    print(f"最小投放间隔: {min_gap:.6f}s (要求≥1.0s)")

    individual_cover_times = []
    for i, (t1, t2) in enumerate(smoke_params):
        cover_time = calculate_single_smoke_cover_time(v_u, theta_deg, t1, t2)
        individual_cover_times.append(cover_time)
        print(f"烟幕弹 {i + 1} 单独遮蔽时间: {cover_time:.3f}s")

    # 输出烟幕弹详细参数
    results = []
    for i, (t1, t2) in enumerate(smoke_params):
        burst_pos, burst_time = calculate_smoke_burst_single(v_u, theta_deg, t1, t2)

        fall_distance = 0.5 * g * t2 ** 2
        initial_height = FY1_initial[2]  # 无人机初始高度

        print(f"  烟幕弹 {i + 1}:")
        print(f"    • 投放时间: {t1:.6f}s")
        print(f"    • 起爆时间: {t2:.6f}s")
        print(f"    • 爆开时间: {burst_time:.6f}s")
        print(f"    • 爆开位置: ({burst_pos[0]:.2f}, {burst_pos[1]:.2f}, {burst_pos[2]:.2f})")
        print(f"    • 下落距离: {fall_distance:.2f}m")
        print(f"    • 初始高度: {initial_height:.2f}m")
        print(f"    • 单独遮蔽时间: {individual_cover_times[i]:.3f}s")

        results.append({
            '烟幕弹序号': i + 1,
            '投放时间(s)': t1,
            '起爆时间(s)': t2,
            '爆开时间(s)': burst_time,
            '爆开X坐标(m)': burst_pos[0],
            '爆开Y坐标(m)': burst_pos[1],
            '爆开Z坐标(m)': burst_pos[2],
            '下落距离(m)': fall_distance,
            '单独遮蔽时间(s)': individual_cover_times[i]
        })

    # 保存结果
    df = pd.DataFrame(results)
    df.to_excel('result3_optimal.xlsx', index=False)
    print("\n结果已保存到 result3_optimal.xlsx")


    return best_params, best_value, individual_cover_times


def validate_and_visualize_all_smokes(v_u, theta_deg, smoke_params, individual_cover_times):
    smoke_params = smoke_params.reshape(3, 2)
    bursts = []
    for i, (t1, t2) in enumerate(smoke_params):
        burst_pos, burst_time = calculate_smoke_burst_single(v_u, theta_deg, t1, t2)
        bursts.append((burst_pos, burst_time))
        print(
            f"烟幕弹 {i + 1}: 投放时间={t1:.3f}s, 起爆时间={t2:.3f}s, 爆开位置=({burst_pos[0]:.1f}, {burst_pos[1]:.1f}, {burst_pos[2]:.1f})")

    burst_times = [burst_time for _, burst_time in bursts]
    start_time = min(burst_times)
    end_time = max(burst_times) + 30
    times = np.linspace(start_time, end_time, 2000)

    all_qualities = []
    missile_positions = []
    smoke_positions_all = [[] for _ in range(3)]

    for t in times:
        M_t = missile_position(t)
        missile_positions.append(M_t)

        max_quality = 0
        for i, (burst_pos, burst_time) in enumerate(bursts):
            sink_distance = smoke_sink_speed * (t - burst_time)
            S_t = burst_pos - np.array([0, 0, sink_distance])
            smoke_positions_all[i].append(S_t)

            if S_t[2] >= 0:  # 只考虑地面以上的烟幕
                quality = calculate_cover_quality_improved(M_t, S_t, target_point)
                max_quality = max(max_quality, quality)

        all_qualities.append(max_quality)

    # 绘制综合结果
    fig = plt.figure(figsize=(16, 12))

    # 颜色定义（修复画图错误）
    colors = ['r', 'g', 'b']  # 使用单字符颜色代码

    # 1. 遮蔽质量随时间变化
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(times - start_time, all_qualities, 'b-', linewidth=2)
    plt.xlabel('时间 (s)')
    plt.ylabel('遮蔽质量')
    plt.title('三枚烟幕弹综合遮蔽质量随时间变化')
    plt.grid(True, alpha=0.3)

    # 标记各烟幕弹爆开时间
    for i, burst_time in enumerate(burst_times):
        plt.axvline(x=burst_time - start_time, color=colors[i], linestyle='--',
                    label=f'烟幕{i + 1}爆开: {burst_time - start_time:.1f}s')
    plt.legend()

    # 2. 导弹和烟幕轨迹（XY平面）
    ax2 = plt.subplot(2, 2, 2)
    missile_x = [pos[0] for pos in missile_positions]
    missile_y = [pos[1] for pos in missile_positions]
    plt.plot(missile_x, missile_y, 'r-', label='导弹轨迹', linewidth=2)

    for i, smoke_positions in enumerate(smoke_positions_all):
        smoke_x = [pos[0] for pos in smoke_positions]
        smoke_y = [pos[1] for pos in smoke_positions]
        plt.plot(smoke_x, smoke_y, color=colors[i], linestyle='--',
                 label=f'烟幕{i + 1}轨迹', linewidth=2, alpha=0.7)

    plt.scatter(target_point[0], target_point[1], c='orange', s=200, label='目标', zorder=5)
    for i, (burst_pos, _) in enumerate(bursts):
        plt.scatter(burst_pos[0], burst_pos[1], color=colors[i], s=100,
                    label=f'烟幕{i + 1}爆开位置', zorder=5)

    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.title('导弹和烟幕轨迹（XY平面）')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 高度变化
    ax3 = plt.subplot(2, 2, 3)
    altitudes = [pos[2] for pos in missile_positions]
    plt.plot(times - start_time, altitudes, 'r-', label='导弹高度', linewidth=2)

    for i, smoke_positions in enumerate(smoke_positions_all):
        smoke_altitudes = [pos[2] for pos in smoke_positions]
        plt.plot(times - start_time, smoke_altitudes, color=colors[i], linestyle='--',
                 label=f'烟幕{i + 1}高度', linewidth=2, alpha=0.7)

    plt.xlabel('时间 (s)')
    plt.ylabel('高度 (m)')
    plt.title('高度变化')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. 单独遮蔽时间分布
    ax4 = plt.subplot(2, 2, 4)
    plt.bar(range(1, 4), individual_cover_times, color=colors, alpha=0.7)
    plt.xlabel('烟幕弹序号')
    plt.ylabel('遮蔽时间 (s)')
    plt.title('各烟幕弹单独遮蔽时间')
    for i, time in enumerate(individual_cover_times):
        plt.text(i + 1, time + 0.1, f'{time:.1f}s', ha='center', va='bottom', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('three_smokes_cover_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 计算总有效遮蔽时间
    effective_qualities = [q for q in all_qualities if q > 0.1]
    total_effective_time = len(effective_qualities) * (times[1] - times[0])

    print(f"\n总有效遮蔽时间 (>0.1质量阈值): {total_effective_time:.3f}s")
    return total_effective_time


if __name__ == "__main__":
    best_params, best_value, individual_cover_times = efficient_optimization()

    v_u, theta_deg = best_params[0], best_params[1]
    smoke_params = best_params[2:]
    total_effective_time = validate_and_visualize_all_smokes(v_u, theta_deg, smoke_params, individual_cover_times)

    print(f"无人机速度: {v_u:.6f} m/s")
    print(f"无人机航向: {theta_deg:.6f}°")
    print(f"总遮蔽积分: {best_value:.6f}")
    print(f"总有效遮蔽时间: {total_effective_time:.3f}s")
    print(f"各烟幕弹单独遮蔽时间: {individual_cover_times}")