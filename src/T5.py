import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numba
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

g = 9.8
smoke_radius = 10
smoke_sink_speed = 3
missile_speed = 300

target_point = np.array([0, 200, 0])
fake_target = np.array([0, 0, 0])

missiles_initial = np.array([
    [20000, 0, 2000],  # M1
    [19000, 600, 2100],  # M2
    [18000, -600, 1900]  # M3
])

uavs_initial = np.array([
    [17800, 0, 1800],  # FY1
    [12000, 1400, 1400],  # FY2
    [6000, -3000, 700],  # FY3
    [11000, 2000, 1800],  # FY4
    [13000, -2000, 1300]  # FY5
])

missile_directions = np.zeros((3, 3))
for i in range(3):
    direction = fake_target - missiles_initial[i]
    missile_directions[i] = direction / np.linalg.norm(direction)


@numba.njit(fastmath=True)
def calculate_missile_position_numba(missile_idx, t):
    return missiles_initial[missile_idx] + missile_speed * t * missile_directions[missile_idx]


@numba.njit(fastmath=True)
def calculate_cover_quality_numba(M, S, P, smoke_radius):
    try:
        MS = S - M
        MP = P - M
        distance_MS = np.linalg.norm(MS)

        if distance_MS < smoke_radius:
            return 1.0 - (distance_MS / smoke_radius) ** 2

        distance_MP = np.linalg.norm(MP)
        if distance_MS < 1e-12 or distance_MP < 1e-12:
            return 0.0

        cos_beta = np.dot(MP, MS) / (distance_MP * distance_MS)
        critical_angle = np.arcsin(min(1.0, smoke_radius / distance_MS))
        cos_critical = np.cos(critical_angle)

        if cos_beta >= cos_critical:
            relative_position = (cos_beta - cos_critical) / (1 - cos_critical + 1e-10)
            quality = np.sqrt(relative_position)
        else:
            falloff = 1.0 - (cos_critical - cos_beta) / (cos_critical + 1.0)
            quality = max(0.0, 0.2 * falloff ** 3)

        return min(1.0, max(0.0, quality))
    except:
        return 0.0


def calculate_smoke_burst(uav_idx, params):
    v_u, theta_deg, t1, t2 = params
    theta = np.deg2rad(theta_deg)
    uav_velocity = v_u * np.array([np.cos(theta), np.sin(theta), 0])

    drop_position = uavs_initial[uav_idx] + uav_velocity * t1
    fall_time = t2
    horizontal_displacement = uav_velocity * fall_time
    vertical_displacement = 0.5 * g * fall_time ** 2
    burst_position = drop_position + horizontal_displacement - np.array([0, 0, vertical_displacement])
    burst_time = t1 + fall_time

    return burst_position, burst_time


def objective_function_multi(params, uav_smoke_assignments):
    total_integral = 0
    param_idx = 0
    smoke_info = []

    for uav_idx, num_smokes in enumerate(uav_smoke_assignments):
        for _ in range(num_smokes):
            v_u = params[param_idx]
            theta_deg = params[param_idx + 1]
            t1 = params[param_idx + 2]
            t2 = params[param_idx + 3]
            param_idx += 4

            burst_pos, burst_time = calculate_smoke_burst(uav_idx, [v_u, theta_deg, t1, t2])
            end_time = burst_time + 20
            smoke_info.append((uav_idx, burst_pos, burst_time, end_time))

    if not smoke_info:
        return 0.0

    all_start_times = [info[2] for info in smoke_info]
    all_end_times = [info[3] for info in smoke_info]
    start_time = max(0, min(all_start_times) - 1)
    end_time = max(all_end_times) + 1
    time_steps = np.arange(start_time, end_time, 0.5)

    for t in time_steps:
        max_quality = np.zeros(3)

        for (uav_idx, burst_pos, burst_time, end_time) in smoke_info:
            if t < burst_time or t > end_time:
                continue

            sink_distance = smoke_sink_speed * (t - burst_time)
            S_t = burst_pos - np.array([0, 0, sink_distance])
            if S_t[2] < 0:
                continue

            for missile_idx in range(3):
                M_t = calculate_missile_position_numba(missile_idx, t)
                quality = calculate_cover_quality_numba(M_t, S_t, target_point, smoke_radius)
                if quality > max_quality[missile_idx]:
                    max_quality[missile_idx] = quality

        total_integral += np.sum(max_quality) * 0.5

    return -total_integral


def staged_optimization(uav_smoke_assignments):
    """分阶段优化：先优化投放时间，再优化所有参数"""
    print("=== 第一阶段：优化投放和起爆时间 ===")

    # 第一阶段：只优化投放和起爆时间
    bounds_stage1 = []
    initial_guess_stage1 = []

    for uav_idx, num_smokes in enumerate(uav_smoke_assignments):
        for _ in range(num_smokes):
            # 只优化t1和t2，固定速度和航向
            bounds_stage1.extend([(0, 10), (0, 20)])
            initial_guess_stage1.extend([2.0, 6.0])

    # 使用固定速度和航向
    fixed_params = []
    for uav_idx, num_smokes in enumerate(uav_smoke_assignments):
        for _ in range(num_smokes):
            # 为每架无人机设置合理的初始速度和航向
            if uav_idx == 0:  # FY1
                fixed_params.extend([120, 180])
            elif uav_idx == 1:  # FY2
                fixed_params.extend([110, 190])
            elif uav_idx == 2:  # FY3
                fixed_params.extend([130, 170])
            elif uav_idx == 3:  # FY4
                fixed_params.extend([125, 200])
            else:  # FY5
                fixed_params.extend([115, 160])

    def stage1_objective(time_params):
        full_params = []
        time_idx = 0
        param_idx = 0
        for uav_idx, num_smokes in enumerate(uav_smoke_assignments):
            for _ in range(num_smokes):
                full_params.extend([fixed_params[param_idx], fixed_params[param_idx + 1],
                                    time_params[time_idx], time_params[time_idx + 1]])
                time_idx += 2
                param_idx += 2
        return objective_function_multi(full_params, uav_smoke_assignments)

    # 第一阶段优化
    result_stage1 = differential_evolution(
        stage1_objective,
        bounds_stage1,
        maxiter=50,
        popsize=10,
        mutation=(0.7, 1.0),
        recombination=0.8,
        tol=0.1,
        disp=True
    )

    print(f"第一阶段完成，最优值: {-result_stage1.fun:.2f}")

    # 第二阶段：使用第一阶段结果作为初始值，优化所有参数
    print("\n=== 第二阶段：优化所有参数 ===")

    # 构建第二阶段的初始猜测
    initial_guess_stage2 = []
    time_idx = 0
    param_idx = 0
    for uav_idx, num_smokes in enumerate(uav_smoke_assignments):
        for _ in range(num_smokes):
            initial_guess_stage2.extend([
                fixed_params[param_idx],
                fixed_params[param_idx + 1],
                result_stage1.x[time_idx],
                result_stage1.x[time_idx + 1]
            ])
            time_idx += 2
            param_idx += 2

    # 完整参数边界
    bounds_full = []
    for _ in range(sum(uav_smoke_assignments)):
        bounds_full.extend([(70, 140), (0, 360), (0, 10), (0, 20)])

    # 第二阶段优化
    result_full = differential_evolution(
        lambda x: objective_function_multi(x, uav_smoke_assignments),
        bounds_full,
        x0=initial_guess_stage2,
        maxiter=100,
        popsize=12,
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=0.05,
        disp=True,
        polish=True,
        strategy='best1bin'
    )

    return result_full


def print_detailed_results(result, uav_smoke_assignments):
    """详细输出所有参数"""
    print("\n" + "=" * 80)
    print("优化结果详细参数")
    print("=" * 80)

    param_idx = 0
    for uav_idx, num_smokes in enumerate(uav_smoke_assignments):
        uav_name = f'FY{uav_idx + 1}'
        print(f"\n{uav_name} 投放策略:")
        print("-" * 50)

        for smoke_idx in range(num_smokes):
            v_u = result.x[param_idx]
            theta_deg = result.x[param_idx + 1]
            t1 = result.x[param_idx + 2]
            t2 = result.x[param_idx + 3]

            burst_pos, burst_time = calculate_smoke_burst(uav_idx, [v_u, theta_deg, t1, t2])

            print(f"烟幕弹 {smoke_idx + 1}:")
            print(f"  飞行速度: {v_u:.2f} m/s")
            print(f"  航向角: {theta_deg:.2f}°")
            print(f"  投放时间: {t1:.2f} s")
            print(f"  起爆延迟: {t2:.2f} s")
            print(f"  爆开时间: {burst_time:.2f} s")
            print(f"  爆开位置: ({burst_pos[0]:.2f}, {burst_pos[1]:.2f}, {burst_pos[2]:.2f}) m")
            print()

            param_idx += 4


def visualize_scenario(result, uav_smoke_assignments):
    """可视化无人机、导弹和烟幕位置"""
    fig = plt.figure(figsize=(15, 10))

    ax1 = fig.add_subplot(121, projection='3d')

    uav_colors = ['red', 'blue', 'green', 'orange', 'purple']
    uav_names = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']

    for i, (uav_pos, color, name) in enumerate(zip(uavs_initial, uav_colors, uav_names)):
        ax1.scatter(uav_pos[0], uav_pos[1], uav_pos[2], c=color, s=100, label=name, marker='^')

    missile_colors = ['darkred', 'darkblue', 'darkgreen']
    missile_names = ['M1', 'M2', 'M3']

    for i, (missile_pos, color, name) in enumerate(zip(missiles_initial, missile_colors, missile_names)):
        ax1.scatter(missile_pos[0], missile_pos[1], missile_pos[2], c=color, s=100, label=name, marker='o')

        trajectory = np.array([missile_pos, fake_target])
        ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                 c=color, linestyle='--', alpha=0.7)

    ax1.scatter(target_point[0], target_point[1], target_point[2],
                c='gold', s=200, label='真目标', marker='*')
    ax1.scatter(fake_target[0], fake_target[1], fake_target[2],
                c='black', s=200, label='假目标', marker='x')

    param_idx = 0
    smoke_positions = []
    for uav_idx, num_smokes in enumerate(uav_smoke_assignments):
        for _ in range(num_smokes):
            v_u = result.x[param_idx]
            theta_deg = result.x[param_idx + 1]
            t1 = result.x[param_idx + 2]
            t2 = result.x[param_idx + 3]

            burst_pos, _ = calculate_smoke_burst(uav_idx, [v_u, theta_deg, t1, t2])
            smoke_positions.append(burst_pos)
            ax1.scatter(burst_pos[0], burst_pos[1], burst_pos[2],
                        c=uav_colors[uav_idx], s=80, marker='s', alpha=0.7)

            param_idx += 4

    ax1.set_xlabel('X坐标 (m)')
    ax1.set_ylabel('Y坐标 (m)')
    ax1.set_zlabel('Z坐标 (m)')
    ax1.set_title('3D场景可视化')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax2 = fig.add_subplot(122)

    for i, (uav_pos, color, name) in enumerate(zip(uavs_initial, uav_colors, uav_names)):
        ax2.scatter(uav_pos[0], uav_pos[1], c=color, s=100, label=name, marker='^')

    for i, (missile_pos, color, name) in enumerate(zip(missiles_initial, missile_colors, missile_names)):
        ax2.scatter(missile_pos[0], missile_pos[1], c=color, s=100, label=name, marker='o')
        trajectory = np.array([missile_pos, fake_target])
        ax2.plot(trajectory[:, 0], trajectory[:, 1], c=color, linestyle='--', alpha=0.7)

    ax2.scatter(target_point[0], target_point[1], c='gold', s=200, label='真目标', marker='*')
    ax2.scatter(fake_target[0], fake_target[1], c='black', s=200, label='假目标', marker='x')

    for i, smoke_pos in enumerate(smoke_positions):
        ax2.scatter(smoke_pos[0], smoke_pos[1], c='magenta', s=60, marker='s', alpha=0.7)

    ax2.set_xlabel('X坐标 (m)')
    ax2.set_ylabel('Y坐标 (m)')
    ax2.set_title('2D平面投影')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scenario_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_results_to_excel(result, uav_smoke_assignments, filename):
    data = []
    param_idx = 0

    for uav_idx, num_smokes in enumerate(uav_smoke_assignments):
        uav_name = f'FY{uav_idx + 1}'
        for smoke_idx in range(num_smokes):
            v_u = result.x[param_idx]
            theta_deg = result.x[param_idx + 1]
            t1 = result.x[param_idx + 2]
            t2 = result.x[param_idx + 3]
            param_idx += 4

            burst_pos, burst_time = calculate_smoke_burst(uav_idx, [v_u, theta_deg, t1, t2])

            data.append({
                '无人机编号': uav_name,
                '烟幕弹编号': smoke_idx + 1,
                '飞行速度(m/s)': round(v_u, 2),
                '航向角(度)': round(theta_deg, 2),
                '投放时间(s)': round(t1, 2),
                '起爆时间(s)': round(t2, 2),
                '爆开时间(s)': round(burst_time, 2),
                '爆开位置X(m)': round(burst_pos[0], 2),
                '爆开位置Y(m)': round(burst_pos[1], 2),
                '爆开位置Z(m)': round(burst_pos[2], 2)
            })

    pd.DataFrame(data).to_excel(filename, index=False)
    print(f"结果已保存到 {filename}")


def main():
    uav_smoke_assignments = [2, 2, 2, 2, 2]

    print("=== 多无人机多烟幕弹优化开始（分阶段优化）===")
    start_time = datetime.now()


    result = staged_optimization(uav_smoke_assignments)

    print(f"\n优化耗时: {(datetime.now() - start_time).total_seconds():.2f}秒")

    if result.success:
        print(f"优化成功！最优遮蔽积分: {-result.fun:.2f}")
        print_detailed_results(result, uav_smoke_assignments)


        visualize_scenario(result, uav_smoke_assignments)


        save_results_to_excel(result, uav_smoke_assignments, 'result3.xlsx')
    else:
        print("优化失败，尝试减少烟幕弹数量或调整参数")


if __name__ == "__main__":
    main()