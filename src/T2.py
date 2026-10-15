import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

g = 9.8
smoke_radius = 10  # 烟幕半径
smoke_sink_speed = 3  # 烟幕下沉速度
missile_speed = 300  # 导弹速度

FY1_initial = np.array([17800, 0, 1800])
M1_initial = np.array([20000, 0, 2000])
target_point = np.array([0, 200, 0])  # 真
fake_target = np.array([0, 0, 0])  # 假


def calculate_missile_direction():
    direction = fake_target - M1_initial
    return direction / np.linalg.norm(direction)
missile_direction = calculate_missile_direction()

def missile_position(t):
    return M1_initial + missile_speed * t * missile_direction


def calculate_smoke_burst(params):
    v_u, theta_deg, t1, t2 = params
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

        # 导弹在烟幕内
        if distance_MS < smoke_radius:
            return 1.0 - (distance_MS / smoke_radius) ** 2  # 中心区域质量更高

        cos_beta = np.dot(MP, MS) / (np.linalg.norm(MP) * distance_MS)

        critical_angle = np.arcsin(smoke_radius / distance_MS)
        cos_critical = np.cos(critical_angle)

        if cos_beta >= cos_critical:
            relative_position = (cos_beta - cos_critical) / (1 - cos_critical + 1e-10)
            quality = np.sqrt(relative_position)
        else:
            falloff = 1.0 - (cos_critical - cos_beta) / (cos_critical + 1.0)
            quality = max(0, 0.2 * falloff ** 3)  # 快速衰减

        return min(1.0, max(0.0, quality))

    except:
        return 0.0


def objective_function_improved(params):
    try:
        v_u, theta_deg, t1, t2 = params

        # 更严格的参数边界检查
        if not (70 <= v_u <= 140 and 0 <= theta_deg <= 360 and
                0 <= t1 <= 10 and 0 <= t2 <= 20 and t1 + t2 <= 30):
            return 10000 + np.sum(np.abs(np.clip(params, [70, 0, 0, 0], [140, 360, 10, 20]) - params))

        theta = np.deg2rad(theta_deg)

        uav_velocity = v_u * np.array([np.cos(theta), np.sin(theta), 0])

        drop_position = FY1_initial + uav_velocity * t1
        horizontal_displacement = uav_velocity * t2
        vertical_displacement = 0.5 * g * t2 ** 2
        burst_position = drop_position + horizontal_displacement - np.array([0, 0, vertical_displacement])
        burst_time = t1 + t2

        if burst_position[2] < 0:
            return 10000 + abs(burst_position[2]) * 100  # 根据地下深度增加惩罚

        total_integral = 0
        max_dt = 0.01
        min_dt = 0.001
        t_current = max(burst_time, 0)
        t_end = min(burst_time + 20, 40)

        if t_current >= t_end:
            return 10000

        prev_quality = 0
        iteration_count = 0
        max_iterations = 10000  # 防止无限循环

        while t_current < t_end and iteration_count < max_iterations:
            dt = max(min_dt, min(max_dt, 0.1 / (abs(prev_quality - 0.5) + 0.1)))

            # 导弹位置
            M_t = missile_position(t_current)

            sink_distance = smoke_sink_speed * (t_current - burst_time)
            S_t = burst_position - np.array([0, 0, sink_distance])

            if S_t[2] < 0:
                t_current += dt
                iteration_count += 1
                continue

            # 计算平滑遮蔽质量
            quality = calculate_cover_quality_improved(M_t, S_t, target_point)
            total_integral += quality * dt

            prev_quality = quality
            t_current += dt
            iteration_count += 1

        print(f"参数: v_u={v_u:.1f}, theta={theta_deg:.1f}°, t1={t1:.3f}, t2={t2:.3f} -> 积分: {total_integral:.6f}")
        return -total_integral

    except Exception as e:
        print(f"目标函数计算错误: {e}")
        return 10000


def optimize_smoke_strategy_advanced():
    bounds = [(70, 140), (0, 360), (0, 10), (0, 20)]

    # 使用更密集的初始点网格（集中在可能的最优区域）
    initial_guesses = []

    # 第一层：粗网格搜索
    for v in np.linspace(80, 130, 3):
        for theta in np.linspace(170, 190, 5):
            for t1 in np.linspace(1, 3, 3):
                for t2 in np.linspace(4, 7, 4):
                    initial_guesses.append([v, theta, t1, t2])

    # 第二层：更精细的网格
    for v in np.linspace(100, 120, 2):
        for theta in np.linspace(175, 185, 4):
            for t1 in np.linspace(1.5, 2.5, 3):
                for t2 in np.linspace(5, 6, 3):
                    initial_guesses.append([v, theta, t1, t2])

    np.random.seed(42)
    for _ in range(5):
        initial_guesses.append([
            np.random.uniform(70, 140),
            np.random.uniform(0, 360),
            np.random.uniform(0, 10),
            np.random.uniform(0, 20)
        ])

    best_result = None
    best_value = float('inf')
    results = []
    print(f"总共尝试 {len(initial_guesses)} 个初始点")
    for i, x0 in enumerate(initial_guesses):
        try:
            # 使用不同的优化方法
            result = minimize(
                objective_function_improved,
                x0,
                method='L-BFGS-B',  # 使用L-BFGS-B方法
                bounds=bounds,
                options={
                    'maxiter': 200,
                    'disp': False,
                    'ftol': 1e-8,
                    'gtol': 1e-8
                }
            )

            results.append((result.x, result.fun, result.success))

            if result.fun < best_value and result.success:
                best_result = result
                best_value = result.fun
            else:
                return 0

        except Exception as e:
            print(f"✗ 优化失败: {e}")
            continue

    # 对结果进行局部精细化
    if best_result is not None:
        try:
            refined_result = minimize(
                objective_function_improved,
                best_result.x,
                method='Nelder-Mead',  # 单纯形法对局部精细化效果好
                options={
                    'maxiter': 100,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'disp': True
                }
            )

            if refined_result.fun < best_value:
                best_result = refined_result
                best_value = refined_result.fun
        except Exception as e:
            print(f"✗ 精细化失败: {e}")

    if best_result is not None:
        best_params = best_result.x
        best_value = -best_result.fun
        return best_params, best_value
    else:
        return None, None


def validate_and_visualize(params):
    v_u, theta_deg, t1, t2 = params
    burst_position, burst_time = calculate_smoke_burst(params)

    # 计算遮蔽时间积分
    total_integral = -objective_function_improved(params)
    print(f"遮蔽时间积分: {total_integral:.6f}")

    # 生成时间序列数据用于绘图
    times = np.linspace(burst_time, burst_time + 30, 1000)
    qualities = []
    missile_positions = []
    smoke_positions = []

    for t in times:
        M_t = missile_position(t)
        sink_distance = smoke_sink_speed * (t - burst_time)
        S_t = burst_position - np.array([0, 0, sink_distance])
        quality = calculate_cover_quality_improved(M_t, S_t, target_point)

        qualities.append(quality)
        missile_positions.append(M_t)
        smoke_positions.append(S_t)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(times - burst_time, qualities, 'b-', linewidth=2)
    plt.xlabel('爆开后时间 (s)')
    plt.ylabel('遮蔽质量')
    plt.title('遮蔽质量随时间变化')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    missile_x = [pos[0] for pos in missile_positions]
    missile_y = [pos[1] for pos in missile_positions]
    smoke_x = [pos[0] for pos in smoke_positions]
    smoke_y = [pos[1] for pos in smoke_positions]

    plt.plot(missile_x, missile_y, 'r-', label='导弹轨迹')
    plt.plot(smoke_x, smoke_y, 'g--', label='烟幕轨迹')
    plt.scatter(target_point[0], target_point[1], c='blue', s=100, label='目标')
    plt.scatter(burst_position[0], burst_position[1], c='orange', s=100, label='爆开位置')
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.title('导弹和烟幕轨迹')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    altitudes = [pos[2] for pos in missile_positions]
    smoke_altitudes = [pos[2] for pos in smoke_positions]
    plt.plot(times - burst_time, altitudes, 'r-', label='导弹高度')
    plt.plot(times - burst_time, smoke_altitudes, 'g--', label='烟幕高度')
    plt.xlabel('爆开后时间 (s)')
    plt.ylabel('高度 (m)')
    plt.title('高度变化')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    effective_qualities = [q for q in qualities if q > 0.1]
    plt.hist(effective_qualities, bins=20, alpha=0.7, color='purple')
    plt.xlabel('遮蔽质量')
    plt.ylabel('频率')
    plt.title('遮蔽质量分布')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('smoke_cover_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return total_integral


def test_optimal_region():
    test_points = []
    for v in [100, 110, 120, 130]:
        for theta in [175, 180, 185]:
            for t1 in [1.5, 2.0, 2.5]:
                for t2 in [5.0, 5.5, 6.0]:
                    test_points.append([v, theta, t1, t2])

    results = []
    for params in test_points:
        value = -objective_function_improved(params)
        results.append((params, value))
        print(f"测试点 {params} -> 积分: {value:.6f}")

    best_test = max(results, key=lambda x: x[1])
    print(f"\n最佳测试点: {best_test[0]} -> 积分: {best_test[1]:.6f}")

    return best_test[0], best_test[1]


if __name__ == "__main__":
    test_params, test_value = test_optimal_region()
    best_params, best_value = optimize_smoke_strategy_advanced()

    if best_params is not None:
        final_value = validate_and_visualize(best_params)

        v_u, theta_deg, t1, t2 = best_params
        burst_position, burst_time = calculate_smoke_burst(best_params)
        print(f"• 飞行速度: {v_u:.6f} m/s")
        print(f"• 航向角: {theta_deg:.6f}°")
        print(f"• 投放时间: {t1:.6f} s 后")
        print(f"• 起爆时间: {t2:.6f} s 后")
        print(f"• 烟幕爆开时间: {burst_time:.6f} s")
        print(f"• 烟幕爆开位置: ({burst_position[0]:.6f}, {burst_position[1]:.6f}, {burst_position[2]:.6f})")
        print(f"• 预计最大遮蔽积分: {final_value:.6f}")
    print("完成")