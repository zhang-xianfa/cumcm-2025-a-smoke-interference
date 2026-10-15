# ## 烟幕干扰弹投放策略优化 - 问题4 (差分进化算法)
# 差分进化，t4

import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.usetex'] = False
g = 9.8
smoke_radius = 10
smoke_sink_speed = 3
smoke_effective_duration = 20
missile_speed = 300
FY1_initial = np.array([17800, 0, 1800])
FY2_initial = np.array([12000, 1400, 1400])
FY3_initial = np.array([6000, -3000, 700])
M1_initial = np.array([20000, 0, 2000])
# 目标点
target_point = np.array([0, 200, 0])
fake_target = np.array([0, 0, 0])

def calculate_missile_direction_m1():
    direction = fake_target - M1_initial
    return direction / np.linalg.norm(direction)

missile_direction_m1 = calculate_missile_direction_m1()

def missile_position_m1(t):
    return M1_initial + missile_speed * t * missile_direction_m1

def calculate_smoke_burst_single(uav_initial, params):
    v_u, theta_deg, t1, t2 = params
    theta = np.deg2rad(theta_deg)

    uav_velocity_vector = v_u * np.array([np.cos(theta), np.sin(theta), 0])
    drop_position = uav_initial + uav_velocity_vector * t1

    horizontal_displacement = uav_velocity_vector * t2
    vertical_displacement = 0.5 * g * t2 ** 2
    burst_position = drop_position + horizontal_displacement - np.array([0, 0, vertical_displacement])

    burst_time = t1 + t2
    return burst_position, burst_time, drop_position

def calculate_cover_quality_improved(M, S, P):
    try:
        MS = S - M
        MP = P - M
        distance_MS = np.linalg.norm(MS)
        norm_MP = np.linalg.norm(MP)

        if distance_MS < 1e-12 or norm_MP < 1e-12:
            return 0.0

        cos_beta = np.dot(MP, MS) / (norm_MP * distance_MS)
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

def calculate_total_cover_integral_multi(params_list, uav_initials, debug=False):

    num_smokes = len(params_list)
    burst_positions = []
    burst_times = []

    for i in range(num_smokes):
        params = params_list[i]
        uav_initial = uav_initials[i]
        v_u, theta_deg, t1, t2 = params

        if not (70 <= v_u <= 140 and 0 <= theta_deg <= 360 and 0 <= t1 <= 10 and 0 <= t2 <= 20):
            if hasattr(calculate_total_cover_integral_multi, 'return_single_contributions'):
                return 0.0, [0.0] * num_smokes
            return 0.0

        if t1 + t2 > 30:
            if hasattr(calculate_total_cover_integral_multi, 'return_single_contributions'):
                return 0.0, [0.0] * num_smokes
            return 0.0

        burst_pos, burst_time, _ = calculate_smoke_burst_single(uav_initial, params)
        if burst_pos[2] < 0:
            if hasattr(calculate_total_cover_integral_multi, 'return_single_contributions'):
                return 0.0, [0.0] * num_smokes
            return 0.0

        burst_positions.append(burst_pos)
        burst_times.append(burst_time)

    total_integral = 0.0
    dt = 0.01
    t_start_sim = 0.0
    t_end_sim = 70.0
    current_t = t_start_sim

    while current_t < t_end_sim:
        M_t = missile_position_m1(current_t)
        max_quality = 0.0

        for i in range(num_smokes):
            burst_time = burst_times[i]
            if current_t < burst_time or (current_t - burst_time) > smoke_effective_duration:
                continue

            sink_distance = smoke_sink_speed * (current_t - burst_time)
            S_t = burst_positions[i] - np.array([0, 0, sink_distance])

            if S_t[2] < 0:
                continue

            quality = calculate_cover_quality_improved(M_t, S_t, target_point)
            if quality > max_quality:
                max_quality = quality

        total_integral += max_quality * dt
        current_t += dt

    if debug:
        print(f"调试信息 - 总遮蔽积分: {total_integral:.6f}")

    if hasattr(calculate_total_cover_integral_multi, 'return_single_contributions'):
        single_contributions = []
        for i in range(len(params_list)):
            single_contributions.append(calculate_single_uav_cover(uav_initials[i], params_list[i]))
        return total_integral, single_contributions

    return total_integral


def calculate_single_uav_cover(uav_initial, params, debug=False):

    v_u, theta_deg, t1, t2 = params
    if not (70 <= v_u <= 140 and 0 <= theta_deg <= 360 and 0 <= t1 <= 10 and 0 <= t2 <= 20):
        return 0.0
    if t1 + t2 > 30:
        return 0.0

    burst_pos, burst_time, _ = calculate_smoke_burst_single(uav_initial, params)
    if burst_pos[2] < 0:
        return 0.0

    total_integral = 0.0
    dt = 0.01
    t_start_sim = 0.0
    t_end_sim = 80.0
    current_t = t_start_sim

    while current_t < t_end_sim:
        M_t = missile_position_m1(current_t)

        if current_t < burst_time or (current_t - burst_time) > smoke_effective_duration:
            current_t += dt
            continue

        sink_distance = smoke_sink_speed * (current_t - burst_time)
        S_t = burst_pos - np.array([0, 0, sink_distance])

        if S_t[2] < 0:
            current_t += dt
            continue

        quality = calculate_cover_quality_improved(M_t, S_t, target_point)
        total_integral += quality * dt
        current_t += dt

    if debug:
        print(f"单无人机遮蔽积分: {total_integral:.6f}")

    return total_integral

def fitness_function_multi(params, debug=False):
    params_fy1 = params[0:4]
    params_fy2 = params[4:8]
    params_fy3 = params[8:12]
    params_list = [params_fy1, params_fy2, params_fy3]
    uav_initials = [FY1_initial, FY2_initial, FY3_initial]
    total_integral = calculate_total_cover_integral_multi(params_list, uav_initials, debug)
    return -total_integral

POPULATION_SIZE_DE = 50
MAX_GENERATIONS_DE = 1000
F_DE = 0.8
CR_DE = 0.9

PARAM_BOUNDS_DE = [
    (70, 140), (0, 360), (0.1, 10), (0.1, 20),
    (70, 140), (0, 360), (0.1, 10), (0.1, 20),
    (70, 140), (0, 360), (0.1, 10), (0.1, 20)
]
NUM_PARAMS = len(PARAM_BOUNDS_DE)

def initialize_population_de(pop_size, bounds):
    population = np.zeros((pop_size, NUM_PARAMS))
    for i in range(NUM_PARAMS):
        min_val, max_val = bounds[i]
        population[:, i] = np.random.uniform(min_val, max_val, size=pop_size)
    return population

def repair_individual(individual, bounds):
    repaired = np.clip(individual, [b[0] for b in bounds], [b[1] for b in bounds])
    return repaired

def differential_evolution_optimizer(fitness_func, bounds, pop_size, generations, F, CR):
    population = initialize_population_de(pop_size, bounds)
    fitness = np.array([fitness_func(ind) for ind in population])
    best_fitness_overall = np.min(fitness)
    best_individual_overall = population[np.argmin(fitness)].copy()

    best_fitness_history = [best_fitness_overall]
    avg_fitness_history = [np.mean(fitness)]

    print(f"差分进化 - 初始最佳适应度（负遮蔽时间）：{best_fitness_overall:.6f}")
    print(f"差分进化 - 初始平均适应度（负遮蔽时间）：{np.mean(fitness):.6f}")

    for generation in range(generations):
        new_population = np.copy(population)

        for i in range(pop_size):
            candidates = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            mutant_vector = population[r1] + F * (population[r2] - population[r3])
            cross_points = np.random.rand(NUM_PARAMS) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, NUM_PARAMS)] = True
            trial_vector = np.where(cross_points, mutant_vector, population[i])
            repaired_trial = repair_individual(trial_vector, bounds)
            trial_fitness = fitness_func(repaired_trial)
            if trial_fitness < fitness[i]:
                new_population[i] = repaired_trial
                fitness[i] = trial_fitness
                if trial_fitness < best_fitness_overall:
                    best_fitness_overall = trial_fitness
                    best_individual_overall = repaired_trial.copy()

        population = new_population
        best_fitness_history.append(best_fitness_overall)
        avg_fitness_history.append(np.mean(fitness))

        if (generation + 1) % 10 == 0:
            print(f"差分进化 - 代 {generation + 1}/{generations}, 当前最佳适应度: {best_fitness_overall:.6f}, 平均: {np.mean(fitness):.6f}")

    print(f"\n--- 差分进化优化完成 ---")
    print(f"找到的最优适应度: {best_fitness_overall:.6f}")
    return best_individual_overall, best_fitness_history, avg_fitness_history

def local_refinement(best_params):
    print("\n--- 进行局部精细化优化 ---")
    bounds_list = PARAM_BOUNDS_DE
    lower_bounds = [b[0] for b in bounds_list]
    upper_bounds = [b[1] for b in bounds_list]
    bounds_obj = Bounds(lower_bounds, upper_bounds)

    try:
        result = minimize(
            fitness_function_multi,
            best_params,
            method='Nelder-Mead',
            bounds=bounds_obj,
            options={'maxiter': 800, 'disp': True}
        )
        if result.success:
            refined_fitness = -result.fun
            print(f"局部精细化成功！精细化后最大遮蔽时间：{refined_fitness:.6f} s")
            return result.x, refined_fitness
        else:
            print(f"局部精细化未能成功: {result.message}")
            original_fitness = -fitness_function_multi(best_params)
            return best_params, original_fitness
    except Exception as e:
        print(f"局部精细化错误: {e}")
        original_fitness = -fitness_function_multi(best_params)
        return best_params, original_fitness

def validate_and_visualize_multi(best_params_refined):
    params_fy1 = best_params_refined[0:4]
    params_fy2 = best_params_refined[4:8]
    params_fy3 = best_params_refined[8:12]
    params_list = [params_fy1, params_fy2, params_fy3]
    uav_initials = [FY1_initial, FY2_initial, FY3_initial]

    burst_pos1, burst_time1, drop_pos1 = calculate_smoke_burst_single(FY1_initial, params_fy1)
    burst_pos2, burst_time2, drop_pos2 = calculate_smoke_burst_single(FY2_initial, params_fy2)
    burst_pos3, burst_time3, drop_pos3 = calculate_smoke_burst_single(FY3_initial, params_fy3)

    print(f"\n=== 问题4：最优投放策略总结 (3无人机干扰M1) ===")
    print("\n--- 无人机 FY1 策略 ---")
    print(f"  飞行速度 v: {params_fy1[0]:.3f} m/s")
    print(f"  航向角 θ: {params_fy1[1]:.3f}°")
    print(f"  投放时间 t1: {params_fy1[2]:.3f} s")
    print(f"  起爆延迟 t2: {params_fy1[3]:.3f} s")
    print(f"  投放位置: ({drop_pos1[0]:.1f}, {drop_pos1[1]:.1f}, {drop_pos1[2]:.1f}) m")
    print(f"  烟幕爆开时间: {burst_time1:.3f} s")
    print(f"  烟幕爆开位置: ({burst_pos1[0]:.1f}, {burst_pos1[1]:.1f}, {burst_pos1[2]:.1f}) m")

    print("\n--- 无人机 FY2 策略 ---")
    print(f"  飞行速度 v: {params_fy2[0]:.3f} m/s")
    print(f"  航向角 θ: {params_fy2[1]:.3f}°")
    print(f"  投放时间 t1: {params_fy2[2]:.3f} s")
    print(f"  起爆延迟 t2: {params_fy2[3]:.3f} s")
    print(f"  投放位置: ({drop_pos2[0]:.1f}, {drop_pos2[1]:.1f}, {drop_pos2[2]:.1f}) m")
    print(f"  烟幕爆开时间: {burst_time2:.3f} s")
    print(f"  烟幕爆开位置: ({burst_pos2[0]:.1f}, {burst_pos2[1]:.1f}, {burst_pos2[2]:.1f}) m")

    print("\n--- 无人机 FY3 策略 ---")
    print(f"  飞行速度 v: {params_fy3[0]:.3f} m/s")
    print(f"  航向角 θ: {params_fy3[1]:.3f}°")
    print(f"  投放时间 t1: {params_fy3[2]:.3f} s")
    print(f"  起爆延迟 t2: {params_fy3[3]:.3f} s")
    print(f"  投放位置: ({drop_pos3[0]:.1f}, {drop_pos3[1]:.1f}, {drop_pos3[2]:.1f}) m")
    print(f"  烟幕爆开时间: {burst_time3:.3f} s")
    print(f"  烟幕爆开位置: ({burst_pos3[0]:.1f}, {burst_pos3[1]:.1f}, {burst_pos3[2]:.1f}) m")

    total_integral = calculate_total_cover_integral_multi(params_list, uav_initials, debug=True)
    calculate_total_cover_integral_multi.return_single_contributions = True
    total_integral, single_contributions = calculate_total_cover_integral_multi(params_list, uav_initials, debug=True)
    calculate_total_cover_integral_multi.return_single_contributions = False

    fy1_int, fy2_int, fy3_int = single_contributions

    print(f"\n--- 单无人机遮蔽贡献 ---")
    print(f"  FY1 单独遮蔽时间: {fy1_int:.6f} s")
    print(f"  FY2 单独遮蔽时间: {fy2_int:.6f} s")
    print(f"  FY3 单独遮蔽时间: {fy3_int:.6f} s")

    return total_integral, single_contributions

def save_results_to_excel(best_params_refined, total_integral,single_contributions):
    params_fy1 = best_params_refined[0:4]
    params_fy2 = best_params_refined[4:8]
    params_fy3 = best_params_refined[8:12]

    burst_pos1, burst_time1, drop_pos1 = calculate_smoke_burst_single(FY1_initial, params_fy1)
    burst_pos2, burst_time2, drop_pos2 = calculate_smoke_burst_single(FY2_initial, params_fy2)
    burst_pos3, burst_time3, drop_pos3 = calculate_smoke_burst_single(FY3_initial, params_fy3)

    results_data = {
        '无人机': ['FY1', 'FY2', 'FY3', '总计'],
        '飞行速度v_u (m/s)': [f"{params_fy1[0]:.3f}", f"{params_fy2[0]:.3f}", f"{params_fy3[0]:.3f}", '-'],
        '航向角theta (°)': [f"{params_fy1[1]:.3f}", f"{params_fy2[1]:.3f}", f"{params_fy3[1]:.3f}", '-'],
        '投放时间t1 (s)': [f"{params_fy1[2]:.3f}", f"{params_fy2[2]:.3f}", f"{params_fy3[2]:.3f}", '-'],
        '起爆延迟t2 (s)': [f"{params_fy1[3]:.3f}", f"{params_fy2[3]:.3f}", f"{params_fy3[3]:.3f}", '-'],
        '投放X坐标(m)': [f"{drop_pos1[0]:.1f}", f"{drop_pos2[0]:.1f}", f"{drop_pos3[0]:.1f}", '-'],
        '投放Y坐标(m)': [f"{drop_pos1[1]:.1f}", f"{drop_pos2[1]:.1f}", f"{drop_pos3[1]:.1f}", '-'],
        '投放Z坐标(m)': [f"{drop_pos1[2]:.1f}", f"{drop_pos2[2]:.1f}", f"{drop_pos3[2]:.1f}", '-'],
        '烟幕爆开时间(s)': [f"{burst_time1:.3f}", f"{burst_time2:.3f}", f"{burst_time3:.3f}", '-'],
        '烟幕爆开X坐标(m)': [f"{burst_pos1[0]:.1f}", f"{burst_pos2[0]:.1f}", f"{burst_pos3[0]:.1f}", '-'],
        '烟幕爆开Y坐标(m)': [f"{burst_pos1[1]:.1f}", f"{burst_pos2[1]:.1f}", f"{burst_pos3[1]:.1f}", '-'],
        '烟幕爆开Z坐标(m)': [f"{burst_pos1[2]:.1f}", f"{burst_pos2[2]:.1f}", f"{burst_pos3[2]:.1f}", '-'],
        '单独遮蔽贡献(s)': [f"{single_contributions[0]:.6f}", f"{single_contributions[1]:.6f}",
                            f"{single_contributions[2]:.6f}", '-'],
        '总遮蔽时间(s)': ['-', '-', '-', f"{total_integral:.6f}"]
    }

    results_df = pd.DataFrame(results_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'result2.xlsx'  # 按题目要求保存为result2.xlsx
    results_df.to_excel(filename, index=False)
    print(f"\n结果已保存到: {filename}")

    return results_df

def plot_fitness_history(best_history, avg_history):
    plt.figure(figsize=(10, 6))
    plt.plot([-f for f in best_history], 'b-', label='最佳遮蔽时间')
    plt.plot([-f for f in avg_history], 'r-', label='平均遮蔽时间')
    plt.xlabel('代数')
    plt.ylabel('遮蔽时间 (s)')
    plt.title('差分进化优化过程')
    plt.legend()
    plt.grid(True)
    plt.savefig('problem4_DE_evolution.png', dpi=300)
    print("进化曲线图已保存为 'problem4_DE_evolution.png'")


def plot_multi_visualizations(best_params_refined, single_contributions, best_history, avg_history):
    params_fy1 = best_params_refined[0:4]
    params_fy2 = best_params_refined[4:8]
    params_fy3 = best_params_refined[8:12]
    params_list = [params_fy1, params_fy2, params_fy3]
    uav_initials = [FY1_initial, FY2_initial, FY3_initial]

    plt.figure(figsize=(10, 6))
    plt.plot([-f for f in best_history], 'b-', label='最佳遮蔽时间')
    plt.plot([-f for f in avg_history], 'r-', label='平均遮蔽时间')
    plt.xlabel('迭代代数')
    plt.ylabel('遮蔽时间 (s)')
    plt.title('DE算法适应度进化曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig('de_evolution.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    uav_names = ['FY1', 'FY2', 'FY3']
    bars = plt.bar(uav_names, single_contributions, color=['blue', 'green', 'red'])
    plt.xlabel('无人机名称')
    plt.ylabel('遮蔽时间 (s)')
    plt.title('单无人机有效干扰时长对比')
    for bar, value in zip(bars, single_contributions):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{value:.2f}', ha='center', va='bottom')
    plt.grid(True, axis='y')
    plt.savefig('single_uav_contribution.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    time_points = np.linspace(0, 80, 800)

    for i, (params, uav_initial) in enumerate(zip(params_list, uav_initials)):
        burst_pos, burst_time, _ = calculate_smoke_burst_single(uav_initial, params)
        quality_over_time = []

        for t in time_points:
            if t < burst_time or (t - burst_time) > smoke_effective_duration:
                quality_over_time.append(0)
                continue

            M_t = missile_position_m1(t)
            sink_distance = smoke_sink_speed * (t - burst_time)
            S_t = burst_pos - np.array([0, 0, sink_distance])

            if S_t[2] < 0:
                quality_over_time.append(0)
                continue

            quality = calculate_cover_quality_improved(M_t, S_t, target_point)
            quality_over_time.append(quality)

        plt.plot(time_points, quality_over_time, label=f'FY{i + 1}', linewidth=2)

    plt.xlabel('时间 (s)')
    plt.ylabel('遮蔽质量')
    plt.title('遮蔽质量随时间动态变化')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.savefig('cover_quality_dynamic.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))

    missile_traj_x = [M1_initial[0], fake_target[0]]
    missile_traj_y = [M1_initial[1], fake_target[1]]
    plt.plot(missile_traj_x, missile_traj_y, 'r-', linewidth=2, label='导弹轨迹')

    colors = ['orange', 'green', 'purple']
    labels = ['FY1烟幕', 'FY2烟幕', 'FY3烟幕']

    for i, (params, uav_initial) in enumerate(zip(params_list, uav_initials)):
        burst_pos, _, _ = calculate_smoke_burst_single(uav_initial, params)
        plt.scatter(burst_pos[0], burst_pos[1], color=colors[i], s=100, label=labels[i], marker='o')

    plt.scatter(target_point[0], target_point[1], color='blue', s=150, label='真目标', marker='s')
    plt.scatter(fake_target[0], fake_target[1], color='black', s=150, label='假目标', marker='^')

    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.title('导弹与烟幕空间轨迹（XY平面）')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('spatial_trajectory.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))

    all_qualities = []
    dt = 0.1
    current_t = 0
    while current_t < 80:
        M_t = missile_position_m1(current_t)
        max_quality = 0.0

        for i, (params, uav_initial) in enumerate(zip(params_list, uav_initials)):
            burst_pos, burst_time, _ = calculate_smoke_burst_single(uav_initial, params)

            if current_t < burst_time or (current_t - burst_time) > smoke_effective_duration:
                continue

            sink_distance = smoke_sink_speed * (current_t - burst_time)
            S_t = burst_pos - np.array([0, 0, sink_distance])

            if S_t[2] < 0:
                continue

            quality = calculate_cover_quality_improved(M_t, S_t, target_point)
            if quality > max_quality:
                max_quality = quality

        all_qualities.append(max_quality)
        current_t += dt

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, bin_edges = np.histogram(all_qualities, bins=bins)
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    bin_labels = [f'{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}' for i in range(len(bin_edges) - 1)]

    plt.bar(bin_labels, hist / sum(hist) * 100, alpha=0.7)
    plt.xlabel('遮蔽质量区间')
    plt.ylabel('时间占比 (%)')
    plt.title('遮蔽质量分布直方图')
    plt.grid(True, axis='y')

    for i, v in enumerate(hist / sum(hist) * 100):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')

    plt.savefig('cover_quality_distribution.png', dpi=300)
    plt.close()

    print("所有可视化图表已保存完成!")

if __name__ == "__main__":
    print("=== 问题4：无人机协同烟幕投放策略优化 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n[STEP 1/4] 运行差分进化算法...")
    start_time_de = time.time()
    best_params_de, best_history, avg_history = differential_evolution_optimizer(
        fitness_function_multi,
        PARAM_BOUNDS_DE,
        POPULATION_SIZE_DE,
        MAX_GENERATIONS_DE,
        F_DE,
        CR_DE
    )
    end_time_de = time.time()
    print(f"差分进化耗时: {end_time_de - start_time_de:.2f} 秒")

    print("\n[STEP 2/4] 进行局部精细化...")
    start_time_refine = time.time()
    refined_params, refined_fitness = local_refinement(best_params_de)
    end_time_refine = time.time()
    print(f"局部精细化耗时: {end_time_refine - start_time_refine:.2f} 秒")

    print("\n[STEP 3/4] 验证并输出最终策略...")
    final_total_integral, single_contributions = validate_and_visualize_multi(refined_params)

    print("\n[STEP 4/4] 保存结果到 Excel 并生成可视化...")
    save_results_to_excel(refined_params, final_total_integral, single_contributions)
    plot_multi_visualizations(refined_params, single_contributions, best_history, avg_history)

    print("\n=== 优化计算完成 ===")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")