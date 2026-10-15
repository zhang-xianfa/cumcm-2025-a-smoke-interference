%% 
g = 9.8;
FY1_v = 120;
FY1_t = 1.5;
Smoke_t = 3.6;
M1_v = 300;
Smoke_v = 3;
P = [0, 200, 0];

%% 首先计算烟幕爆开的位置S
x1 = FY1_t * FY1_v;
x2 = FY1_v * Smoke_t;
h = 1/2 * g * Smoke_t^2;

Smoke_x = 17800 - x1 - x2;
Smoke_y = 0;
Smoke_z = 1800 - h;
Smoke_r = 10;

S = [Smoke_x, Smoke_y, Smoke_z];
disp(S);
%% 计算烟雾弹爆开时导弹的位置M1
M1_S = M1_v * (FY1_t + Smoke_t);
c = 20000 / sqrt(20000^2 + 2000^2);
s = 2000 / sqrt(20000^2 + 2000^2);
M1_x = 20000 - M1_S * c;
M1_y = 0;
M1_z = 2000 - M1_S * s;
M1 = [M1_x, M1_y, M1_z];

M1_v_x = M1_v * c;
M1_v_z = M1_v * s;
disp(M1);
%% 求圆锥
syms t x y z real
M_x = M1_x - M1_v_x * t;
M_y = 0;
M_z = M1_z - M1_v_z * t;
M = [M_x, M_y, M_z];

S_x = Smoke_x;
S_y = 0;
S_z = Smoke_z - Smoke_v * t;
S = [S_x, S_y, S_z];


d = S - M;
L = norm(d);
MT = sqrt(L^2 - Smoke_r^2);
cos_theta = MT/L;

Q = [x,y,z];
MQ = Q-M;

cone_equation = ( dot(MQ, d)/ (norm(MQ)*norm(d)) ) - cos_theta == 0;

cone_equation = simplify(cone_equation);
disp('圆锥的解析方程为：');
disp(cone_equation);

cone_poly = expand(lhs(cone_equation));

disp('圆锥的多项式形式：');
disp(cone_poly == 0);

%%  第一次包裹时间是2.9140
t_out = (M1_x - Smoke_x) / M1_v_x;

%% 画图

T = 2.9290;  % 给定时间
Z = 10; % 给定高度平面

% 假设以下参数已定义（根据实际数据修改）
% M1_x, M1_v_x, M1_z, M1_v_z, Smoke_x, Smoke_z, Smoke_v, Smoke_r

M_x_t = M1_x - M1_v_x * T;
M_z_t = M1_z - M1_v_z * T;
S_x_t = Smoke_x;
S_z_t = Smoke_z - Smoke_v * T;

M_T = [M_x_t, 0, M_z_t];
S_T = [S_x_t, 0, S_z_t];

d_vector = S_T - M_T;
d_norm = norm(d_vector);

% 圆锥半顶角的正弦和余弦
sin_theta = Smoke_r / d_norm;
cos_theta = sqrt(1 - sin_theta^2); % MT/L


syms x y real

Q = [x, y, Z];
M_T_vec = [M_x_t, 0, M_z_t];
d_vec = d_vector;

% 圆锥方程：(MQ·d)/(|MQ||d|) = cos_theta
MQ = Q - M_T_vec;
norm_MQ_sq = (x - M_x_t)^2 + y^2 + (Z - M_z_t)^2;
dot_MQ_d = dot(MQ, d_vec);
norm_d_sq = norm(d_vec)^2;

cone_eq = dot_MQ_d^2 - cos_theta^2 * norm_MQ_sq * norm_d_sq;
cone_eq_expanded = expand(cone_eq);

disp('在Z=10平面，T时刻的椭圆方程为：');
disp(cone_eq_expanded == 0);


% 系数提取
syms x y real;

% 使用求导方法提取系数
A_coeff = diff(cone_eq_expanded, x, 2) / 2;
A_coeff = simplify(A_coeff);
B_coeff = diff(cone_eq_expanded, y, 2) / 2;
B_coeff = simplify(B_coeff);
C_temp = diff(cone_eq_expanded, x);
C_coeff = simplify(C_temp - 2*A_coeff*x);
D_coeff = subs(cone_eq_expanded, {x, y}, {0, 0});
D_coeff = simplify(D_coeff);

disp('提取的系数:');
fprintf('A = %s\n', char(A_coeff));
fprintf('B = %s\n', char(B_coeff));
fprintf('C = %s\n', char(C_coeff));
fprintf('D = %s\n', char(D_coeff));

[A_num, A_den] = numden(A_coeff);
[B_num, B_den] = numden(B_coeff);
[C_num, C_den] = numden(C_coeff);
[D_num, D_den] = numden(D_coeff);


% 使用提取的系数进行椭圆分析
A = A_coeff;
B = B_coeff;
C = C_coeff;
D = D_coeff;

% 配方转化为标准椭圆方程
eq = A*x^2 + B*y^2 + C*x + D == 0;

% 对x配方：A(x-h)^2 + B y^2 = E
h = -C/(2*A);  % x方向中心
E = (C^2)/(4*A) - D;  % 配方后的常数项

% 标准方程：(x-h)^2/(E/A) + y^2/(E/B) = 1
a_sq = E/A;  % x方向的半轴平方
b_sq = E/B;  % y方向的半轴平方

% 检查是否为椭圆（a_sq > 0, b_sq > 0）
if double(a_sq) > 0 && double(b_sq) > 0
    a = sqrt(a_sq);
    b = sqrt(b_sq);
    
    fprintf('\n椭圆参数:\n');
    fprintf('椭圆中心 (h, k): (%s, 0)\n', char(vpa(h, 10)));
    fprintf('半长轴 a: %s\n', char(vpa(a, 10)));
    fprintf('半短轴 b: %s\n', char(vpa(b, 10)));
    
    % 转换为数值用于绘图
    h_num = double(h);
    a_num = double(a);
    b_num = double(b);
    
    % 生成椭圆点
    theta = linspace(0, 2*pi, 1000);
    x_vals = h_num + a_num * cos(theta);
    y_vals = b_num * sin(theta);
    
    % 生成额外的圆：圆心(0,200)，半径7
    circle_theta = linspace(0, 2*pi, 1000);
    circle_x = 0 + 7 * cos(circle_theta);
    circle_y = 200 + 7 * sin(circle_theta);
    
    % 计算导弹和烟幕在Z=10平面的投影位置
    M_x_t = M1_x - M1_v_x * T;
    M_z_t = M1_z - M1_v_z * T;
    S_x_t = Smoke_x;
    S_z_t = Smoke_z - Smoke_v * T;
    
    % 绘图
    figure;
    plot(x_vals, y_vals, 'b-', 'LineWidth', 2);
    hold on;
    % 绘制额外的圆
    plot(circle_x, circle_y, 'r-', 'LineWidth', 2);
    plot(h_num, 0, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    plot(M_x_t, 0, 'g^', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    plot(S_x_t, 0, 'ms', 'MarkerSize', 10, 'MarkerFaceColor', 'm');
    % 标记新添加圆的圆心
    plot(0, 200, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    
    % 添加标注
    text(h_num, 0, ' 椭圆中心', 'VerticalAlignment', 'bottom');
    text(M_x_t, 0, '  导弹位置', 'VerticalAlignment', 'bottom');
    text(S_x_t, 0, '  烟幕中心', 'VerticalAlignment', 'bottom');
    text(0, 200, '  新增圆圆心', 'VerticalAlignment', 'bottom');
    
    xlabel('x 坐标');
    ylabel('y 坐标');
    title('2.929时刻，Z=10平面上的烟幕椭圆投影及真目标');
    legend('烟幕椭圆', '真目标(0,200),r=7', '椭圆中心', '导弹位置', '烟幕中心', '真目标', 'Location', 'best');
    grid on;
    axis equal;  % 保持横纵比一致
    
    % 调整坐标轴范围：Y轴扩大显示范围（关键修改）
    % 计算当前Y轴元素的最大最小值
    y_min = min([y_vals, circle_y, 0, 200]);
    y_max = max([y_vals, circle_y, 0, 200]);
    % 扩展Y轴范围（上下各增加当前范围的50%，确保足够宽松）
    y_range = y_max - y_min;
    y_min_expanded = y_min - 0.5 * y_range;  % 向下扩展50%
    y_max_expanded = y_max + 0.5 * y_range;  % 向上扩展50%
    
    % X轴范围保持自适应，Y轴用扩展后的范围
    x_min = min([x_vals, circle_x, M_x_t, S_x_t, 0]) - 10;
    x_max = max([x_vals, circle_x, M_x_t, S_x_t, 0]) + 10;
    axis([x_min, x_max, y_min_expanded, y_max_expanded]);
    
else
    error('方程不构成椭圆（a²或b²非正）');
end












