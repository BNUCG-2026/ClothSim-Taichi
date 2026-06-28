import taichi as ti
import math

# 初始化 Taichi，使用 GPU 加速运算
ti.init(arch=ti.gpu)

# ================= 物理与网格参数 =================
N = 20             # 布料网格分辨率 N x N
mass = 1.0         # 质点质量
dt = 5e-4          # 时间步长
k_s = 10000.0      # 结构弹簧劲度系数
k_shear = 6000.0   # 剪切弹簧劲度系数 (选做)
k_bend = 4000.0    # 弯曲弹簧劲度系数 (选做)
k_d = 1.0          # 阻尼系数
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0  # 速度上限，防止数值爆炸

# 碰撞球参数 (选做)
ball_radius = 0.25
ball_center = ti.Vector([0.0, 0.2, 0.0])

# ================= 数据场定义 =================
x = ti.Vector.field(3, dtype=float, shape=N * N)       # 位置
v = ti.Vector.field(3, dtype=float, shape=N * N)       # 速度
f = ti.Vector.field(3, dtype=float, shape=N * N)       # 受力
is_fixed = ti.field(dtype=int, shape=N * N)            # 是否为固定点

# 隐式欧拉专用的预测缓存场
x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

# 扩展弹簧数据场容量：包含结构、剪切、弯曲弹簧后，最大弹簧数约为 N * N * 12
max_springs = N * N * 12
spring_indices = ti.field(dtype=int, shape=max_springs * 2) 
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_types = ti.field(dtype=int, shape=max_springs) # 0: 结构, 1: 剪切, 2: 弯曲
num_springs = ti.field(dtype=int, shape=())

# ================= 初始化 Kernel =================

@ti.kernel
def init_positions():
    """初始化质点位置与固定状态"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        # 将布料水平放置在上方
        x[idx] = ti.Vector([i * 0.045 - 0.425, 0.6, j * 0.045 - 0.425])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        # 固定一侧边缘的两个角点，让其自然下垂并落到球上
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.kernel
def init_springs():
    """初始化所有类型的弹簧 (结构、剪切、弯曲)"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        
        # 1. 结构弹簧 (Structural)
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
            spring_types[c] = 0
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()
            spring_types[c] = 0

        # 2. 剪切弹簧 (Shear) - 选做
        if i < N - 1 and j < N - 1:
            idx_diag1 = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag1])
            spring_lengths[c] = (x[idx] - x[idx_diag1]).norm()
            spring_types[c] = 1
        if i < N - 1 and j > 0:
            idx_diag2 = (i + 1) * N + (j - 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag2])
            spring_lengths[c] = (x[idx] - x[idx_diag2]).norm()
            spring_types[c] = 1

        # 3. 弯曲弹簧 (Bending) - 选做
        if i < N - 2:
            idx_bend_r = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_bend_r])
            spring_lengths[c] = (x[idx] - x[idx_bend_r]).norm()  # <- 修改为这行
            spring_types[c] = 2
        if j < N - 2:
            idx_bend_d = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_bend_d])
            spring_lengths[c] = (x[idx] - x[idx_bend_d]).norm()
            spring_types[c] = 2

@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()

# ================= Inline 物理计算函数 =================

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    """核心受力计算，区分不同弹簧类型系数"""
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
        
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        d = pos[idx_a] - pos[idx_b]
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            
            # 根据弹簧类型选择对应的劲度系数
            current_k = k_s
            if spring_types[i] == 1:
                current_k = k_shear
            elif spring_types[i] == 2:
                current_k = k_bend
                
            f_spring = -current_k * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

@ti.func
def resolve_collision(pos: ti.template(), vel: ti.template(), idx: int):
    """球体碰撞处理 (选做)"""
    dir = pos[idx] - ball_center
    dist = dir.norm()
    if dist < ball_radius:
        # 1. 位置投影回球体表面（稍微加一点偏移防止粘连）
        normal = dir / dist
        pos[idx] = ball_center + normal * (ball_radius + 1e-4)
        
        # 2. 移除向球心运动的法向速度成分
        v_normal = vel[idx].dot(normal)
        if v_normal < 0:
            vel[idx] -= v_normal * normal # 完美滑动摩擦（切向保留）

# ================= 积分器 Kernel =================

@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            resolve_collision(x, v, i) # 积分后碰撞处理

@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
            resolve_collision(x, v, i) # 积分后碰撞处理

@ti.kernel
def step_implicit_iter():
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
                resolve_collision(x_next, v_next, i) # 迭代中施加碰撞约束
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]

# ================= 主函数与 GGUI =================
def main():
    init_cloth()

    window = ti.ui.Window("GAMES101 - Mass Spring System Pro", (1024, 1024))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.8, 1.8)
    camera.lookat(0.0, 0.2, 0.0)

    current_method = 1 
    paused = False

    while window.running:
        # GUI
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.35, 0.35)
        window.GUI.text("Solver:")
        
        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler"): current_method = 0; init_cloth()
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler"): current_method = 1; init_cloth()
        if window.GUI.button(prefix_2 + "Implicit Euler (3-Iter)"): current_method = 2; init_cloth()

        window.GUI.text("")
        pause_label = "Resume" if paused else "Pause"
        if window.GUI.button(pause_label): paused = not paused
        if window.GUI.button("Reset"): init_cloth()
        window.GUI.end()

        # 物理步更新
        if not paused:
            for _ in range(40):
                if current_method == 0: step_explicit()
                elif current_method == 1: step_semi_implicit()
                elif current_method == 2: step_implicit_iter()

        # 3D 渲染配置
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.3, 0.3, 0.3))
        scene.point_light(pos=(1.0, 2.0, 1.0), color=(1, 1, 1))

        # 绘制网格顶点和弹簧
        scene.particles(x, radius=0.012, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.0, color=(0.8, 0.8, 0.8))
        
        # 绘制碰撞球 (利用单点 particles 模拟球体渲染)
        ball_pos = ti.Vector.field(3, dtype=float, shape=1)
        ball_pos[0] = ball_center
        scene.particles(ball_pos, radius=ball_radius, color=(0.9, 0.3, 0.3))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()