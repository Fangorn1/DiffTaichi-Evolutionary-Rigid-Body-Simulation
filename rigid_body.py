from robot_config import robots
import sys
import taichi as ti
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import copy

real = ti.f32
ti.init(default_fp=ti.f32, arch=ti.cpu, flatten_if=True)

max_steps = 4096
vis_interval = 256
output_vis_interval = 16
steps = 2048
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()

use_toi = False

x = vec()
v = vec()
rotation = scalar()
# angular velocity
omega = scalar()

halfsize = vec()

inverse_mass = scalar()
inverse_inertia = scalar()

v_inc = vec()
x_inc = vec()
rotation_inc = scalar()
omega_inc = scalar()

head_id = 3
goal = vec()

n_objects = 0
# target_ball = 0
elasticity = 0.0
ground_height = 0.1
gravity = -9.8
friction = 1.0
penalty = 1e4
damping = 10

gradient_clip = 30
spring_omega = 30
default_actuation = 0.1

n_springs = 0
spring_anchor_a = ti.field(ti.i32)
spring_anchor_b = ti.field(ti.i32)
# spring_length = -1 means it is a joint
spring_length = scalar()
spring_offset_a = vec()
spring_offset_b = vec()
spring_phase = scalar()
spring_actuation = scalar()
spring_stiffness = scalar()

n_sin_waves = 10

actuation = scalar()

max_n_objects = 7
max_n_springs = 12


def allocate_fields():
    ti.root.dense(ti.i, max_steps).dense(ti.j, max_n_objects).place(
        x, v, rotation, rotation_inc, omega, v_inc, x_inc, omega_inc
    )
    ti.root.dense(ti.i, max_n_objects).place(halfsize, inverse_mass, inverse_inertia)
    ti.root.dense(ti.i, max_n_springs).place(
        spring_anchor_a, spring_anchor_b, spring_length, spring_offset_a,
        spring_offset_b, spring_stiffness, spring_phase, spring_actuation
    )
    ti.root.dense(ti.ij, (max_steps, max_n_springs)).place(actuation)
    ti.root.place(loss, goal)
    ti.root.lazy_grad()


dt = 0.001
learning_rate = 0.001


@ti.func
def rotation_matrix(r):
    return ti.Matrix([[ti.cos(r), -ti.sin(r)], [ti.sin(r), ti.cos(r)]])


@ti.kernel
def initialize_properties():
    for i in range(n_objects):
        inverse_mass[i] = 1.0 / (4 * halfsize[i][0] * halfsize[i][1])
        inverse_inertia[i] = 1.0 / (4 / 3 * halfsize[i][0] * halfsize[i][1] *
                                    (halfsize[i][0] * halfsize[i][0] +
                                     halfsize[i][1] * halfsize[i][1]))
        # ti.print(inverse_mass[i])
        # ti.print(inverse_inertia[i])

@ti.kernel
def initialize_states():
    for i in range(max_n_objects):
        x[0, i] = [0.0, 0.0]
        v[0, i] = [0.0, 0.0]
        rotation[0, i] = 0.0
        omega[0, i] = 0.0


@ti.func
def to_world(t, i, rela_x):
    rot = rotation[t, i]
    rot_matrix = rotation_matrix(rot)

    rela_pos = rot_matrix @ rela_x
    rela_v = omega[t, i] * ti.Vector([-rela_pos[1], rela_pos[0]])

    world_x = x[t, i] + rela_pos
    world_v = v[t, i] + rela_v

    return world_x, world_v, rela_pos


@ti.func
def apply_impulse(t, i, impulse, location, toi_input):
    # ti.print(toi)
    delta_v = impulse * inverse_mass[i]
    delta_omega = (location - x[t, i]).cross(impulse) * inverse_inertia[i]

    toi = ti.min(ti.max(0.0, toi_input), dt)

    ti.atomic_add(x_inc[t + 1, i], toi * (-delta_v))
    ti.atomic_add(rotation_inc[t + 1, i], toi * (-delta_omega))

    ti.atomic_add(v_inc[t + 1, i], delta_v)
    ti.atomic_add(omega_inc[t + 1, i], delta_omega)


@ti.kernel
def collide(t: ti.i32):
    for i in range(n_objects):
        hs = halfsize[i]
        for k in ti.static(range(4)):
            # the corner for collision detection
            offset_scale = ti.Vector([k % 2 * 2 - 1, k // 2 % 2 * 2 - 1])

            corner_x, corner_v, rela_pos = to_world(t, i, offset_scale * hs)
            corner_v = corner_v + dt * gravity * ti.Vector([0.0, 1.0])

            # Apply impulse so that there's no sinking
            normal = ti.Vector([0.0, 1.0])
            tao = ti.Vector([1.0, 0.0])

            rn = rela_pos.cross(normal)
            rt = rela_pos.cross(tao)
            impulse_contribution = inverse_mass[i] + (rn) ** 2 * \
                                   inverse_inertia[i]
            timpulse_contribution = inverse_mass[i] + (rt) ** 2 * \
                                    inverse_inertia[i]

            rela_v_ground = normal.dot(corner_v)

            impulse = 0.0
            timpulse = 0.0
            new_corner_x = corner_x + dt * corner_v
            toi = 0.0
            if rela_v_ground < 0 and new_corner_x[1] < ground_height:
                impulse = -(1 +
                            elasticity) * rela_v_ground / impulse_contribution
                if impulse > 0:
                    # friction
                    timpulse = -corner_v.dot(tao) / timpulse_contribution
                    timpulse = ti.min(friction * impulse,
                                      ti.max(-friction * impulse, timpulse))
                    if corner_x[1] > ground_height:
                        toi = -(corner_x[1] - ground_height) / ti.min(
                            corner_v[1], -1e-3)

            apply_impulse(t, i, impulse * normal + timpulse * tao,
                          new_corner_x, toi)

            penalty = 0.0
            if new_corner_x[1] < ground_height:
                # apply penalty
                penalty = -dt * penalty * (
                    new_corner_x[1] - ground_height) / impulse_contribution

            apply_impulse(t, i, penalty * normal, new_corner_x, 0)


@ti.kernel
def apply_spring_force(t: ti.i32):
    for i in range(n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a, vel_a, rela_a = to_world(t, a, spring_offset_a[i])
        pos_b, vel_b, rela_b = to_world(t, b, spring_offset_b[i])
        dist = pos_a - pos_b
        length = dist.norm() + 1e-4

        act = actuation[t, i]

        is_joint = spring_length[i] == -1

        target_length = spring_length[i] * (1.0 + spring_actuation[i] * act)
        if is_joint:
            target_length = 0.0
        impulse = dt * (length -
                        target_length) * spring_stiffness[i] / length * dist

        if is_joint:
            rela_vel = vel_a - vel_b
            rela_vel_norm = rela_vel.norm() + 1e-1
            impulse_dir = rela_vel / rela_vel_norm
            impulse_contribution = inverse_mass[a] + \
              impulse_dir.cross(rela_a) ** 2 * inverse_inertia[
                                     a] + inverse_mass[b] + impulse_dir.cross(rela_b) ** 2 * \
                                   inverse_inertia[
                                     b]
            # project relative velocity
            impulse += rela_vel_norm / impulse_contribution * impulse_dir

        apply_impulse(t, a, -impulse, pos_a, 0.0)
        apply_impulse(t, b, impulse, pos_b, 0.0)



@ti.kernel
def advance_toi(t: ti.i32):
    for i in range(n_objects):
        s = ti.exp(-dt * damping)
        v[t, i] = s * v[t - 1, i] + v_inc[t, i] + dt * gravity * ti.Vector(
            [0.0, 1.0])
        x[t, i] = x[t - 1, i] + dt * v[t, i] + x_inc[t, i]
        omega[t, i] = s * omega[t - 1, i] + omega_inc[t, i]
        rotation[t, i] = rotation[t - 1,
                                  i] + dt * omega[t, i] + rotation_inc[t, i]


@ti.kernel
def advance_no_toi(t: ti.i32):
    for i in range(n_objects):
        s = ti.exp(-dt * damping)
        v[t, i] = s * v[t - 1, i] + v_inc[t, i] + dt * gravity * ti.Vector(
            [0.0, 1.0])
        x[t, i] = x[t - 1, i] + dt * v[t, i]
        omega[t, i] = s * omega[t - 1, i] + omega_inc[t, i]
        rotation[t, i] = rotation[t - 1, i] + dt * omega[t, i]


@ti.kernel
def compute_loss(t: ti.i32):
    loss[None] = (x[t, head_id] - goal[None]).norm()


gui = ti.GUI('Rigid Body Simulation', (512, 512), background_color=0xFFFFFF)


@ti.kernel
def sinusoidal_open_loop(t: ti.i32):
    for i in range(n_springs):
        amplitude = 150.0
        frequency = 12.0
        actuation[t, i] = ti.tanh(amplitude * ti.sin(frequency * t * dt))


def forward(output=None, visualize=True):

    initialize_properties()

    interval = vis_interval
    total_steps = steps
    if output:
        print(output)
        interval = output_vis_interval
        os.makedirs('rigid_body/{}/'.format(output), exist_ok=True)
        total_steps *= 2

    goal[None] = [0.9, 0.4]
    for t in range(1, total_steps):
        sinusoidal_open_loop(t - 1)
        collide(t - 1)
        apply_spring_force(t - 1)
        if use_toi:
            advance_toi(t)
        else:
            advance_no_toi(t)

        for i in range(n_objects):
            if x[t, i][0] != x[t, i][0] or v[t, i][0] != v[t, i][0]:
                print(f"NaN detected at t={t}, obj={i}: x={x[t, i]}, v={v[t, i]}")
                return

        if (t + 1) % interval == 0 and visualize:

            for i in range(n_objects):
                points = []
                for k in range(4):
                    offset_scale = [[-1, -1], [1, -1], [1, 1], [-1, 1]][k]
                    rot = rotation[t, i]
                    rot_matrix = np.array([[math.cos(rot), -math.sin(rot)],
                                           [math.sin(rot),
                                            math.cos(rot)]])

                    pos = np.array([x[t, i][0], x[t, i][1]
                                    ]) + offset_scale * rot_matrix @ np.array(
                                        [halfsize[i][0], halfsize[i][1]])

                    points.append((pos[0], pos[1]))

                for k in range(4):
                    gui.line(points[k],
                             points[(k + 1) % 4],
                             color=0x0,
                             radius=2)

            for i in range(n_springs):

                def get_world_loc(i, offset):
                    rot = rotation[t, i]
                    rot_matrix = np.array([[math.cos(rot), -math.sin(rot)],
                                           [math.sin(rot),
                                            math.cos(rot)]])
                    pos = np.array([[x[t, i][0]], [
                        x[t, i][1]
                    ]]) + rot_matrix @ np.array([[offset[0]], [offset[1]]])
                    return pos

                pt1 = get_world_loc(spring_anchor_a[i], spring_offset_a[i])
                pt2 = get_world_loc(spring_anchor_b[i], spring_offset_b[i])

                color = 0xFF2233

                if spring_actuation[i] != 0 and spring_length[i] != -1:
                    a = actuation[t - 1, i] * 0.5
                    color = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))

                if spring_length[i] == -1:
                    gui.line(pt1, pt2, color=0x000000, radius=9)
                    gui.line(pt1, pt2, color=color, radius=7)
                else:
                    gui.line(pt1, pt2, color=0x000000, radius=7)
                    gui.line(pt1, pt2, color=color, radius=5)

            gui.line((0.05, ground_height - 5e-3),
                     (0.95, ground_height - 5e-3),
                     color=0x0,
                     radius=5)

            file = None
            if output:
                file = f'rigid_body/{output}/{t:04d}.png'
            gui.show(file=file)

    loss[None] = 0
    compute_loss(steps - 1)


@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            v_inc[t, i] = ti.Vector([0.0, 0.0])
            x_inc[t, i] = ti.Vector([0.0, 0.0])
            rotation_inc[t, i] = 0.0
            omega_inc[t, i] = 0.0


def setup_robot(objects, springs, h_id):
    global head_id
    head_id = h_id
    global n_objects, n_springs
    n_objects = len(objects)
    n_springs = len(springs)

    print('n_objects=', n_objects, '   n_springs=', n_springs)

    initialize_states()

    for i in range(n_objects):
        x[0, i] = objects[i][0]
        halfsize[i] = [max(objects[i][1][0], 0.01), max(objects[i][1][1], 0.01)]
        rotation[0, i] = objects[i][2]

    for i in range(n_springs):
        s = springs[i]
        spring_anchor_a[i] = s[0]
        spring_anchor_b[i] = s[1]
        spring_offset_a[i] = s[2]
        spring_offset_b[i] = s[3]
        spring_length[i] = s[4]
        spring_stiffness[i] = max(s[5], 40.0)
        if s[6]:
            spring_actuation[i] = s[6]
        else:
            spring_actuation[i] = default_actuation
        #print(f"Spring {i}: a={s[0]}, b={s[1]}, len={spring_length[i]}, stiff={spring_stiffness[i]}, act={spring_actuation[i]}")


def optimize(robot, iterations=5, toi=True, visualize=False):
    global use_toi
    use_toi = toi

    objects, springs, head_id = robot
    setup_robot(objects, springs, head_id)

    losses = []
    for iter in range(iterations):
        clear_states()

        with ti.ad.Tape(loss):
            forward(visualize=visualize)
        if loss[None] != loss[None]:
            print(f"Iter={iter}, Loss=NaN - Simulation failed")

        total_norm_sqr_len = 0
        total_norm_sqr_stiff = 0
        for i in range(n_springs):
            if spring_length[i] == -1:
                continue
            len_grad = spring_length.grad[i]
            stiff_grad = spring_stiffness.grad[i]
            if len_grad != len_grad or stiff_grad != stiff_grad:
                print(f"Iter={iter}, NaN in gradients at spring {i}: len={len_grad}, stiff={stiff_grad}")
                return None, None
            len_grad = min(max(len_grad, -1000.0), 1000.0)
            stiff_grad = min(max(stiff_grad, -1000.0), 1000.0)
            total_norm_sqr_len += len_grad ** 2
            total_norm_sqr_stiff += stiff_grad ** 2

        print(f"Iter={iter}, Loss={loss[None]}, LenGradNorm={total_norm_sqr_len}, StiffGradNorm={total_norm_sqr_stiff}")

        gradient_clip = 1.0
        length_scale = learning_rate * min(
            1.0, gradient_clip / (total_norm_sqr_len**0.5 + 1e-4))
        stiff_scale = learning_rate * min(
            1.0, gradient_clip / (total_norm_sqr_stiff**0.5 + 1e-4))
        length_scale = max(length_scale, 1e-6)
        stiff_scale = max(stiff_scale, 1e-6)

        print(f"Scales: length={length_scale}, stiff={stiff_scale}")
        for i in range(n_springs):
            if spring_length[i] == -1:
                continue

            spring_length[i] -= length_scale * spring_length.grad[i]
            spring_length[i] = max(spring_length[i], 0.01)

            spring_stiffness[i] -= stiff_scale * spring_stiffness.grad[i]
            spring_stiffness[i] = max(spring_stiffness[i], 0.01)

        losses.append(loss[None])

    opt_robot = (
        [obj.copy() for obj in objects],
        [[spring_anchor_a[i], spring_anchor_b[i], spring_offset_a[i].to_numpy(),
          spring_offset_b[i].to_numpy(), spring_length[i], spring_stiffness[i],
          spring_actuation[i]] for i in range(n_springs)],
        head_id
    )

    return opt_robot, losses


def generate_robot(base):
    objects, springs, head_id = base()

    objects = [[list(obj[0]), list(obj[1]), obj[2]] for obj in copy.deepcopy(objects)]
    springs = copy.deepcopy(springs)

    for obj in objects:
        # Halfsize
        obj[1][0] = max(obj[1][0] * np.random.uniform(0.5, 1.5), 0.02)
        obj[1][1] = max(obj[1][1] * np.random.uniform(0.5, 1.5), 0.02)

        '''
        # Position
        obj[0][0] += np.random.uniform(-0.05, 0.05)
        obj[0][0] = max(obj[0][0], 0.05)
        obj[0][1] += np.random.uniform(-0.05, 0.05)
        obj[0][1] = max(obj[0][1], ground_height + obj[1][1] + 0.01)
        '''

    for spr in springs:
        if spr[4] != -1:
            spr[4] = max(spr[4] * np.random.uniform(0.5, 1.5), 0.02)
        spr[5] = max(spr[5] * np.random.uniform(0.5, 1.5), 0.02)
        if spr[6] != 0:
            spr[6] *= np.random.uniform(0.5, 1.5)

    return objects, springs, head_id


def eval_fitness(objects, springs, head_id):
    setup_robot(objects, springs, head_id)
    clear_states()
    forward(visualize=False)

    return -loss[None]


def crossover(parent1, parent2):
    robot1, _ = parent1
    robot2, _ = parent2
    objects1, springs1, head_id1 = robot1
    objects2, springs2, head_id2 = robot2
    min_objects = min(len(objects1), len(objects2))
    min_springs = min(len(springs1), len(springs2))

    child_objects = []
    for i in range(min_objects):
        x = [(a + b) / 2 for a, b in zip(objects1[i][0], objects2[i][0])]
        halfsize = [(a + b) / 2 for a, b in zip(objects1[i][1], objects2[i][1])]
        rotation = (objects1[i][2] + objects2[i][2]) / 2
        x[1] = max(x[1], ground_height + halfsize[1] + 0.01)
        child_objects.append([x, halfsize, rotation])
    child_springs = []
    for i in range(min_springs):
        spr = springs1[i].copy()
        if springs1[i][4] != -1 and springs2[i][4] != -1:
            spr[4] = (springs1[i][4] + springs2[i][4]) / 2
        else:
            spr[4] = -1
        spr[5] = (springs1[i][5] + springs2[i][5]) / 2
        spr[6] = (springs1[i][6] + springs2[i][6]) / 2
        child_springs.append(spr)
    child_head_id = np.random.choice([head_id1, head_id2])

    return child_objects, child_springs, child_head_id


def mutate(robot_config):
    objects, springs, head_id = robot_config
    for obj in objects:
        # Halfsize
        obj[1][0] = max(obj[1][0] * np.random.uniform(0.9, 1.1), 0.02)
        obj[1][1] = max(obj[1][1] * np.random.uniform(0.9, 1.1), 0.02)
    for spr in springs:
        if spr[4] != -1:
            spr[4] += np.random.uniform(-0.01, 0.01)
            spr[4] = max(spr[4], 0.02)
        spr[5] += np.random.uniform(-0.1, 0.1)
        spr[5] = max(spr[5], 0.01)
    return objects, springs, head_id


def evolution(population, generations=5):
    fitness = []
    pop_size = len(population)
    for gen in range(generations):
        new_pop = []
        for i in range(pop_size):
            p1, p2 = np.random.choice(len(population), 2, replace=False)
            p1, p2 = population[p1], population[p2]
            child_config = crossover(p1, p2)
            child_config = mutate(child_config)
            new_pop.append(child_config)

        opt_pop = []
        for robot_config in new_pop:
            opt_robot, losses = optimize(robot_config, iterations=5, visualize=False)
            if opt_robot is not None:
                opt_pop.append((opt_robot, losses))

        if not opt_pop:
            opt_pop = population[:5]
            print(f"Generation {gen}: All optimizations failed, reusing parents")

        scores = []
        count = 0
        for robot, losses in opt_pop:
            setup_robot(*robot)
            clear_states()
            forward(output=f"gen{gen}/robot{count}", visualize=True)
            score = -loss[None]
            scores.append(score)
            count += 1

        top = np.argsort(scores)[-5:]
        population = [opt_pop[i] for i in top]
        fitness.append(max(scores))
        print(f"Generation {gen}: Best Fitness = {max(scores)}")

    return population, fitness


def main():
    allocate_fields()

    population = [(generate_robot(np.random.choice(robots)), None) for i in range(10)]
    best_pop, fitness = evolution(population, generations=10)

    best_robot, best_losses = best_pop[0]
    setup_robot(*best_robot)
    forward('evolved_best', visualize=True)

    plt.plot(fitness)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Evolutionary Fitness Over Generations")
    plt.show()


if __name__ == '__main__':
    main()