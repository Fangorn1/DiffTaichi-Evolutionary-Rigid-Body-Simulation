import math

def add_object(objects, x, halfsize, rotation=0):
    objects.append([x, halfsize, rotation])
    return len(objects) - 1


# actuation 0.0 will be translated into default actuation
def add_spring(springs, a, b, offset_a, offset_b, length, stiffness, actuation=0.0):
    springs.append([a, b, offset_a, offset_b, length, stiffness, actuation])


def build_robot_skeleton_tree(tree_depth=2):
    objects = []
    springs = []
    pos = [.3, 0.7]
    size = [0.02, 0.02]
    l = 0.1
    s = 15
    new_obj_id = 1
    add_object(objects, x=pos, halfsize=size)
    parents = {0: pos}
    while tree_depth > 0:
        nodes = parents
        parents = {}
        for node in nodes.keys():
            child1_pos = [nodes[node][0] - 0.06, nodes[node][1] - 0.08]
            add_object(objects, x=child1_pos, halfsize=size)
            add_spring(springs, node, new_obj_id, [0.0, 0.00], [0.0, 0.0], l, s)
            parents[new_obj_id] = child1_pos
            new_obj_id += 1

            child2_pos = [nodes[node][0] + 0.06, nodes[node][1] - 0.08]
            add_object(objects, x=child2_pos, halfsize=size)
            add_spring(springs, node, new_obj_id, [0.0, 0.00], [0.0, 0.0], l, s)
            parents[new_obj_id] = child2_pos
            new_obj_id += 1
        tree_depth -= 1

    return objects, springs, 0


def build_robot_skeleton(leg_segments=2):
    objects = []
    springs = []
    pos = [0.3, 0.6]
    size = [0.15, 0.05]
    l = 0.05
    s = 60.0
    new_obj_id = 1
    add_object(objects, x=pos, halfsize=size)

    leg_size = [0.02, 0.05]
    offset = [-size[0] + leg_size[0], -size[1]]
    leg_pos = [pos[0] - size[0] + leg_size[0], pos[1] - size[1] - leg_size[1]]

    for leg in range(3):
        add_object(objects, x=leg_pos, halfsize=leg_size)
        add_spring(springs, 0, new_obj_id, offset, [0.0, 0.0], l, s)
        add_spring(springs, 0, new_obj_id, offset, [0.0, leg_size[1]], -1, s)

        parent_pos = leg_pos
        leg_pos = [leg_pos[0] + size[0], leg_pos[1]]
        offset = [offset[0] + size[0] - leg_size[0], offset[1]]

        parent_id = new_obj_id
        new_obj_id += 1

        for seg in range(1, leg_segments):
            child_pos = [parent_pos[0], parent_pos[1] - l*2]
            add_object(objects, x=child_pos, halfsize=leg_size)
            add_spring(springs, parent_id, new_obj_id, [0.0, 0.0], [0.0, 0.0], l*2, s)
            add_spring(springs, parent_id, new_obj_id, [0.0, -l], [0.0, l], -1, s)
            parent_pos = child_pos
            parent_id = new_obj_id
            new_obj_id += 1

    return objects, springs, 0


def robotA():
    objects = []
    springs = []
    add_object(x=[0.3, 0.25], halfsize=[0.15, 0.03])
    add_object(x=[0.2, 0.15], halfsize=[0.03, 0.02])
    add_object(x=[0.3, 0.15], halfsize=[0.03, 0.02])
    add_object(x=[0.4, 0.15], halfsize=[0.03, 0.02])
    add_object(x=[0.4, 0.3], halfsize=[0.005, 0.03])

    l = 0.12
    s = 15
    add_spring(0, 1, [-0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 1, [-0.1, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 2, [-0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 2, [0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 3, [0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 3, [0.1, 0.00], [0.0, 0.0], l, s)
    # -1 means the spring is a joint
    add_spring(0, 4, [0.1, 0], [0, -0.05], -1, s)

    return objects, springs, 0


def robotC():
    objects = []
    springs = []
    add_object(x=[0.3, 0.25], halfsize=[0.15, 0.03])
    add_object(x=[0.2, 0.15], halfsize=[0.03, 0.02])
    add_object(x=[0.3, 0.15], halfsize=[0.03, 0.02])
    add_object(x=[0.4, 0.15], halfsize=[0.03, 0.02])

    l = 0.12
    s = 15
    add_spring(0, 1, [-0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 1, [-0.1, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 2, [-0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 2, [0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 3, [0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 3, [0.1, 0.00], [0.0, 0.0], l, s)

    return objects, springs, 3


l_thigh_init_ang = 10
l_calf_init_ang = -10
r_thigh_init_ang = 10
r_calf_init_ang = -10
initHeight = 0.15

hip_pos = [0.3, 0.5 + initHeight]
thigh_half_length = 0.11
calf_half_length = 0.11

foot_half_length = 0.08


def rotAlong(half_length, deg, center):
    ang = math.radians(deg)
    return [
        half_length * math.sin(ang) + center[0],
        -half_length * math.cos(ang) + center[1]
    ]


half_hip_length = 0.08


def robotLeg():
    objects = []
    springs = []
    #hip
    add_object(hip_pos, halfsize=[0.06, half_hip_length])
    hip_end = [hip_pos[0], hip_pos[1] - (half_hip_length - 0.01)]

    #left
    l_thigh_center = rotAlong(thigh_half_length, l_thigh_init_ang, hip_end)
    l_thigh_end = rotAlong(thigh_half_length * 2.0, l_thigh_init_ang, hip_end)
    add_object(l_thigh_center,
               halfsize=[0.02, thigh_half_length],
               rotation=math.radians(l_thigh_init_ang))
    add_object(rotAlong(calf_half_length, l_calf_init_ang, l_thigh_end),
               halfsize=[0.02, calf_half_length],
               rotation=math.radians(l_calf_init_ang))
    l_calf_end = rotAlong(2.0 * calf_half_length, l_calf_init_ang, l_thigh_end)
    add_object([l_calf_end[0] + foot_half_length, l_calf_end[1]],
               halfsize=[foot_half_length, 0.02])

    #right
    add_object(rotAlong(thigh_half_length, r_thigh_init_ang, hip_end),
               halfsize=[0.02, thigh_half_length],
               rotation=math.radians(r_thigh_init_ang))
    r_thigh_end = rotAlong(thigh_half_length * 2.0, r_thigh_init_ang, hip_end)
    add_object(rotAlong(calf_half_length, r_calf_init_ang, r_thigh_end),
               halfsize=[0.02, calf_half_length],
               rotation=math.radians(r_calf_init_ang))
    r_calf_end = rotAlong(2.0 * calf_half_length, r_calf_init_ang, r_thigh_end)
    add_object([r_calf_end[0] + foot_half_length, r_calf_end[1]],
               halfsize=[foot_half_length, 0.02])

    s = 200

    thigh_relax = 0.9
    leg_relax = 0.9
    foot_relax = 0.7

    thigh_stiff = 5
    leg_stiff = 20
    foot_stiff = 40

    #left springs
    add_spring(0, 1, [0, (half_hip_length - 0.01) * 0.4],
               [0, -thigh_half_length],
               thigh_relax * (2.0 * thigh_half_length + 0.22), thigh_stiff)
    add_spring(1, 2, [0, thigh_half_length], [0, -thigh_half_length],
               leg_relax * 4.0 * thigh_half_length, leg_stiff, 0.08)
    add_spring(
        2, 3, [0, 0], [foot_half_length, 0],
        foot_relax *
        math.sqrt(pow(thigh_half_length, 2) + pow(2.0 * foot_half_length, 2)),
        foot_stiff)

    add_spring(0, 1, [0, -(half_hip_length - 0.01)], [0.0, thigh_half_length],
               -1, s)
    add_spring(1, 2, [0, -thigh_half_length], [0.0, thigh_half_length], -1, s)
    add_spring(2, 3, [0, -thigh_half_length], [-foot_half_length, 0], -1, s)

    #right springs
    add_spring(0, 4, [0, (half_hip_length - 0.01) * 0.4],
               [0, -thigh_half_length],
               thigh_relax * (2.0 * thigh_half_length + 0.22), thigh_stiff)
    add_spring(4, 5, [0, thigh_half_length], [0, -thigh_half_length],
               leg_relax * 4.0 * thigh_half_length, leg_stiff, 0.08)
    add_spring(
        5, 6, [0, 0], [foot_half_length, 0],
        foot_relax *
        math.sqrt(pow(thigh_half_length, 2) + pow(2.0 * foot_half_length, 2)),
        foot_stiff)

    add_spring(0, 4, [0, -(half_hip_length - 0.01)], [0.0, thigh_half_length],
               -1, s)
    add_spring(4, 5, [0, -thigh_half_length], [0.0, thigh_half_length], -1, s)
    add_spring(5, 6, [0, -thigh_half_length], [-foot_half_length, 0], -1, s)

    return objects, springs, 3


def robotB():
    objects = []
    springs = []
    body = add_object([0.15, 0.25], [0.1, 0.03])
    back = add_object([0.08, 0.22], [0.03, 0.10])
    front = add_object([0.22, 0.22], [0.03, 0.10])

    rest_length = 0.22
    stiffness = 50
    act = 0.03
    add_spring(body,
               back, [0.08, 0.02], [0.0, -0.08],
               rest_length,
               stiffness,
               actuation=act)
    add_spring(body,
               front, [-0.08, 0.02], [0.0, -0.08],
               rest_length,
               stiffness,
               actuation=act)

    add_spring(body, back, [-0.08, 0.0], [0.0, 0.03], -1, stiffness)
    add_spring(body, front, [0.08, 0.0], [0.0, 0.03], -1, stiffness)

    return objects, springs, body


robots = [build_robot_skeleton]
