import adv_utils

class advantage:

    def __init__(self, metrics):
        self.w1 = metrics.w1
        self.w2 = metrics.w2
        self.w3 = metrics.w3

    def compute_stability_advantage(state, action):
        # 重心稳定性
        cog_stability = compute_center_of_gravity_shift(state, action)
        
        # 接触面积稳定性
        contact_stability = evaluate_contact_quality(placed_objects, new_object)
        
        # 群体相对稳定性（GRPO核心）
        relative_stability = compare_group_stability(
            current_group=state.placed_objects,
            alternative_groups=get_alternative_placements(state, action)
        )
        
        return w1·cog_stability + w2·contact_stability + w3·relative_stability

    def compute_efficiency_advantage(state, action):
        # 即时空间利用率
        immediate_utilization = volume_utilization_gain(state, action)
        
        # 紧密度奖励（不规则物体的形状契合）
        shape_fitting = evaluate_geometric_compatibility(
            new_object, neighboring_objects
        )
        
        return w₁·immediate_utilization + w₃·shape_fitting

    def compute_feasibility_advantage(state, action):
        # 物理约束满足度
        physics_score = evaluate_physics_constraints(state, action)
        
        # 碰撞惩罚
        collision_penalty = compute_collision_cost(new_placement)
        
        # 访问性（后续物体的可放置性）
        accessibility = evaluate_remaining_space_accessibility(state_after_action)
        
        return w₁·physics_score - w₂·collision_penalty + w₃·accessibility

    def compute_future_advantage(state, action):
        # 蒙特卡洛树搜索估计
        future_value = estimate_future_packing_potential(state_after_action)
        
        # 剩余物体的装载难度变化
        remaining_difficulty = assess_remaining_objects_complexity(
            state.remaining_objects, state_after_action
        )
        
        # 空间约束的变化趋势
        constraint_evolution = predict_constraint_tightening(state_after_action)
        
        return w₁·future_value - w₂·remaining_difficulty - w₃·constraint_evolution