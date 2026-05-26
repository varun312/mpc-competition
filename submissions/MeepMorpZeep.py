from __future__ import annotations

import math
from typing import Any

import numpy as np


# This file is meant to be the starter submission students can edit.
#
# The controller below is a simple sampling MPC:
# 1. Build a small menu of possible controls.
# 2. Simulate each control for a short horizon.
# 3. Score each simulated path by goal distance, steering effort, and obstacle risk.
# 4. Return the first control from the lowest-cost simulated path.
#
# There is no optimizer here. The "MPC" part is the repeated predict, score,
# and execute loop inside act(). This keeps the code easy to change while still
# giving students the same tuning ideas as a larger MPC controller.


# Main prediction settings. Increasing the horizon sees farther ahead, but each
# act() call tests more simulated steps and may run slower.
MPC_CONFIG = {
    "dt": 0.1,
    "horizon": 19,
    "near_goal_distance": 1.0,
}

# Discrete controls to try at every step. A real optimizer would search over a
# continuous control sequence. This baseline searches a small grid instead.
CONTROL_GRID = {
    "slow_speeds": [0.0, 0.3, 0.6, 0.9],
    "cruise_speeds": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "turn_rates": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
}

# Cost weights. Larger values make that term matter more during rollout
# scoring. These are the first numbers worth tuning.
WEIGHTS = {
    "goal_position": 1.6,
    "goal_heading": 0.08,
    "turn_rate": 0.05,
    "terminal_goal": 36.0,
    "collision": 8000.0,
    "collision_depth": 1600.0,
    "near_obstacle": 18.0,
    "far_obstacle": 1.2,
    "stopping": 20.0,
}

# Clearance bands around obstacles. Positive clearance means no collision.
# Negative clearance means the robot overlaps the obstacle.
SAFETY = {
    "near_obstacle_distance": 0.50,
    "far_obstacle_distance": 1.2,
}


class Agent:
    def __init__(self) -> None:
        self.previous_control = np.array([0.0, 0.0], dtype=float)

    def act(self, observation: dict[str, Any]) -> list[float]:
        state = np.array(observation["state"], dtype=float)
        goal = np.array(observation["goal"], dtype=float)
        active_obstacles = get_active_obstacles(observation)
        control_limits = observation["control_limits"]
        robot_radius = float(observation["robot_radius"])

        best_control = np.array([0.0, 0.0], dtype=float)
        best_cost = float("inf")

        # This is the receding horizon loop. Each candidate is held constant in
        # the prediction, but only the first control is used by the simulator.
        # The next call to act() repeats the search from the new state.
        for candidate_control in candidate_controls(state, goal, control_limits):
            rollout_cost = evaluate_rollout(
                state,
                goal,
                candidate_control,
                self.previous_control,
                active_obstacles,
                control_limits,
                robot_radius,
            )
            if rollout_cost < best_cost:
                best_cost = rollout_cost
                best_control = candidate_control

        self.previous_control = apply_control_limits(
            best_control, self.previous_control, control_limits
        )
        return [float(best_control[0]), float(best_control[1])]


def candidate_controls(
    state: np.ndarray, goal: np.ndarray, control_limits: dict[str, float]
) -> list[np.ndarray]:
    distance_to_goal = float(np.linalg.norm(goal[:2] - state[:2]))
    speeds = CONTROL_GRID["cruise_speeds"]
    if distance_to_goal < float(MPC_CONFIG["near_goal_distance"]):
        # Close to a waypoint, slower controls reduce overshoot.
        speeds = CONTROL_GRID["slow_speeds"]

    target_heading = math.atan2(goal[1] - state[1], goal[0] - state[0])
    heading_error = angle_difference(target_heading, state[2])
    proportional_turn = clamp(2.0 * heading_error, -1.0, 1.0)

    # The fixed grid explores simple turns. The proportional turns bias the
    # search toward the current waypoint, so the baseline is not pure brute
    # force.
    turn_rates = set(CONTROL_GRID["turn_rates"])
    turn_rates.add(round(proportional_turn, 3))
    turn_rates.add(round(proportional_turn - 0.35, 3))
    turn_rates.add(round(proportional_turn + 0.35, 3))

    return [
        np.array(
            [
                clamp(speed, control_limits["v_min"], control_limits["v_max"]),
                clamp(
                    turn_rate, -control_limits["omega_max"], control_limits["omega_max"]
                ),
            ],
            dtype=float,
        )
        for speed in speeds
        for turn_rate in sorted(turn_rates)
    ]


def evaluate_rollout(
    state: np.ndarray,
    goal: np.ndarray,
    requested_control: np.ndarray,
    previous_control: np.ndarray,
    active_obstacles: list[dict[str, float]],
    control_limits: dict[str, float],
    robot_radius: float,
) -> float:
    predicted_state = state.copy()
    predicted_control = previous_control.copy()
    total_cost = 0.0

    for step_index in range(1, int(MPC_CONFIG["horizon"]) + 1):
        # Rate limits matter in this problem, so the predicted control is
        # ramped exactly like the simulator will ramp it.
        predicted_control = apply_control_limits(
            requested_control, predicted_control, control_limits
        )
        predicted_state = step_unicycle(predicted_state, predicted_control)

        goal_distance = float(np.linalg.norm(goal[:2] - predicted_state[:2]))
        heading_error = abs(angle_difference(goal[2], predicted_state[2]))
        total_cost += WEIGHTS["goal_position"] * goal_distance
        total_cost += WEIGHTS["goal_heading"] * heading_error
        total_cost += WEIGHTS["turn_rate"] * abs(predicted_control[1])

        prediction_time = step_index * float(MPC_CONFIG["dt"])
        for obstacle in active_obstacles:
            # Dynamic scenes give obstacle velocity. Predicting moving
            # obstacles is the main reason this baseline beats a pure waypoint
            # tracker.
            predicted_obstacle = predict_obstacle(obstacle, prediction_time)
            clearance = rectangle_clearance(
                predicted_state, predicted_obstacle, robot_radius
            )
            total_cost += collision_cost(clearance)

    terminal_distance = float(np.linalg.norm(goal[:2] - predicted_state[:2]))
    total_cost += WEIGHTS["terminal_goal"] * terminal_distance
    if requested_control[0] < 0.2 and terminal_distance > 0.7:
        total_cost += WEIGHTS["stopping"]

    return float(total_cost)


def get_active_obstacles(observation: dict[str, Any]) -> list[dict[str, float]]:
    """Return the nearest obstacles selected by the simulator."""
    active_indices = set(observation["active_obstacle_indices"])
    return [
        obstacle
        for obstacle_index, obstacle in enumerate(observation["obstacles"])
        if obstacle_index in active_indices
    ]


def collision_cost(clearance: float) -> float:
    """Convert robot-obstacle clearance into a soft penalty."""
    if clearance < 0.0:
        return WEIGHTS["collision"] + WEIGHTS["collision_depth"] * abs(clearance)
    if clearance < SAFETY["near_obstacle_distance"]:
        return (
            WEIGHTS["near_obstacle"]
            * (SAFETY["near_obstacle_distance"] - clearance) ** 2
        )
    if clearance < SAFETY["far_obstacle_distance"]:
        return (
            WEIGHTS["far_obstacle"] * (SAFETY["far_obstacle_distance"] - clearance) ** 2
        )
    return 0.0


def apply_control_limits(
    requested_control: np.ndarray,
    previous_control: np.ndarray,
    control_limits: dict[str, float],
) -> np.ndarray:
    time_step = float(MPC_CONFIG["dt"])
    requested_speed = clamp(
        requested_control[0], control_limits["v_min"], control_limits["v_max"]
    )
    requested_turn_rate = clamp(
        requested_control[1], -control_limits["omega_max"], control_limits["omega_max"]
    )

    maximum_speed_change = control_limits["a_max"] * time_step
    maximum_turn_change = control_limits["omega_dot_max"] * time_step
    return np.array(
        [
            clamp(
                requested_speed,
                previous_control[0] - maximum_speed_change,
                previous_control[0] + maximum_speed_change,
            ),
            clamp(
                requested_turn_rate,
                previous_control[1] - maximum_turn_change,
                previous_control[1] + maximum_turn_change,
            ),
        ],
        dtype=float,
    )


def step_unicycle(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Advance [x, y, theta] using the unicycle model."""
    time_step = float(MPC_CONFIG["dt"])
    return np.array(
        [
            state[0] + control[0] * math.cos(state[2]) * time_step,
            state[1] + control[0] * math.sin(state[2]) * time_step,
            normalize_angle(state[2] + control[1] * time_step),
        ],
        dtype=float,
    )


def predict_obstacle(
    obstacle: dict[str, float], prediction_time: float
) -> dict[str, float]:
    obstacle_angle = float(obstacle["angle"])
    obstacle_speed = float(obstacle["v"])
    return {
        "cx": float(obstacle["cx"])
        + obstacle_speed * math.cos(obstacle_angle) * prediction_time,
        "cy": float(obstacle["cy"])
        + obstacle_speed * math.sin(obstacle_angle) * prediction_time,
        "width": float(obstacle["width"]),
        "height": float(obstacle["height"]),
        "angle": normalize_angle(
            obstacle_angle + float(obstacle["omega"]) * prediction_time
        ),
    }


def rectangle_clearance(
    state: np.ndarray, obstacle: dict[str, float], robot_radius: float
) -> float:
    # Work in the obstacle frame so a rotated rectangle can be checked like a
    # plain rectangle. The signed distance formula then gives positive
    # clearance outside the rectangle and negative clearance inside it.
    translated_x = state[0] - obstacle["cx"]
    translated_y = state[1] - obstacle["cy"]
    cos_angle = math.cos(-obstacle["angle"])
    sin_angle = math.sin(-obstacle["angle"])
    local_x = translated_x * cos_angle - translated_y * sin_angle
    local_y = translated_x * sin_angle + translated_y * cos_angle

    outside_x = abs(local_x) - obstacle["width"] / 2.0
    outside_y = abs(local_y) - obstacle["height"] / 2.0
    outside_distance = math.hypot(max(outside_x, 0.0), max(outside_y, 0.0))
    inside_distance = min(max(outside_x, outside_y), 0.0)
    return outside_distance + inside_distance - robot_radius


def clamp(value: float, lower_bound: float, upper_bound: float) -> float:
    return max(lower_bound, min(upper_bound, value))


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_difference(target_angle: float, source_angle: float) -> float:
    return normalize_angle(target_angle - source_angle)
