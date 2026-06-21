from __future__ import annotations

import math
from typing import Any

import numpy as np


# --- CONFIGURATION ---
MPC_CONFIG = {
    "dt": 0.1,
    "horizon": 22,
    "near_goal_distance": 1.0,
}

# Expanded grid for better precision
CONTROL_GRID = {
    "slow_speeds": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "cruise_speeds": [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 2.9, 3.2],
    "turn_rates": [-1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5],
}

# Tuned for "Time to Goal" (minimized travel time)
WEIGHTS = {
    "goal_position": 1.6,
    "goal_heading": 0.08,
    "turn_rate": 0.05,
    "terminal_goal": 36.0,
    "collision": 8000.0,
    "collision_depth": 1600.0,
    "near_obstacle": 18.0,
    "far_obstacle": 0.8,
    "stopping": 20.0,
}

SAFETY = {
    "near_obstacle_distance": 0.45,
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
    speeds = CONTROL_GRID["cruise_speeds"] if distance_to_goal >= MPC_CONFIG["near_goal_distance"] else CONTROL_GRID["slow_speeds"]

    target_heading = math.atan2(goal[1] - state[1], goal[0] - state[0])
    heading_error = angle_difference(target_heading, state[2])
    proportional_turn = clamp(2.0 * heading_error, -1.0, 1.0)

    turn_rates = CONTROL_GRID["turn_rates"]  # Use the full turn rate grid
    turn_rates = np.append(turn_rates, [proportional_turn])
    turn_rates = np.unique(turn_rates)

    candidates = []
    for speed in speeds:
        for turn_rate in turn_rates:
            control = np.array([speed, turn_rate], dtype=float)
            control[0] = clamp(control[0], control_limits["v_min"], control_limits["v_max"])
            control[1] = clamp(control[1], -control_limits["omega_max"], control_limits["omega_max"])
            candidates.append(control)

    return candidates


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

        for obstacle in active_obstacles:
            predicted_obstacle = predict_obstacle(obstacle, step_index * float(MPC_CONFIG["dt"]))
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
