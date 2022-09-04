from __future__ import annotations

import random
from typing import Any

from gym import Env
from gym.spaces import Box
from gym.spaces import Discrete
from numpy.typing import NDArray

import numpy as np


class MultiDriller(Env):  # type: ignore
    """Simple driller environment for multiple wells"""

    def __init__(self, env_config: dict[str, Any]) -> None:
        """Initialize environment with config dictionary."""

        self.model = np.loadtxt(env_config["model_path"],
                                delimiter=env_config["delim"])

        self.nrow, self.ncol = self.model.shape
        self.state = np.zeros((self.nrow, self.ncol), dtype=bool)

        self.available_pipe = env_config["available_pipe"]

        self.num_wells = env_config["num_wells"]

        self.wells_drilled = 0
        self.reward = 0
        self.multi_reward = 0

        self.production = 0
        self.pipe_used = 0
        self.trajectory: list[list[int]] = []
        self.bit_location: list[int] = []
        self.surface_location = []
        self.last_action = None

        self.multi_trajectory: list[list[list[int]]] = []
        self.action_space = Discrete(4)

        self.observation_space = Box(low=0, high=1,
                                     shape=(self.nrow, self.ncol),
                                     dtype="bool")
        self.reset_well()
        self.reset()

    # ----------------------------------------------------------------------------------------------------------------

    def step(self, action: int) -> tuple[NDArray[np.bool_], int, bool, dict[str, Any]]:
        """Take step based on action."""

        done = False
        #         reward = 0
        #         self.reset_well()

        actions = {
            0: [1, 0],  # down
            1: [0, -1],  # left
            2: [0, 1],  # right
            3: [-1, 0],  # up
        }

        dz_dx = actions[action]
        new_location = [prev + now for prev, now in zip(self.bit_location, dz_dx)]

        self.bit_location = new_location

        self.trajectory.append(new_location)
        newrow, newcol = new_location

        self.pipe_used += 1

        if newrow < 1 or newrow >= self.nrow:
            done = True
            self.reward = -100
        #             print('    Number of Rows exceeded')
        #             reward = 0

        elif newcol < 0 or newcol >= self.ncol:
            done = True
            self.reward = -100
        #             print('    Number of Cols exceeded')
        #             reward = 0

        else:
            self.reward = self.model[newrow, newcol] + self.pipe_used / 2
            if len(self.trajectory) > 0:
                self.update_state()

        if self.pipe_used == self.available_pipe:
            done = True
            self.reward = 0
        #             print('    Done with total pipes')

        if self.bit_location in self.trajectory[:-1]:
            done = True
            self.reward = -100
        #             print('    Crashed onto itself')
        #             reward = 0

        if self.bit_location in [item for sublist in self.multi_trajectory for item in sublist]:
            done = True
            self.reward = -100
        #             print('    Crashed into a different well')

        # Avoid immediate 180 degree turns
        if (self.last_action != None):
            if (np.add(actions[action], actions[self.last_action]).tolist() == [0, 0]):
                #                 done = True
                self.reward = -100
            #                 print('    Immediate 180 degree turn')

        info: dict[str, Any] = {}

        if done:
            self.wells_drilled += 1
            self.multi_reward += self.reward

            if len(self.trajectory) > 0:
                self.multi_trajectory.append(self.trajectory)

            self.reset_well()

            if self.wells_drilled < self.num_wells:
                done = False

            return self.state, self.multi_reward, done, info
        else:
            self.last_action = action
            #             print(f'Last action: {actions[self.last_action]}')
            return self.state, self.reward, done, info

    # ----------------------------------------------------------------------------------------------------------------

    def update_state(self) -> None:
        """Update state method."""
        traj_i, traj_j = np.asarray(self.trajectory).T
        self.state[traj_i, traj_j] = 1

    # ----------------------------------------------------------------------------------------------------------------

    def render(self) -> None:
        """Gym environment rendering."""
        raise NotImplementedError("No renderer implemented yet.")

    # ----------------------------------------------------------------------------------------------------------------

    def reset_well(self) -> NDArray[np.bool_]:
        """Reset the status of the environment."""

        # random surface location  that was not used before
        self.surface_hole_location = [0, random.choice(list(set(range(0, self.ncol - 1)) - set(self.surface_location)))]

        # Log the surface locations already used
        self.surface_location.append(self.surface_hole_location[1])

        self.bit_location = self.surface_hole_location
        self.trajectory = [self.surface_hole_location]
        self.pipe_used = 0
        self.reward = 0

        return self.state

    # ----------------------------------------------------------------------------------------------------------------

    def reset(self) -> NDArray[np.bool_]:

        """Reset the status of the environment."""
        self.state = np.zeros((self.nrow, self.ncol), dtype=bool)
        self.multi_trajectory = []
        self.surface_location = []
        self.multi_reward = 0
        self.wells_drilled = 0
        return self.state
