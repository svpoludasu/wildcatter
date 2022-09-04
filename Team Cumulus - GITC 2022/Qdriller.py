from __future__ import annotations
from typing import Any
import numpy as np



class QDriller:  # type: ignore
    """Driller environment for horizontal wells with self.rewards based on Q learning"""

    def __init__(self, env_config: dict[str, Any]) -> None:
        """Initialize environment with config dictionary."""

        self.rewards = np.loadtxt(env_config["model_path"],
                                  delimiter=env_config["delim"])

        self.available_pipe = env_config["available_pipe"]

        # Normalizing the model
        self.rewards = self.rewards * (100 / self.rewards.max())

        self.rewards[np.less(self.rewards, 0)] = -100
        self.rewards[self.rewards == 0] = -1

        self.actions = ['up', 'right', 'down', 'left']

        self.q_values = np.zeros((self.rewards.shape[0],
                                  self.rewards.shape[1],
                                  len(self.actions)))

        self.trajectory = []
        self.end = 0

        self.action_cache = np.nan

    # ----------------------------------------------------------------------------------------------------------------

    # define a function that determines if the specified location is a terminal state
    def is_terminal_state(self, current_row_index, current_column_index):
        if ((len(self.trajectory) > 1) &
                (self.rewards[current_row_index, current_column_index] == -100)):
            self.end = 1
            return True
        else:
            return False

    # ----------------------------------------------------------------------------------------------------------------

    # define a function that will choose a random starting location
    def get_starting_location(self):
        # get a random column index
        current_row_index = np.random.randint(self.rewards.shape[0])
        current_column_index = np.random.randint(self.rewards.shape[1])
        return current_row_index, current_column_index
    # ----------------------------------------------------------------------------------------------------------------

    # numeric action codes: 0 = up, 1 = right, 2 = down, 3 = left
    # define a function that will decide the valid actions to avoid crashing into itself
    def get_valid_actions(self, current_row_index, current_column_index):
        va = [0, 1, 2, 3]
        try:
            # Avoid turning back into itself
            if [current_row_index - 1, current_column_index] in self.trajectory:
                va.remove(0)
            if [current_row_index, current_column_index + 1] in self.trajectory:
                va.remove(1)
            if [current_row_index + 1, current_column_index] in self.trajectory:
                va.remove(2)
            if [current_row_index, current_column_index - 1] in self.trajectory:
                va.remove(3)

            # Remove left move if it is the first column
            if current_column_index == 0:
                va.remove(3)

            #             # Remove up move if it is the first row
            #             if current_row_index == 0:
            #                 va.remove(0)

            # Force to move down when at surface
            if current_row_index == 0:
                return [2]

            # Remove right move if it is the last column
            if current_column_index == (self.rewards.shape[1] - 1):
                va.remove(1)

            # Remove down move if it is the last row
            if current_row_index == (self.rewards.shape[0] - 1):
                va.remove(2)

            # Avoid going up if is gonna hit the surface
            if (current_row_index - 1) == 0:
                va.remove(0)

        #             # Avoid wellbore looping
        #             if self.action_cache.notna():
        #                 va.remove(self.action_cache)

        except:
            #             self.end = 1
            pass

        return va

    # ----------------------------------------------------------------------------------------------------------------

    # define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
    def get_next_action(self, current_row_index, current_column_index, epsilon):

        valid_actions = self.get_valid_actions(current_row_index, current_column_index)

        if len(valid_actions) == 0:
            self.end = 1

        if (len(valid_actions) != 0) & (np.random.random() < epsilon):
            action = max(valid_actions,
                         key=lambda i: self.q_values[current_row_index, current_column_index].tolist()[i])
            #             print(f'Valid Actions: {valid_actions}, Picked Action: {action}')
            return action
        else:
            return np.random.randint(len(self.actions))

    # ----------------------------------------------------------------------------------------------------------------

    def get_next_action_train(self, current_row_index, current_column_index, epsilon):
        if np.random.random() < epsilon:
            return np.argmax(self.q_values[current_row_index, current_column_index])
        else:
            return np.random.randint(len(self.actions))

    # ----------------------------------------------------------------------------------------------------------------

    # define a function that will get the next location based on the chosen action
    def get_next_location(self, current_row_index, current_column_index, action_index):

        new_row_index = current_row_index
        new_column_index = current_column_index

        if self.actions[action_index] == 'up' and current_row_index > 0:
            new_row_index -= 1

        elif self.actions[action_index] == 'right' and current_column_index < self.rewards.shape[1] - 1:
            new_column_index += 1

        elif self.actions[action_index] == 'down' and current_row_index < self.rewards.shape[0] - 1:
            new_row_index += 1

        elif self.actions[action_index] == 'left' and current_column_index > 0:
            new_column_index -= 1
        else:
            self.end = 1

        return new_row_index, new_column_index

    # ----------------------------------------------------------------------------------------------------------------

    def get_rewards(self, row_index, column_index):

        # From Model
        reward = self.rewards[row_index, column_index] * 15

        # To encourage to maintain the shortest path
        reward += -len(self.trajectory) * 25

        # To ensure that a horizontal well is drilled
        reward += abs(self.trajectory[-1][1] - self.trajectory[0][1]) * 10
        #                     print(reward)

        # To make sure max  amount of target pipes are used
        reward += -(self.available_pipe - len(self.trajectory)) * 5

        # Adding a -ve reward to encourage the agent to visit unique rows,columns
        rows = [i[0] for i in self.trajectory]
        columns = [i[1] for i in self.trajectory]

        reward += -(len(rows) - len(set(rows))) * 10
        reward += -(len(columns) - len(set(columns))) * 20

        #                     # Add a -ve reward to identify simultaneous right/left turns in the to avoid wellbore tornado effect
        #                     if (action_index == self.action_cache):
        #                         reward += -100

        return reward

    # ----------------------------------------------------------------------------------------------------------------

    # Define a function to train and populate the q table
    def populate_q_table(self, num_episodes, epsilon=0.1, discount_factor=0.9, learning_rate=0.9):
        print('Training Started!')
        for episode in range(num_episodes):
            self.reset()

            # get the starting location for this episode
            row_index, column_index = self.get_starting_location()
            self.trajectory.append([row_index, column_index])


            # continue taking actions (i.e., moving) until we reach a terminal state
            while not (self.is_terminal_state(row_index, column_index) | (self.end == 1)):

                # choose which action to take (i.e., where to move next)
                action_index = self.get_next_action(row_index, column_index, epsilon)

                # perform the chosen action, and transition to the next state (i.e., move to the next location)
                old_row_index, old_column_index = row_index, column_index  # store the old row and column indexes
                row_index, column_index = self.get_next_location(row_index, column_index, action_index)

                self.trajectory.append([row_index, column_index])
                reward = self.get_rewards(row_index, column_index)

                if (action_index == 1) | (action_index == 3):
                    self.action_cache = action_index

                old_q_value = self.q_values[old_row_index, old_column_index, action_index]

                temporal_difference = reward + (
                        discount_factor * np.max(self.q_values[row_index, column_index])) - old_q_value

                # update the Q-value for the previous state and action pair
                new_q_value = old_q_value + (learning_rate * temporal_difference)

                self.q_values[old_row_index, old_column_index, action_index] = new_q_value

            if (episode != 0) & ((episode + 1) % 100_000 == 0):
                print(f'    {"{:,}".format(episode + 1)} episodes completed')

        print('Training Complete!')

    # ----------------------------------------------------------------------------------------------------------------

    # Define a function that will get the shortest path
    def get_shortest_path(self, start_row_index, start_column_index):
        self.reset()
        current_row_index, current_column_index = start_row_index, start_column_index
        self.trajectory.append([current_row_index, current_column_index])

        pipes_used = 0

        while not (self.is_terminal_state(current_row_index, current_column_index) | (self.end == 1)):
            # get the best action to take
            action_index = self.get_next_action(current_row_index, current_column_index, 1.)

            # move to the next location on the path, and add the new location to the list
            current_row_index, current_column_index = self.get_next_location(current_row_index, current_column_index,
                                                                             action_index)
            pipes_used += 1

            if (pipes_used == self.available_pipe):
                self.end = 1
                print('Pipes Over')

            if ([current_row_index, current_column_index] in self.trajectory):
                self.end = 1
                print(f'Index in trajectory - [{current_row_index},{current_column_index}]')

            else:
                self.trajectory.append([current_row_index, current_column_index])

        return self.trajectory

    # ----------------------------------------------------------------------------------------------------------------

    # Define a function that will reset everything
    def reset(self):
        self.trajectory = []
        self.end = 0
        self.action_cache = []