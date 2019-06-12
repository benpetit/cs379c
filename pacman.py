import numpy as np
import gym

class POPacman:
    colors = {"pacman": (210, 164, 74), # orange
              "black":(0, 0, 0),
              "wall":(228, 111, 111),
              "background": (0, 28, 136)}

    def __init__(self, radius = 60, recentering = False):
        self._radius = radius
        self._recentering = recentering
        self._env = gym.make('MsPacman-v0')
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.reward_range = self._env.reward_range
        self.metadata = self._env.metadata
        self.unwrapped = self._env.unwrapped

    def reset(self):
        return self._env.reset()

    def seed(self, s):
        return self._env.seed(s)

    def render(self):
        self._env.render()

    def render_partial(self, obs):
        im = plt.imshow(obs, animated=True)

    def step(self, action):
        s, r, d, i = self._env.step(action)
        s = self._adapt(s)
        return s, r, d, i

    def _adapt(self, obs):
        radius = self._radius
        windows = (210, 160)
        board = (170, 160)

        indices = np.where(np.all(obs == self.colors["pacman"], axis=-1))
        mean_x, mean_y = np.mean(indices[0]), np.mean(indices[1])

        # Make circle
        xx, yy = np.mgrid[:windows[0], :windows[1]]
        circle = (xx-mean_x)**2+(yy-mean_y)**2
        mask = (circle < radius**2)
        mask = np.stack([mask]*3, axis=-1)
        new_obs = mask*obs

        if self._recentering:
            new_board = np.zeros((2*radius, 2*radius, 3), dtype=obs.dtype)
            max_x = min(int(mean_x + radius), windows[0])
            min_x = max(int(mean_x - radius), 0)
            lx = (max_x - min_x) // 2
            rx = (max_x - min_x) - lx
            max_y = min(int(mean_y + radius), windows[1])
            min_y = max(int(mean_y - radius), 0)
            ly = (max_y - min_y) // 2
            ry = (max_y - min_y) - ly
            new_board[radius-lx:radius+rx,
                      radius-ly:radius+ry] \
                = new_obs[min_x:max_x,
                          min_y:max_y]
            new_obs = new_board

        return new_obs
