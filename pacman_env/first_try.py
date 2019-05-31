import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym

windows = (210, 160)

board = (170, 160)
mean_x_board, mean_y_board = board[0]//2, board[1]//2

colors = {"pacman": (210, 164, 74), # orange
          "black":(0, 0, 0),
          "wall":(228, 111, 111),
          "background": (0, 28, 136)}

plot = True
recentering = True
radius = 20


IMAGES = []


def adapt_obs(obs):
    """
    Adapting the observation from pacman

    """

    # Locate Pac-man
    indices = np.where(np.all(obs == colors["pacman"], axis=-1))
    mean_x, mean_y = np.mean(indices[0]), np.mean(indices[1])

    # Make circle
    xx, yy = np.mgrid[:windows[0], :windows[1]]
    circle = (xx-mean_x)**2+(yy-mean_y)**2
    mask = (circle < radius**2)
    mask = np.stack([mask]*3, axis=-1)
    new_obs = mask*obs

    if recentering:
        #new_board = np.zeros_like(obs)
        new_board = np.zeros((2*radius, 2*radius, 3), dtype=obs.dtype)
        max_x = min(int(mean_x + radius), windows[0])
        min_x = max(int(mean_x - radius), 0)
        lx = (max_x - min_x) // 2
        rx = (max_x - min_x) - lx
        max_y = min(int(mean_y + radius), windows[1])
        min_y = max(int(mean_y - radius), 0)
        ly = (max_y - min_y) // 2
        ry = (max_y - min_y) - ly
        #new_board[mean_x_board-lx:mean_x_board+rx,
        #          mean_y_board-ly:mean_y_board+ry] \
        #    = new_obs[min_x:max_x,
        #              min_y:max_y]
        print(np.min(new_obs), np.max(new_obs), type(new_obs))
        new_board[radius-lx:radius+rx,
                  radius-ly:radius+ry] \
            = new_obs[min_x:max_x,
                      min_y:max_y]
        new_obs = new_board
        print(np.min(new_obs), np.max(new_obs), type(new_obs))
    if plot:
        im = plt.imshow(new_obs, animated=True)
        IMAGES.append([im])
    return new_obs


env = gym.make('MsPacman-v0')
env.reset()
for _ in range(1000):
    #env.render()
    obs, _, _, _ = env.step(env.action_space.sample()) # take a random action
    obs = adapt_obs(obs)
env.close()
print("Simulation over")
if plot:
    fig = plt.figure()
    ani = animation.ArtistAnimation(fig, IMAGES, interval=50, blit=True,
                                repeat_delay=1000)
    plt.show()

