import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym

windows = (210, 160)

board = (170, 160)

colors = {"pacman": (210, 164, 74), # orange
          "black":(0, 0, 0),
          "wall":(228, 111, 111),
          "background": (0, 28, 136)}


IMAGES = []
def adapt_obs(obs, plot=True):
    '''Adapting the observation from pacman'''

    ## Locate Pacman
    indices = np.where(np.all(obs == colors["pacman"], axis=-1))
    mean_x, mean_y = np.mean(indices[0]), np.mean(indices[1])
    ## Make circle
    xx, yy = np.mgrid[:windows[0], :windows[1]]
    circle = (xx-mean_x)**2+(yy-mean_y)**2
    mask = (circle < 400)
    mask = np.stack([mask]*3, axis=-1)
    new_obs = mask*obs
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
fig = plt.figure()
ani = animation.ArtistAnimation(fig, IMAGES, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()

