import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pygame
from collections import namedtuple

class PlatformerEnv(gym.Env):
    """
    Description:
        Platformer environment for reinforcement learning with gym library
    
    Source:
        This environment corresponds to the simple-platformer designed by Maxence Blanc (https://github.com/maxenceblanc/simple-platformer)

    Observation:
        Type: Box(3)
        Num     Observation                 Min         Max
        0       Player Position             0           720
        1       Player Velocity             -Inf        Inf
        2       Chunk End Position          0           720
    
    Actions:
        Type: Discrete(2)
        Num     Action
        0       Move player to the left
        1       Move player to the right

    Reward:
        Reward is 1 for every step in the direction of the arrival and -1 otherwise.

    Starting State:
        Player Position is randomly setted up to a value in [180..540]
        Chunk End Position is randomly setted up to a value in [32..712]
        Player Velocity is assigned to 0
    
    Episode Termination:
        Player Position is not between [-10..10]
        Episode length is greater than 100
        Solved Requirements
        Considered solved when the Player reaches the Chunk End.
    """

    def __init__(self):

        # CONFIGURATIONS

        # Coefficient of Proportions (to increase physics proportionally)
        self.proportion = 1

        # Size of blocks
        self.block_width = 16*self.proportion
        self.block_height = 16*self.proportion

        # For the display
        self.visibility_x = 45 # Width of WINDOW in amount of blocks.
        self.visibility_y = 2 # Height of WINDOW in amount of chunk height.

        # Chunk
        self.initWorld()

        self.chunk_height = len(self.chunk) # in amount of blocks
        self.chunk_width  = len(self.chunk[0])

        self.size_x = self.block_width * self.visibility_x # Width of WINDOW in pixels.
        self.size_y = self.chunk_height * self.block_height * self.visibility_y # Height of WINDOW in pixels.

        # Player size
        self.player_width = 1*self.block_width
        self.player_height = 2*self.block_height

        # Start
        self.start_x = None
        self.start_y = None

        # Colors
        self.white  = (255,255,255)
        self.grey   = (30,30,30)
        self.orange = (255,125,0)
        self.red    = (255,0,0)
        self.green  = (0,255,0)
        self.purple = (153,0,204)
        self.blue   = (57,155,216)

        # Acceleration
        self.acceleration_x = 1*self.proportion
        self.acceleration_y = 1.67*self.proportion #1.67
        self.coeff_acceleration_x = 1.3
        self.slowdown_x = self.coeff_acceleration_x * self.acceleration_x

        # Block initialisation
        self.block = namedtuple('Block',('x','y','width','height','color'))
        self.block_list = []

        # Speed
        self.speed_x = 16 * self.proportion # speed_x max
        self.speed_y = self.block_height * (7/8) # speed_y max

        # Action space
        self.action_space = spaces.Discrete(2)

        # Observation space
        low = np.array([
            0,
            np.finfo(np.float32).min,
            0
        ], dtype=np.float32)

        high = np.array([
            720,
            np.finfo(np.float32).max,
            720
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        # FIXME: useless ?
        # self.steps_beyond_done = None
    
    def initWorld(self):
        # Chunk
        self.chunk = [
            " "*self.visibility_x,
            " "*self.visibility_x,
            " "*self.visibility_x,
            " "*self.visibility_x,
            " "*self.visibility_x,
            " "*self.visibility_x,
            " "*self.visibility_x,
            " "*self.visibility_x,
            " "*self.visibility_x,
            "W"*self.visibility_x
        ]
    
    def seed(self, seed=None):
        """ TODO
        """
        self.np_random, seed = seeding.np_random(seed)
        return([seed])
    
    def slowdown(self, speed_x):
        """ TODO
        """
        if 1 > speed_x/self.slowdown_x>-1:
            return(0)
        else:
            return(int(speed_x/self.slowdown_x))

    def move(self, action):
        """ TODO
        """
        player_position_x = self.state[0]
        player_speed_x = self.state[1]

        if action == 0:
            if player_speed_x > 0:
                # SLOWDOWN
                player_speed_x = self.slowdown(player_speed_x)
            else:
                player_speed_x -= self.acceleration_x
        else:
            if player_speed_x < 0:
                # SLOWDOWN
                player_speed_x = self.slowdown(player_speed_x)
            else:
                player_speed_x += self.acceleration_x
        
        player_position_x += player_speed_x

        return(player_position_x, player_speed_x)
    
    def success(self, position_x):
        if self.start_x<=self.state[2]:
            return(True if position_x >= self.state[2]-8 else False)
        else:
            return(True if position_x <= self.state[2]+8 else False)

    def step(self, action):
        """ Updates the environment state
        """
        # Check whether the action is valid or not
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # Updates player_position_x, player_speed_x
        player_position_x, player_speed_x = self.move(action)
        # Episode completed if observation values out of bounds or end point reached
        done =  not self.observation_space.contains([player_position_x, player_speed_x, self.state[2]]) \
                or self.success(player_position_x)
        done = bool(done)
        # Sets the reward for the given action
        if self.success(player_position_x):
            reward = 10.0
        elif abs(self.state[0]-self.state[2])>abs(player_position_x-self.state[2]):
            reward = 1.0
        elif abs(self.state[0]-self.state[2])==abs(player_position_x-self.state[2]):
            reward = 0.0
        else:
            reward = -1
        
        # Updates state
        self.state[0], self.state[1] = player_position_x, player_speed_x

        return(self.state, reward, done, {})

    def reset(self):
        """ Resets the state of the environment randomly
        """
        self.initWorld()
        # Sets player position
        player_position_x = (self.size_x/2)*(np.random.random_sample()+(1/2)) # random value between [180, 540[
        player_speed_x = 0
        # Sets end point position
        end_position_x = (self.size_x-40)*np.random.random_sample() + 32 # random value between [20, 700[
        end_index = int(end_position_x // 16)-1
        # Puts end point on map
        self.chunk[-1]=self.chunk[-1][:end_index]+'E'+self.chunk[-1][end_index+1:]
        # Updates state problem 
        self.state = np.array([player_position_x, player_speed_x, end_position_x])
        # Store the first position of the player
        self.start_x = player_position_x
        self.start_y = self.size_y - self.block_height - self.player_height
        return(self.state)

    def levelGeneration(self):
        """ Generates the level block list.

        Code:
            End point is purple, otherwise blocks are white.
        """

        x, y = 0, (self.visibility_y-1)*self.chunk_height*self.block_height

        for column in range(self.chunk_width):
            for row in range(self.chunk_height):
                if self.chunk[row][column] == "W":
                    self.block_list.append(self.block(x,y,self.block_width, self.block_height,self.white))
                elif self.chunk[row][column]=="E":
                    self.block_list.append(self.block(x,y,self.block_width, self.block_height,self.purple))
                
                y += self.block_height

            x += self.block_width
            y = (self.visibility_y-1)*self.chunk_height*self.block_height

    def render(self):
        """ Displays a graphical view of the environment
        """
        # Creates the window
        self.viewer = pygame.display.set_mode((self.size_x, self.size_y)) # Dimensions of WINDOW
        # Updates self.block_list
        self.levelGeneration()
        # Draws the background
        self.viewer.fill(self.grey)
        # Draws each block
        for block in self.block_list:
            pygame.draw.rect(self.viewer, block.color, pygame.Rect(block.x, block.y, block.width, block.height))
        # Draws the player
        player = self.block(self.state[0]-int(self.block_width/2), self.start_y, self.player_width, self.player_height, self.blue)
        pygame.draw.rect(self.viewer, player.color, pygame.Rect(player.x, player.y, player.width, player.height))
        # Refreshes the window
        pygame.display.update()

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # TEST

    # environement init
    env = PlatformerEnv()
    # clock init
    clock = pygame.time.Clock()
    # loop
    for i in range(50):
        env.reset()
        for t in range(100):
            # sets number of fps
            clock.tick(20)
            # uniform probability to go left or right
            action = 0 if np.random.random_sample()<0.5 else 1
            # updates the environment according to the chosen action
            _, reward, done, _ = env.step(action)
            # displays environment
            env.render()
            # break is episode completed
            if done:
                break