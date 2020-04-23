import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pygame
from collections import namedtuple
from itertools import count

class DiscretePlatformerEnv(gym.Env):
    """
    Description:
    ------------
        Discrete platformer environment for reinforcement learning with gym library
    
    Source:
    -------
        This environment corresponds to the discrete environement of the PlatformerEnv
    
    Observation:
    ------------
        Type: Box(2)
        Num     Observation             Min     Max
        0       Player Position         0       209
        1       Map End Position        0       200
    
    Actions:
    --------
        Type: Discrete(2)
        Num     Action
        0       Move player to the left
        1       Move player to the right
    
    Reward:
    -------
        Condition                   Reward
        Reach map end point         +10.0
        Move toward end point       +1.0
        Move away end point         -2.0
        Leave map                   -10.0
    
    Starting State (default):
    ---------------
        Player Position is randomly setted up to a value in [10:10:90]U[110:10:190]
        Map End Position is setted up to 100.
    
    Episode Termination:
    --------------------
        Player Position is not between [0..209]
        Solved Requirements
        Considered solved when the Player reaches the map end point
    """

    def __init__(self):

        # CONFIGURATIONS

        # Coefficient of Proportions (to increase physics proportionally)
        self.proportion = 1

        # Size of blocks
        self.block_width = 10*self.proportion
        self.block_height = 10*self.proportion

        # For the display
        self.visibility_x = 21  # width of viewer in amount of blocks. 
        self.visibility_y = 2   # height of viewer in amount of map height.

        # Initializes map
        self.initWorld()

        self.map_height = len(self.map) # in amount of blocks
        self.map_width  = len(self.map[0])

        self.size_x = self.block_width * self.visibility_x
        self.size_y = self.map_height * self.block_height * self.visibility_y

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

        # Block initialisation
        self.block = namedtuple('Block',('x','y','width','height','color'))
        self.block_list = []

        # Action space
        self.action_space = spaces.Discrete(2)

        # Observation space
        low = np.array([
            0,
            0
        ], dtype=np.float32)

        high = np.array([
            self.visibility_x*self.block_width-1,
            (self.visibility_x-1)*self.block_width
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None


    def initWorld(self):
        """ Initializes the map list
        """
        self.map = [
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

    def move(self, action):
        """ Updates the player position according to the given action
        """
        # Check whether the action is valid or not
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # Gets player position on x-axis
        player_position_x = self.state[0]
        # Updates player position on x-axis
        if action == 0:
            player_position_x += 10*self.proportion
        elif action == 1:
            player_position_x -= 10*self.proportion
        
        return(player_position_x)
    
    def success(self, player_position_x):
        """ Checks if the requirements are solved
        """
        return(True if player_position_x==self.state[1] else False)

    def setEnd(self, end_position_x):
        """ Sets end point position on x-axis
        """
        # Resets map
        self.initWorld()
        # Computes end point index in the map list
        end_index = int(end_position_x // 10)
        # Puts end point on map
        self.map[-1]=self.map[-1][:end_index]+'E'+self.map[-1][end_index+1:]

    def updateState(self, **kwargs):
        """ Updates environment state

        Parameters:
            **kwargs: {
                'player_x_pos': player position on x-axis,
                'end_x_pos': end point position on x-axis
            }
        """
        for key, value in kwargs.items():
            try:
                if key == 'player_x_pos':
                    self.state[0] = int(value)
                elif key == 'end_x_pos':
                    self.setEnd(value)
                    self.state[2] = int(value)
                else:
                    raise Exception("Invalid keyword argument: %s."%(key))
            
            except:
                raise Exception("Invalid value type: {} ({}), only integer \
                and float numbers are accepted.".format(value, type(value)))

    def step(self, action, time=0, time_max=50):
        """ Updates the environment state

        Parameters:
            action: action chosen by the player.
            time: actual duration of the episode.
            time_max: maximal duration of the episode.
        """
        # Updates player_position_x
        player_position_x = self.move(action)
        # Cheks whether the episode is terminated or not
        done = not self.observation_space.contains([player_position_x, self.state[1]]) \
               or self.success(player_position_x) \
               or time >= time_max
        
        # Sets the reward for the transition
        if done:
            if self.success(player_position_x):
                reward = 10.0
            elif not self.observation_space.contains([player_position_x, self.state[1]]):
                reward = -10.0
            else:
                reward = 0.0
        elif abs(self.state[0]-self.state[1])>abs(player_position_x-self.state[1]):
            reward = 1.0
        else:
            reward = -2.0
        
        # Updates the environment state
        self.state[0] = player_position_x

        return(self.state, reward, done, {})
    
    def reset(self, player_position_x=None, end_position_x=None):
        """ Resets the state of the environment
        """
        # Resets map
        self.initWorld()
        # Sets player position randomly if not provided
        if not player_position_x:
            tmp = int(round(80*np.random.random_sample()+10, -1)) # [10:10:90]
            tmp = tmp if np.random.random_sample() > 0.5 else -tmp # [-90:10:-10]U[10:10:90] 
            player_position_x = tmp + 100 # [10:10:90]U[110:10:190]
        # Sets end point position if not provided
        if not end_position_x:
            end_position_x = int(100) # Sets the end point at 100
        
        self.setEnd(end_position_x)
        # Updates environment state
        self.state = np.array([player_position_x, end_position_x])
        # Store the first position of the player
        self.start_x = player_position_x
        self.start_y = self.size_y-self.block_height-self.player_height
        return(self.state)
    
    def levelGeneration(self):
        """ Generates the level block list.

        Code:
            End point is purple, otherwise blocks are white.
        """

        x, y = 0, (self.visibility_y-1)*self.map_height*self.block_height

        for column in range(self.map_width):
            for row in range(self.map_height):
                if self.map[row][column] == "W":
                    self.block_list.append(self.block(x,y,self.block_width, self.block_height,self.white))
                elif self.map[row][column]=="E":
                    self.block_list.append(self.block(x,y,self.block_width, self.block_height,self.purple))
                
                y += self.block_height

            x += self.block_width
            y = (self.visibility_y-1)*self.map_height*self.block_height

    def render(self, mode='human'):
        """ Generates the environment graphical view.

        Parameters:
            mode:   'human' displays a pygame window of the environment.
                    'rgb_array' returns a 3d array of the environment (HWC)
        """
        # Creates the window
        self.viewer = pygame.Surface((self.size_x, self.size_y)) # Dimensions of WINDOW
        # Updates self.block_list
        self.levelGeneration()
        # Draws the background
        self.viewer.fill(self.grey)
        # Draws each block
        for block in self.block_list:
            pygame.draw.rect(self.viewer, block.color, pygame.Rect(block.x, block.y, block.width, block.height))
        # Draws the player
        player = self.block(self.state[0], self.start_y, self.player_width, self.player_height, self.blue)
        pygame.draw.rect(self.viewer, player.color, pygame.Rect(player.x, player.y, player.width, player.height))
        
        if mode=='human':
            screen = pygame.display.set_mode((self.size_x, self.size_y))
            screen.blit(self.viewer, (0,0))
            # Refreshes the window
            pygame.display.update()
            return(None)

        elif mode=='rgb_array':
            return(pygame.surfarray.array3d(self.viewer).swapaxes(0,1))

        else:
            raise Exception('Invalid mode: ', mode)

    def close(self):
        """ Closes pygame window
        """
        pygame.quit()


if __name__=="__main__":
    
    # EXAMPLE
    env = DiscretePlatformerEnv()
    clock = pygame.time.Clock()

    print('Starting 10 episodes...')
    for i in range(10):
        env.reset()
        print('episode %d'%(i))
        for t in count():
            # sets number of frame per second
            clock.tick(15)
            # uniform probability to go left or right
            action = 0 if np.random.random_sample()<0.5 else 1
            # updates the environment according to the chosen action
            state, reward, done, _ = env.step(action, time=t)
            # displays environment
            env.render()
            # get env picture
            env_pic = env.render(mode='rgb_array')
            # break if episode terminated
            if done:
                break