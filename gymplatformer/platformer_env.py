import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pygame
from collections import namedtuple
from itertools import count

class PlatformerEnv(gym.Env):
    """
    Description:
    ------------
        Platformer environment for reinforcement learning with gym library
    
    Source:
    -------
        This environment corresponds to the simple-platformer designed by Maxence Blanc (https://github.com/maxenceblanc/simple-platformer)

    Observation:
    ------------
        Type: Box(3)
        Num     Observation                 Min         Max
        0       Player Position             0           209
        1       Player Velocity             -10         10
        2       Chunk End Position          0           200
    
    Actions:
    --------
        Type: Discrete(2)
        Num     Action
        0       Move player to the right
        1       Move player to the left

    Reward:
    -------
        Condition                   Reward
        Reach map end point         +100.0
        Move toward end point       +1.0
        Move away end point         -2.0
        Leave map                   -10.0

    Starting State (default):
    -------------------------
        Player Position is randomly setted up to a value in [10..90]U[110..190]
        Chunk End Position is setted up to 100
        Player Velocity is assigned to 0
    
    Episode Termination:
    --------------------
        Player Position is not between [0..209]
        Solved Requirements
        Considered solved when the Player reaches the Chunk End.
    """

    def __init__(self):
        """ Inits environment object
        """

        # CONFIGURATIONS

        # Coefficient of Proportions (to increase physics proportionally)
        self.proportion = 1

        # Size of blocks
        self.block_width = 10*self.proportion
        self.block_height = 10*self.proportion

        # For the display
        self.visibility_x = 21 # Width of WINDOW in amount of blocks.
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

        # Speed boundaries
        self.speed_x = self.block_width # maximal speed on x-axis
        self.speed_y = self.block_height * (7/8) # maximal speed on y-axis

        # Action space
        self.action_space = spaces.Discrete(2)

        # Observation space
        low = np.array([
            0,
            -self.speed_x,
            0
        ], dtype=np.float32)

        high = np.array([
            self.visibility_x*self.block_width-1,
            self.speed_x,
            (self.visibility_x-1)*self.block_width
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
    
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
        """ Updates the player position according to the given action
        """
        # Check whether the action is valid or not
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # Gets player position on x-axis
        player_position_x = self.state[0]
        # Gets player speed on x-axis
        player_speed_x = self.state[1]
        # Updates player speed on x-axis
        if action == 0:
            if player_speed_x < 0:
                # SLOWDOWN
                player_speed_x = self.slowdown(player_speed_x)
            else:
                player_speed_x += self.acceleration_x if abs(player_speed_x)<=self.speed_x else 0

        elif action == 1:
            if player_speed_x > 0:
                # SLOWDOWN
                player_speed_x = self.slowdown(player_speed_x)
            else:
                player_speed_x -= self.acceleration_x if abs(player_speed_x)<=self.speed_x else 0

        # Updates player position on-x-axis
        player_position_x += player_speed_x

        return(player_position_x, player_speed_x)
    
    def success(self, player_position_x):
        """ Checks if the requirements are solved
        """
        return(True if player_position_x==self.state[1] else False)
    
    def setEnd(self, end_position_x):
        """ Sets end point position on x-axis
        """
        # Resets map
        self.initWorld()
        # Computes end point index in the chunk list
        end_index = int(end_position_x // 10)
        # Puts end point on map
        self.chunk[-1]=self.chunk[-1][:end_index]+'E'+self.chunk[-1][end_index+1:]

    def updateState(self, **kwargs):
        """ Updates environment state

        Parameters:
            **kwargs: {
                'player_x_pos': player position on x-axis,
                'player_x_spe': player speed on x-axis,
                'end_x_pos': end point position on x-axis
            }
        """
        for key, value in kwargs.items():
            try:
                if key == 'player_x_pos':
                    self.state[0] = int(value)
                elif key == 'player_x_spe':
                    self.state[1] = value
                elif key == 'end_x_pos':
                    self.setEnd(value)
                    self.state[2] = int(value)
                else:
                    raise Exception("Invalid keyword argument: %s."%(key))
            
            except:
                raise Exception("Invalid value type: {} ({}), only integer \
                and float numbers are accepted.".format(value, type(value)))

    def step(self, action, time, time_max=50):
        """ Updates the environment state

        Parameters:
            action: action chosen by the player.
            time: actual duration of the episode.
            time_max: maximal duration of the episode.
        """
        # Updates player_position_x, player_speed_x
        player_position_x, player_speed_x = self.move(action)
        # Cheks whether the episode is terminated or not
        done =  not self.observation_space.contains([player_position_x, player_speed_x, self.state[2]]) \
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
        # Updates state
        self.state[0], self.state[1] = player_position_x, player_speed_x

        return(self.state, reward, done, {})

    def reset(self, player_position_x=None, player_speed_x=None, end_position_x=None):
        """ Resets the state of the environment
        """
        # Resets map
        self.initWorld()
        # Sets player position randomly if not provided
        if not player_position_x:
            tmp = int(round(80*np.random.random_sample()+10, -1)) # [10:10:90]
            tmp = tmp if np.random.random_sample() > 0.5 else -tmp # [-90:10:-10]U[10:10:90] 
            player_position_x = tmp + 100 # [10:10:90]U[110:10:190]
        
        # Sets player speed if not provided
        if not player_speed_x:
            player_speed_x = int(0)

        # Sets end point position if not provided
        if not end_position_x:
            end_position_x = int(100) # Sets the end point at 100
        
        self.setEnd(end_position_x)
        # Updates environment state
        self.state = np.array([player_position_x, player_speed_x, end_position_x])
        # Store the first position of the player
        self.start_x = player_position_x
        self.start_y = self.size_y-self.block_height-self.player_height
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


if __name__ == "__main__":
    
    # EXAMPLE
    env = PlatformerEnv()
    clock = pygame.time.Clock()
    
    print('Starting 10 episodes...')
    for i in range(10):
        env.reset()
        print('episode %d'%(i))
        for t in count():
            # sets number of fps
            clock.tick(15)
            # uniform probability to go left or right
            action = 0 if np.random.random_sample()<0.5 else 1
            # updates the environment according to the chosen action
            state, reward, done, _ = env.step(action, time=t)
            # displays environment
            env.render()
            # get env picture
            env_pic = env.render(mode='rgb_array')
            # break if episode completed
            if done:
                break