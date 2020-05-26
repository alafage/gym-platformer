import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pygame
import random
from collections import namedtuple
from itertools import count

class PlatformerEnv(gym.Env):
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
        Num     Observation                     Min     Max
        0       Player Horizontal Position      0       209
        1       Player Vertical Position        0       159
        2       Player Horizontal Speed         -10     10
        3       Player Vertical Speed           -30     30
        4       Target Horizontal Position      0       200
        5       Target Vertical Position        0       150
    
    Actions:
    --------
        Type: Discrete(2)
        Num     Action
        0       Move player to the right
        1       Move player to the left
        2       Move player to the top
        3       Do nothing
    
    Reward:
    -------
        Condition                       Reward
        Reach map target                +10.0
        Move toward target              +1.0
        Not moving toward target        -2.0
        Leave map                       -10.0
    
    Starting State (default):
    ---------------
        Player Position is randomly setted up to a value in [10:10:90]U[110:10:190]
        Map Target Position is setted up to 100.
    
    Episode Termination:
    --------------------
        Player Position is not between [0..209]
        Solved Requirements
        Considered solved when the Player reaches the map target
    """

    def __init__(self):

        # CONFIGURATIONS

        # Coefficient of Proportions (to increase physics proportionally)
        self.proportion = 1

        # Size of blocks
        self.block_width = 10*self.proportion
        self.block_height = 10*self.proportion

        # For the display
        self.visibility_x = 42  # width of viewer in amount of blocks. 
        self.visibility_y = 1   # height of viewer in amount of map height.

        # Initializes map
        self.initWorld()

        self.map_height = len(self.map) # in amount of blocks
        self.map_width  = len(self.map[0])

        self.size_x = self.block_width * self.visibility_x
        self.size_y = self.map_height * self.block_height * self.visibility_y

        # Player size
        self.player_width = 1*self.block_width
        self.player_height = 2*self.block_height

        # Target size
        self.target_width = 0.5*self.block_width
        self.target_height = 0.5*self.block_height

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
        self.block = namedtuple('Block',('rect','color'))
        self.block_list = []

        # Player initialization
        self.player = None
        # Target initialization
        self.target = None

        # Action space
        self.action_space = spaces.Discrete(4)

        # Observation space
        low = np.array([
            0,
            0,
            -10,
            -10,
            0,
            0
        ], dtype=np.float32)

        high = np.array([
            self.size_x-1,
            self.size_y-1,
            10,
            30,
            self.size_x-1,
            self.size_y-1
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Phisics
        self.x_accel = 1*self.proportion
        self.x_coeff_accel = 1.4
        self.x_slowdown = self.x_coeff_accel*self.x_accel
        self.x_max_speed = 10*self.proportion
        self.y_max_speed = self.block_height * (7/8)
        self.gravity = 1.67*self.proportion

        self.seed()
        self.viewer = None
        self.state = None
        self.completed = None
    
    #### GYM ENVIRONMENT FUNCTIONS

    def seed(self, seed=None):
        """ TODO
        """
        self.np_random, seed = seeding.np_random(seed)
        return([seed])

    def step(self, action, time=0, time_max=50):
        """ Updates the environment state

        Parameters:
            action: action chosen by the player.
            time: actual duration of the episode.
            time_max: maximal duration of the episode.
        """
        if not self.completed:
            # Updates the environment state
            x_speed, y_speed, reached_targets = self.movePlayer(action)
            # Cheks whether the episode is terminated or not
            done = not self.observation_space.contains([self.player.rect.centerx, self.player.rect.centery, x_speed, y_speed, self.target.rect.centerx, self.target.rect.centery]) \
                or self.success() \
                or time >= time_max
            # Sets the reward for the transition
            if done:
                if self.success():
                    reward = 10.0
                elif not self.observation_space.contains([self.player.rect.centerx, self.player.rect.centery, x_speed, y_speed, self.target.rect.centerx, self.target.rect.centery]):
                    reward = -10.0
                else:
                    reward = 0.0
            elif abs(self.state[0]-self.state[4])>abs(self.player.rect.centerx-self.target.rect.centerx):
                reward = 1.0
            else:
                reward = -2.0
            
            # Updates the environment state
            self.state[0] = self.player.rect.centerx
            self.state[1] = self.player.rect.centery
            self.state[2] = x_speed
            self.state[3] = y_speed
            self.state[4] = self.target.rect.centerx
            self.state[5] = self.target.rect.centery

            if done:
                if self.success():
                    self.target = None
                self.completed=True

            return(self.state, reward, done, {})
        else:
            return(None, None, None, None)

    def reset(self, player_position=None, end_position=None):
        """ Resets the state of the environment
        """
        # Resets map
        self.initWorld()
        # Sets player position randomly if not provided
        if not player_position:
            # tmp = int(round(80*np.random.random_sample()+10, -1)) # [10:10:90]
            # tmp = tmp if np.random.random_sample() > 0.5 else -tmp # [-90:10:-10]U[10:10:90] 
            # player_position_x = tmp + 100 # [10:10:90]U[110:10:190]
            self.start_x, self.start_y = self.setPlayer()
        # Sets end point position if not provided
        if not end_position:
            # end_position_x = int(100) # Sets the end point at 100
            x_target, y_target = self.setTarget()
        # Updates player
        self.player = self.block(pygame.Rect(self.start_x, self.start_y, self.player_width, self.player_height), self.blue)
        # Updates target
        self.target = self.block(pygame.Rect(x_target, y_target, self.target_width, self.target_height), self.red)
        # Updates environment state
        self.state = np.array([self.player.rect.centerx, self.player.rect.centery, 0, 0, self.target.rect.centerx, self.target.rect.centery])
        # Updates self.block_list
        self.levelGeneration()
        # Updates environement completion status
        self.completed = False
        return(self.state)

    def render(self, mode='human'):
        """ Generates the environment graphical view.

        Parameters:
            mode:   'human' displays a pygame window of the environment.
                    'rgb_array' returns a 3d array of the environment (HWC)
        """
        # Creates the window
        self.viewer = pygame.Surface((self.size_x, self.size_y)) # Dimensions of WINDOW
        # Draws the background
        self.viewer.fill(self.grey)
        # Draws each block
        for block in self.block_list:
            pygame.draw.rect(self.viewer, block.color, block.rect)
        # Draws the player
        pygame.draw.rect(self.viewer, self.player.color, self.player.rect)
        # Draws the target
        if self.target:
            pygame.draw.rect(self.viewer, self.target.color, self.target.rect)
        
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

    #### CUSTOM FUNCTIONS

    def initWorld(self):
        """ Initializes the map list

        E: potential target
        I: potential init for player
        W: wall
        _: empty
        """
        # self.map = [
        #     " "*self.visibility_x,
        #     " "*self.visibility_x,
        #     " "*self.visibility_x,
        #     " "*self.visibility_x,
        #     " "*self.visibility_x,
        #     " "*self.visibility_x,
        #     " "*self.visibility_x,
        #     " "*self.visibility_x,
        #     " "*self.visibility_x,
        #     "W"*self.visibility_x
        # ]
        # Map without obstacle
        self.map = [
            "__________________________________________",
            "__________________________________________",
            "__________________________________________",
            "__________________________________________",
            "__________________________________________",
            "__________________________________________",
            "__________________________________________",
            "IIIIWWWWEWWWEWEWWEWWEWWEWEWWEWWEWWWWWWIIII"
        ]
        # Map with obstacle
        self.map = [
            "__________________________________________",
            "__________________________________________",
            "__________________________________________",
            "__________________________________________",
            "__________________________________________",
            "__________________________________________",
            "___________________W______________________",
            "IIIIEWWWEWWWEWWWWWWWWWWEWEWWWWWEWWWWWEIIII"
        ]

    def onGround(self):
        """ Checks whether the player is on some ground or not
        """
        for block in self.block_list:
            for pixel in range(-(self.block_width-1), self.block_width):
                if self.player.rect.bottom == block.rect.top and self.player.rect.left == block.rect.left + pixel:
                    return(True)
        
        return(False)

    def slowdown(self, x_speed):
        """ Slows the player down
        """
        if 1>x_speed/self.x_slowdown>-1:
            return(0)
        else:
            return(int(x_speed/self.x_slowdown))

    def horizontalCollisions(self, x_speed):
        """ Handling of horizontal collisions when moving the player.
        """
        reached_targets = []
        for i in range(len(self.block_list)):
            if self.player.rect.colliderect(self.block_list[i].rect):
                if self.block_list[i].color==self.white:
                    if x_speed>0:
                        self.player.rect.right=self.block_list[i].rect.left
                        x_speed = self.slowdown(x_speed)
                    elif x_speed<0:
                        self.player.rect.left=self.block_list[i].rect.right
                        x_speed = self.slowdown(x_speed)
                
                elif self.block_list[i].color==self.red:
                    reached_targets.append(i)

        return(x_speed, reached_targets)
    
    def verticalCollisions(self, y_speed):
        """ Handling of vertical collisions when moving the player.
        """
        reached_targets = []
        for i in range(len(self.block_list)):
            if self.player.rect.colliderect(self.block_list[i].rect):
                if self.block_list[i].color==self.white:
                    if y_speed<0:
                        self.player.rect.top=self.block_list[i].rect.bottom
                        y_speed = 0
                    elif y_speed>0:
                        self.player.rect.bottom=self.block_list[i].rect.top
                        y_speed = 0
                
                elif self.block_list[i].color==self.red:
                    reached_targets.append(i)

        return(y_speed, reached_targets)

    def movePlayer(self, action):
        """ Updates the player position according to the given action
        """
        # Checks whether the action is valid or not
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # Gets actual player's speeds
        x_new_speed = self.state[2]
        y_new_speed = self.state[3]
        # Gets end point position

        if action == 0:
            x_new_speed += self.x_accel
        elif action == 1:
            x_new_speed -= self.x_accel
        else:
            x_new_speed = self.slowdown(x_new_speed)
        
        if not self.onGround():
            y_new_speed += self.gravity
        else:
            if action == 2:
                y_new_speed -= self.y_max_speed
        
        # Reduces speed if exceeding the maximum speed
        x_new_speed = x_new_speed if abs(x_new_speed)<self.x_max_speed else (x_new_speed/abs(x_new_speed))*self.x_max_speed
        y_new_speed = y_new_speed if abs(y_new_speed)<self.y_max_speed else (y_new_speed/abs(y_new_speed))*self.y_max_speed

        # Updates player's horizontal position
        self.player.rect.x += x_new_speed
        # Handles collisions
        x_new_speed, tmp1 = self.horizontalCollisions(x_new_speed)
        # Updates player's vertical position
        self.player.rect.y += y_new_speed
        # Handles collisions
        y_new_speed, tmp2 = self.verticalCollisions(y_new_speed)
        # Union without repetition
        reached_targets = list(set(tmp1)|set(tmp2))

        return(x_new_speed, y_new_speed, reached_targets)

    def success(self):
        """ Checks if the requirements are solved

        DEPRECATED
        """
        return(self.player.rect.colliderect(self.target.rect))

    def setPlayer(self):
        """ Returns player position on x and y axis
        """
        possible_inits = []

        for n_stage in range(self.map_height):
            for n_cell in range(self.map_width):
                if self.map[n_stage][n_cell] == 'I':
                    possible_inits.append({
                        'n_stage': n_stage-1,
                        'n_cell':  n_cell
                    })
        
        player_init = random.sample(possible_inits, 1)[0]

        x_player = player_init['n_cell']*self.block_width*self.proportion
        y_player = player_init['n_stage']*self.block_height*self.proportion

        return(x_player, y_player)

    def setTarget(self):
        """ Returns target position on x and y axis
        """
        possible_targets = []

        for n_stage in range(self.map_height):
            for n_cell in range(self.map_width):
                if self.map[n_stage][n_cell] == 'E':
                    possible_targets.append({
                        'n_stage': n_stage-1,
                        'n_cell':  n_cell
                    })

        target = random.sample(possible_targets, 1)[0]

        x_target = (target['n_cell']*self.block_width+(self.block_width/4))*self.proportion
        y_target = (target['n_stage']*self.block_height+(self.block_height/4))*self.proportion

        return(x_target, y_target)

    def updateState(self, **kwargs):
        """ Updates environment state FIXME: usefull ?

        Parameters:
            **kwargs: {
                'player_x_pos': player position on x-axis,
                'end_x_pos': end point position on x-axis
            }
        """
        for key, value in kwargs.items():
            try:
                if key == 'player_x_pos':
                    # TODO: checking value
                    self.state[0] = int(value)
                elif key == 'end_x_pos':
                    # TODO: checking value
                    self.setEnd(value)
                    self.state[1] = int(value)
                else:
                    raise Exception("Invalid keyword argument: %s."%(key))
            
            except:
                raise Exception("Invalid value type: {} ({}), only integer \
                and float numbers are accepted.".format(value, type(value)))
    
    def levelGeneration(self):
        """ Generates the level block list.

        Code:
            End point is purple, otherwise blocks are white.
        """

        x, y = 0, (self.visibility_y-1)*self.map_height*self.block_height

        for column in range(self.map_width):
            for row in range(self.map_height):
                if self.map[row][column] in "WEI":
                    self.block_list.append(self.block(pygame.Rect(x,y,self.block_width, self.block_height),self.white))
                
                y += self.block_height

            x += self.block_width
            y = (self.visibility_y-1)*self.map_height*self.block_height


if __name__=="__main__":
    
    # EXAMPLE
    env = PlatformerEnv()    
    clock = pygame.time.Clock()

    env.reset()

    while True:
        clock.tick(15)
        env.render()

        key = pygame.key.get_pressed()

        if key[pygame.K_q]:
            state, reward, done, _ = env.step(1)
        elif key[pygame.K_d]:
            state, reward, done, _ = env.step(0)
        elif key[pygame.K_z]:
            state, reward, done, _ = env.step(2)
        elif key[pygame.K_ESCAPE]:
            break
        else:
            state, reward, done, _ = env.step(3)

    # env.close()
    # print('Starting 10 episodes...')
    # for i in range(10):
    #     env.reset()
    #     print('episode %d'%(i))
    #     for t in count():
    #         # sets number of frame per second
    #         clock.tick(15)
    #         # uniform probability to go left or right
    #         # action = 0 if np.random.random_sample()<0.5 else 1
    #         action = 2
    #         # updates the environment according to the chosen action
    #         state, reward, done, _ = env.step(action, time=t)
    #         # displays environment
    #         env.render()
    #         # get env picture
    #         env_pic = env.render(mode='rgb_array')
    #         # break if episode terminated
    #         if done:
    #             break