# fmt: off
from pygame.locals import K_LEFT, K_RIGHT, K_d, K_q, K_z

# fmt: on


class Configuration:
    """ TODO
    """

    def __init__(self) -> None:

        # APP CONFIGURATION

        # folder for run's data files
        self.DATA_FOLDER: str = "data"
        self.DATA_FILE: str = "data"
        # toggles random generation
        self.RANDOM_GEN = False
        # disables losing, for dev/testing purposes
        self.CAN_LOSE = True

        # GAME CONFIGURATION

        # coefficient of Proportions (to increase physics proportionally)
        self.PROPORTION = 1.0
        # size of blocks
        self.BLOCK_WIDTH = int(16 * self.PROPORTION)
        self.BLOCK_HEIGHT = int(16 * self.PROPORTION)
        # size of chunk
        self.CHUNK_HEIGHT = 16
        # for the display
        self.VISIBILITY_X = 45  # width of the window in amount of blocks.
        self.VISIBILITY_Y = 2  # height of the window in amount of chunk height.
        self.SIZE_X = (
            self.BLOCK_WIDTH * self.VISIBILITY_X
        )  # width of the window in pixels.
        self.SIZE_Y = (
            self.CHUNK_HEIGHT * self.BLOCK_HEIGHT * self.VISIBILITY_Y
        )  # height of the window in pixels.
        # player size
        self.PLAYER_WIDTH = int(1 * self.BLOCK_WIDTH)
        self.PLAYER_HEIGHT = int(2 * self.BLOCK_HEIGHT)
        # start
        self.START_X = 0
        self.START_Y = self.SIZE_Y - self.BLOCK_HEIGHT - self.PLAYER_HEIGHT
        # colors
        self.WHITE = (255, 255, 255)
        self.GREY = (30, 30, 30)
        self.ORANGE = (255, 125, 0)
        self.RED = (255, 0, 0)
        # acceleration
        self.ACCELERATION_X = float(1 * self.PROPORTION)
        self.ACCELERATION_Y = 1.67 * self.PROPORTION
        self.COEFF_ACCELERATION_X = 1.3
        self.SLOWDOWN__X = self.COEFF_ACCELERATION_X * self.ACCELERATION_X
        # speed
        self.SPEED_X = 16 * self.PROPORTION  # speed_x max
        self.SPEED_Y = self.BLOCK_HEIGHT * (14 / 16)  # speed_y max
        # camera speed
        self.SPEED_CAMERA_X = 80 * self.PROPORTION

        # GAME CONTROLS

        # Keys
        self.KEY_RIGHT = K_d
        self.KEY_LEFT = K_q
        self.KEY_UP = K_z

        # Camera keys
        self.CAMERA_RIGHT = K_RIGHT
        self.CAMERA_LEFT = K_LEFT
