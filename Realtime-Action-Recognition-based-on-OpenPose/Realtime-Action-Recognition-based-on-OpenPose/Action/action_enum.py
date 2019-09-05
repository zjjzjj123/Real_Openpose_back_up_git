from enum import Enum


class Actions(Enum):
    """
    Actions enum
    """
    # framewise_recognition.h5
    # squat = 0
    # stand = 1
    # walk = 2
    # wave = 3

    # framewise_recognition_under_scene.h5
    # stand = 1
    # walk = 0
    # # operate = 2
    # fall_down = 2
    # # run = 4

    # framewise_recognition_my.h5
    # stand = 1
    # walk = 0
    # # operate = 2
    # fall_down = 2
    # # run = 4

    # framewise_recognition_bobei.h5
    # stand = 1
    walk = 1
    # operate = 2
    fall_down = 0
    # run = 4


if __name__ == '__main__':
    print(Actions.stand.name)
    print(Actions.stand.value)