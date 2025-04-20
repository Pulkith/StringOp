from src.tello_target_tracker.tello_target_tracker.DJITelloPy.djitellopy import Tello, TelloSwarm
import argparse

from src.tello_target_tracker.tello_target_tracker.pose_estimation import PoseEstimation
import time

nets = [
    "wlp0s20f3",
    "wlx8c902de88812",
    "wlxdc6279da55db"
]


def test_swarm(net):
    tello = Tello(net=net)

    try:
        tello.connect()
        print(f"Battery: {tello.get_battery()}%")
        pose_estimation = PoseEstimation(tello, x_init=0)
        tello.takeoff()
        tello.move_up(100)
        print("Moving up")
        for i in range(10):
            tello.move_forward(50)
            tello.move_back(50)
            tello.rotate_clockwise(90)
            print("Moving forward")
    finally:
        tello.land()
    
        pose_estimation.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Tello Swarm")
    parser.add_argument(
        "--net",
        type=int,
        default=2,
        help="Network interface index to use for the Tello drones (default: 0)"
    )
    args = parser.parse_args()

    net = nets[args.net]

    print(f"Using network interface: {net}")
    test_swarm(net)


    