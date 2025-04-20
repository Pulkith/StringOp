from src.tello_target_tracker.tello_target_tracker.DJITelloPy.djitellopy import Tello, TelloSwarm
import argparse

nets = [
    "wlp0s20f3",
    "wlx8c902de88812",
    "wlxdc6279da55db"
]


def test_swarm(net):
    tello = Tello(net=net)
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Tello Swarm")
    parser.add_argument(
        "--net",
        type=int,
        default=1,
        help="Network interface index to use for the Tello drones (default: 0)"
    )
    args = parser.parse_args()

    net = nets[args.net]

    print(f"Using network interface: {net}")
    test_swarm(net)


    