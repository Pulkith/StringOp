from pose_graph import GlobalPose, LocalPose, ObjectAttributes, GlobalState
from convergence_visualization.simulations import weights_sim


class TestWeightsSim:
    def __init__(self):
        self.state = GlobalState()

        for i in range(2):
            self.state.update_drone_pose(
                drone_id=f"drone{i}",
                new_pose=GlobalPose(x=i, y=0, z=0, yaw=0)
            )
            self.state.update_drone_objects(
                drone_id=f"drone{i}",
                objects=[
                    (LocalPose(x=i, y=0, z=0, yaw=0), ObjectAttributes(r=1, g=1, b=1, size_x=1, size_y=1))
                ]
            )
            

        goals = weights_sim.run(
            drone_positions=[
                (self.state.drones[drone_id].x, self.state.drones[drone_id].y)
                for drone_id in self.state.drones.keys()
            ],
            poi_positions=[
                (obj[0].x, obj[0].y)
                for obj in self.state.objects
            ]
        )


TestWeightsSim()