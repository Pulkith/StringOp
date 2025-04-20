from pose_graph import GlobalPose, LocalPose, ObjectAttributes, GlobalState

state = GlobalState()

for i in range(2):
    state.set_drone_data(
        drone_id="drone1",
        objects=[
            (LocalPose(320 + 1*i, 240, 0), ObjectAttributes((255, 0, 0), 1.0, 1.0))
        ],
        drone_pose=GlobalPose(i, 0, 0, yaw=0),
    )
    

    print(state.drones)
    print(state.objects)

