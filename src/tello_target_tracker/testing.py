import threading
import socket
import time
import enum
import datetime
import os

class DroneState(enum.Enum):
    DISCONNECTED = 0
    CONNECTED = 1
    FLYING = 2
    MISSION = 3
    LANDING = 4
    COMPLETED = 5

class TelloController:
    def __init__(self):
        # Connection settings
        self.local_address = ('', 9000)
        self.tello_address = ('192.168.10.1', 8889)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.local_address)
        
        # State management
        self.state = DroneState.DISCONNECTED
        self.response = None
        self.mission_step = 0
        
        # Telemetry data
        self.battery = None
        self.height = None
        self.attitude = {"pitch": None, "roll": None, "yaw": None}
        self.velocity = {"x": None, "y": None, "z": None}
        self.acceleration = {"x": None, "y": None, "z": None}
        self.position = {"x": 0, "y": 0, "z": 0}  # Estimated position (starting at 0,0,0)
        self.drift_correction = {"x": 0, "y": 0}  # Correction factors for drift
        
        # Mission parameters
        self.side_length = 100
        self.altitude_changes = [50, 100, 150, 100]
        
        # Logging
        self.log_file = f"tello_mission_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Start receiver thread
        self.receiver_thread = threading.Thread(target=self._receive_response)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
        
        # Start telemetry thread
        self.telemetry_thread = threading.Thread(target=self._update_telemetry)
        self.telemetry_thread.daemon = True
        
        # Start visualization thread
        self.visualization_thread = threading.Thread(target=self._display_state)
        self.visualization_thread.daemon = True
    
    def _receive_response(self):
        """Thread to receive and process drone responses"""
        while True:
            try:
                data, server = self.sock.recvfrom(1518)
                response = data.decode("utf-8")
                self.log(f"Response: {response}")
                self.response = response
            except Exception as e:
                self.log(f"Error receiving: {str(e)}")
                break
    
    def _update_telemetry(self):
        """Thread to regularly request and update telemetry data"""
        telemetry_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        telemetry_sock.bind(('', 8890))
        
        while True:
            try:
                data, server = telemetry_sock.recvfrom(1518)
                telemetry_str = data.decode("utf-8")
                
                # Parse telemetry data
                telemetry_parts = telemetry_str.strip().split(';')
                telemetry_dict = {}
                
                for part in telemetry_parts:
                    if ':' in part:
                        key, value = part.split(':')
                        telemetry_dict[key] = value
                
                # Update state values
                if 'bat' in telemetry_dict:
                    self.battery = int(telemetry_dict['bat'])
                if 'h' in telemetry_dict:
                    self.height = int(telemetry_dict['h'])
                if 'pitch' in telemetry_dict:
                    self.attitude["pitch"] = int(telemetry_dict['pitch'])
                if 'roll' in telemetry_dict:
                    self.attitude["roll"] = int(telemetry_dict['roll'])
                if 'yaw' in telemetry_dict:
                    self.attitude["yaw"] = int(telemetry_dict['yaw'])
                if 'vgx' in telemetry_dict:
                    self.velocity["x"] = int(telemetry_dict['vgx'])
                if 'vgy' in telemetry_dict:
                    self.velocity["y"] = int(telemetry_dict['vgy'])
                if 'vgz' in telemetry_dict:
                    self.velocity["z"] = int(telemetry_dict['vgz'])
                if 'agx' in telemetry_dict:
                    self.acceleration["x"] = float(telemetry_dict['agx'])
                if 'agy' in telemetry_dict:
                    self.acceleration["y"] = float(telemetry_dict['agy'])
                if 'agz' in telemetry_dict:
                    self.acceleration["z"] = float(telemetry_dict['agz'])
                
                # Estimate position based on velocity (very rough)
                if all(v is not None for v in self.velocity.values()):
                    self.position["x"] += self.velocity["x"] * 0.1  # Assuming 100ms between readings
                    self.position["y"] += self.velocity["y"] * 0.1
                    self.position["z"] = self.height if self.height is not None else self.position["z"] + self.velocity["z"] * 0.1
                
            except Exception as e:
                self.log(f"Telemetry error: {str(e)}")
                time.sleep(0.1)
    
    def _display_state(self):
        """Thread to periodically display the drone state in the console"""
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Format the display
            print("\n" + "="*50)
            print(f"TELLO DRONE STATE MONITOR - {datetime.datetime.now().strftime('%H:%M:%S')}")
            print("="*50)
            
            # Current state
            print(f"\nSTATE: {self.state.name}")
            print(f"Mission Step: {self.mission_step}")
            
            # Battery and connection
            print(f"\nBattery: {self.battery}%" if self.battery is not None else "\nBattery: Unknown")
            
            # Position and altitude
            print("\nPOSITION (estimated):")
            print(f"  X: {self.position['x']:.2f} cm")
            print(f"  Y: {self.position['y']:.2f} cm")
            print(f"  Height: {self.height} cm" if self.height is not None else "  Height: Unknown")
            
            # Attitude
            print("\nATTITUDE:")
            print(f"  Pitch: {self.attitude['pitch']}°" if self.attitude['pitch'] is not None else "  Pitch: Unknown")
            print(f"  Roll: {self.attitude['roll']}°" if self.attitude['roll'] is not None else "  Roll: Unknown")
            print(f"  Yaw: {self.attitude['yaw']}°" if self.attitude['yaw'] is not None else "  Yaw: Unknown")
            
            # Velocity
            print("\nVELOCITY:")
            print(f"  X: {self.velocity['x']} cm/s" if self.velocity['x'] is not None else "  X: Unknown")
            print(f"  Y: {self.velocity['y']} cm/s" if self.velocity['y'] is not None else "  Y: Unknown")
            print(f"  Z: {self.velocity['z']} cm/s" if self.velocity['z'] is not None else "  Z: Unknown")
            
            # Drift analysis
            if all(v is not None for v in [self.attitude['pitch'], self.attitude['roll'], 
                                          self.velocity['x'], self.velocity['y']]):
                if abs(self.attitude['pitch']) > 5 or abs(self.attitude['roll']) > 5:
                    print("\nDRIFT ALERT:")
                    print(f"  Pitch angle: {self.attitude['pitch']}° (±2° expected)")
                    print(f"  Roll angle: {self.attitude['roll']}° (±2° expected)")
                
                if (abs(self.velocity['x']) > 15 or abs(self.velocity['y']) > 15) and self.state != DroneState.MISSION:
                    print(f"  Unexpected motion detected when stationary:")
                    print(f"  X velocity: {self.velocity['x']} cm/s")
                    print(f"  Y velocity: {self.velocity['y']} cm/s")
            
            print("\n" + "="*50)
            time.sleep(0.5)
    
    def log(self, message):
        """Log a message to both console and file"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def send_command(self, command, wait_time=3):
        """Send a command to the drone and wait for response"""
        self.log(f"Sending: {command}")
        self.response = None
        self.sock.sendto(command.encode("utf-8"), self.tello_address)
        
        # Wait for response
        timeout = time.time() + wait_time
        while self.response is None and time.time() < timeout:
            time.sleep(0.1)
        
        return self.response
    
    def connect(self):
        """Initialize SDK mode and start monitoring threads"""
        if self.state == DroneState.DISCONNECTED:
            self.log("Connecting to Tello drone...")
            response = self.send_command("command")
            if response == "ok":
                self.log("Connected to Tello drone")
                self.state = DroneState.CONNECTED
                
                # Get initial battery
                battery_response = self.send_command("battery?")
                if battery_response.isdigit():
                    self.battery = int(battery_response)
                    self.log(f"Battery level: {self.battery}%")
                    
                # Start monitoring threads
                self.telemetry_thread.start()
                self.visualization_thread.start()
                
                return True
            else:
                self.log("Connection failed")
        return False
    
    def takeoff(self):
        """Take off the drone with detailed status"""
        if self.state == DroneState.CONNECTED:
            self.log("Initiating takeoff sequence...")
            
            # Check battery level before takeoff
            if self.battery is not None and self.battery < 20:
                self.log("WARNING: Low battery level. Takeoff not recommended!")
                return False
                
            response = self.send_command("takeoff", 7)
            if response == "ok":
                self.state = DroneState.FLYING
                self.log("Takeoff successful - Hover stabilization in progress")
                time.sleep(3)  # Allow time to stabilize
                self.log("Hover stabilized - Ready for commands")
                return True
            else:
                self.log(f"Takeoff failed: {response}")
        return False
    
    def land(self):
        """Land the drone with detailed status"""
        if self.state in [DroneState.FLYING, DroneState.MISSION]:
            self.log("Initiating landing sequence...")
            response = self.send_command("land", 5)
            if response == "ok":
                self.state = DroneState.LANDING
                self.log("Landing in progress...")
                time.sleep(3)
                self.state = DroneState.COMPLETED
                self.log("Landing completed successfully")
                return True
            else:
                self.log(f"Landing command failed: {response}")
        return False
    
    def monitor_drift(self):
        """Monitor and correct for drift"""
        if self.state in [DroneState.FLYING, DroneState.MISSION]:
            # Check for significant attitude drift
            if (self.attitude["pitch"] is not None and abs(self.attitude["pitch"]) > 4) or \
               (self.attitude["roll"] is not None and abs(self.attitude["roll"]) > 4):
                self.log(f"Drift detected - Pitch: {self.attitude['pitch']}°, Roll: {self.attitude['roll']}°")
                
                # Execute hover command to stabilize
                self.log("Executing hover command to stabilize")
                self.send_command("stop")
                time.sleep(1)
                
                return True
        return False
    
    def execute_mission(self):
        """Execute the square flight pattern with detailed status and drift monitoring"""
        if self.state == DroneState.FLYING:
            self.state = DroneState.MISSION
            self.log("Starting mission: Square flight pattern with altitude changes")
            
            # Mission steps with detailed descriptions
            mission_steps = [
                {"command": "speed 50", "description": "Setting speed to 50 cm/s"},
                {"command": f"up {self.altitude_changes[0]}", "description": f"Ascending to {self.altitude_changes[0]} cm for first side"},
                {"command": f"forward {self.side_length}", "description": f"Flying forward {self.side_length} cm (first side of square)"},
                {"command": f"up {self.altitude_changes[1]}", "description": f"Ascending to altitude {sum(self.altitude_changes[:2])} cm for second side"},
                {"command": f"right {self.side_length}", "description": f"Flying right {self.side_length} cm (second side of square)"},
                {"command": f"up {self.altitude_changes[2]}", "description": f"Ascending to altitude {sum(self.altitude_changes[:3])} cm for third side"},
                {"command": f"back {self.side_length}", "description": f"Flying backward {self.side_length} cm (third side of square)"},
                {"command": f"up {self.altitude_changes[3]}", "description": f"Ascending to altitude {sum(self.altitude_changes)} cm for fourth side"},
                {"command": f"left {self.side_length}", "description": f"Flying left {self.side_length} cm (fourth side of square)"},
                {"command": "down 300", "description": "Descending to starting altitude"}
            ]
            
            # Execute each step
            for i, step in enumerate(mission_steps):
                self.mission_step = i + 1
                self.log(f"Mission step {self.mission_step}/{len(mission_steps)}: {step['description']}")
                
                # Send command
                response = self.send_command(step["command"], 7)
                
                if response != "ok":
                    self.log(f"Mission step failed: {response}")
                    self.state = DroneState.FLYING
                    return False
                
                # After each movement, check for drift
                if "forward" in step["command"] or "back" in step["command"] or \
                   "left" in step["command"] or "right" in step["command"]:
                    self.log("Checking position and orientation...")
                    time.sleep(1)
                    
                    # Monitor and correct drift if necessary
                    if self.monitor_drift():
                        self.log("Drift correction applied - continuing mission")
                    
                    self.log(f"Position estimate: X={self.position['x']:.1f}, Y={self.position['y']:.1f}, Z={self.position['z']:.1f}")
                
                # Short pause between commands
                time.sleep(2)
            
            self.log("Square flight pattern completed successfully!")
            self.mission_step = 0
            return True
        
        return False
    
    def run_mission(self):
        """Run the complete mission sequence with detailed logging"""
        self.log("=== TELLO DRONE MISSION: SQUARE FLIGHT PATTERN ===")
        
        # Connect to the drone
        if not self.connect():
            self.log("Mission aborted: Failed to connect to the drone")
            return False
        
        # Take off
        if not self.takeoff():
            self.log("Mission aborted: Failed to take off")
            return False
        
        # Pause for stability
        self.log("Stabilizing before starting mission...")
        time.sleep(3)
        
        # Execute the square pattern mission
        if not self.execute_mission():
            self.log("Mission execution failed, attempting to land")
            self.land()
            return False
        
        # Return home and land
        self.log("Mission completed, returning home and landing")
        if not self.land():
            self.log("Failed to land")
            return False
        
        self.log("=== MISSION COMPLETED SUCCESSFULLY ===")
        return True
    
    def close(self):
        """Close the socket connection"""
        self.sock.close()
        self.log("Connection closed")
        
        # Final message about log file
        print(f"\nMission log saved to: {self.log_file}")

if __name__ == "__main__":
    # Create and run the controller
    try:
        controller = TelloController()
        controller.run_mission()
    except KeyboardInterrupt:
        print("\nMission aborted by user")
    finally:
        if 'controller' in locals():
            # Ensure the drone lands if interrupted
            if controller.state in [DroneState.FLYING, DroneState.MISSION]:
                controller.land()
            controller.close()