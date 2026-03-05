from pymavlink import mavutil

class DroneController:
    """ Handles MAVLink communicaiton with the drone """
    
    def __init__(self, drone_ip, drone_port):
        self.drone_ip = drone_ip
        self.drone_port = drone_port
        self.master = None
        self.connected = False

    def connect(self):
        """Connect to the drone via MAVLink"""
        connection_string = f"tcp:{self.drone_ip}:{self.drone_port}"
        print(f"Connecting to drone at {connection_string}...")
        try:
            self.master = mavutil.mavlink_connection(connection_string)
            mavutil.mavlink_connection(connection_string)
            self.master.wait_heartbeat(timeout=10)
            self.connected = True
            print(f"✓ Connected to drone (System ID: {self.master.target_system})")
            return True
        except Exception as e:
            print(f"ERROR: Could not connect to drone: {e}")
            return False
    
    def turn(self, angle: int):
        self.master.command_long_send(self.master.target_system, self.master.target_component, mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, angle, 25, 0, 0, 0, 0, 0)
        
    def close(self):
        """Close Connection to drone"""
    if self.master:
        try:
            self.master.close()
            print("Drone connection closed")
        except:
            pass

drone_ip = "192.168.1.100"
drone_port = 5678

controller = DroneController(drone_ip, drone_port)
if not controller.connect():
    print("FATAL: Cannot connect to drone!")

# Have the drone turn by 45 degrees
controller.turn(45)


