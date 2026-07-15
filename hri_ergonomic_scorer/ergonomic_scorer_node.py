import rclpy
from rclpy.node import Node
import numpy as np
import math
from rclpy.qos import qos_profile_sensor_data

# will confirm the exact import path after the ROSbag files
from hri_msgs.msg import Skeleton3D, Skeleton3DList

# Import the original rs9000 REBA algorithm we migrated into our package.
from .reba import RebaScore

def calculate_angle(p1, p2, p3):
    """
    Calculates the angle in degrees between three points. p2 is the vertex. Format: [x, y, z]
    """
    ux, uy, uz = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
    vx, vy, vz = p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]
    
    dot_product = ux*vx + uy*vy + uz*vz
    mag_u = math.sqrt(ux**2 + uy**2 + uz**2)
    mag_v = math.sqrt(vx**2 + vy**2 + vz**2)
    
    if mag_u == 0 or mag_v == 0:
        return 0.0
        
    cos_theta = dot_product / (mag_u * mag_v)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    angle_rad = math.acos(cos_theta)
    return math.degrees(angle_rad)

class ErgonomicScorerNode(Node):
    def __init__(self):
        super().__init__('ergonomic_scorer_node')

        # Subscriber to 3D skeleton data.
        self.subscription = self.create_subscription(
            Skeleton3DList,
            '/humans/bodies/skel3D',
            self.skeleton_callback,
            qos_profile_sensor_data
        )

        # REBA risk threshold parameter.
        self.declare_parameter('high_risk_threshold', 8)

        threshold = self.get_parameter('high_risk_threshold').get_parameter_value().integer_value
        self.get_logger().info(f"Parameter changed. Threshold: {threshold}")

        self.get_logger().info("Ergonomic Scorer Node started. Waiting for 3D Skeleton data...")

    def skeleton_callback(self, msg):
        """
        Parses the incoming Skeleton3DList message and constructs the 
        REBA pose matrix for the first detected human with valid data.
        """
        # Extract the list of skeletons from the message
        human_list = getattr(msg, 'skeletons', None)
        
        # Fallback in case the message structure directly provides a list
        if human_list is None:
            human_list = msg if isinstance(msg, list) else [msg]

        # Ignore empty frames
        if not human_list or len(human_list) == 0:
            return

        # Target the first detected human in the frame
        # TODO: Implement ID tracking (msg.key) for multi-human scenarios
        target_human = human_list[0]

        # Initialize the 16x3 matrix for reba.py
        pose_matrix = np.zeros((16, 3))

        # Prevent log spam when the camera loses tracking (all zeros)
        # Standard ROS4HRI indices -> 1: Neck, 8: Mid-Hip
        if target_human.skeleton[1].x == 0.0 and target_human.skeleton[8].x == 0.0:
            return  

        # Populate the matrix, handling missing (NaN) data
        for i in range(16):
            if math.isnan(target_human.skeleton[i].x):
                pose_matrix[i] = [0.0, 0.0, 0.0]
            else:
                pose_matrix[i] = [
                    target_human.skeleton[i].x, 
                    target_human.skeleton[i].y, 
                    target_human.skeleton[i].z
                ]

        neck_pt = pose_matrix[1]
        hip_pt = pose_matrix[8]
        
        # Create a virtual vertical reference point (Z-axis up)
        vertical_ref_pt = [hip_pt[0], hip_pt[1], hip_pt[2] + 1.0]
        
        # Calculate trunk flexion/extension angle
        trunk_angle = calculate_angle(neck_pt, hip_pt, vertical_ref_pt)
        
        self.get_logger().info(f"Real-time Data - Trunk Flexion Angle: {trunk_angle:.2f} degrees")

        try:
            pass # TODO: Connect REBA algorithm here
        except Exception as e:
            self.get_logger().error(f"REBA calculation error: {e}")

def main(args=None):

    rclpy.init(args=args)

    # Create the ergnomic scorer node
    node = ErgonomicScorerNode()

    try:
        # Keep the node running.
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n[INFO] Node stopped by user.")
    
    finally:

        node.destroy_node()
    
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()