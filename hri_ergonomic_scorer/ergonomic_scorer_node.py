import rclpy
from rclpy.node import Node
import numpy as np
import math

# will confirm the exact import path after the ROSbag files
from hri_msgs.msg import Skeleton3D

# Import the original rs9000 REBA algorithm we migrated into our package.
from .reba import RebaScore

class ErgonomicScorerNode(Node):
    def __init__(self):
        super().__init__('ergonomic_scorer_node')

        # Subscriber to listen to the 3D skeleton data on the network.
        # UPDATE the topic '/humans/bodies/3d' 
        self.subscription = self.create_subscription(
            Skeleton3D,             # The message type we expect to receive
            '/humans/bodies/poses',    # The network topic we are listening to
            self.skeleton_callback, # The function to call when a message arrives
            10                      # QoS profile: Queue size for incomin messages
        )

        # Instantiate our original REBA calculation class
        self.reba_calculator = RebaScore()

        # Log a startup message to the terminal.
        self.get_logger().info("Ergonomic Scorer Node has successfully started. Waiting for 3D Skeleton data...")

    def skeleton_callback(self, msg):
        """
        This function is triggered automatically every time 3D skeleton data is published on the Vulcanexus network.
        Its mission: Convet the incoming ROS message into a clean, error-free 16x3 NumPy matrix expected by the reba.py script.
        """

        # reba.py expects a matrix of 16 rows and 3 columns (x, y, z)
        pose_matrix = np.zeros((16, 3))

        # MISSING DATA AND TOLERANCE
        # Loop through the joints in the incoming message.

        # for i in range(16):
        #     # Check if the camera missed the joint and sent a NaN
        #     if math.isnan(msg.keypoints[i].x):
        #         self.get_logger().warn(f"Joint {i} is missing! Crash prevented,assigning neutral value.")

        #         # Assign a default coordinate to prevent a mathematical crash
        #         pose_matrix[i] = [0.0, 0.0, 0.0]
        #     else:
        #         # If the data is valid write the true coordinates to our matrix
        #         pose_matrix[i] = [msg.keypoints[i].x, msg.keypoints[i].y, msg.keypoints[i].z]

        # REBA CALCULATION
        try:
            # body_params = self.reba_calculator.get_body_angles_fromm_pose_right(pose_matrix)
            # self.reba_calculator.set_body(body_params)
            # score_a, _ = self.reba_calculator.compute_score_a()

            # TODO: Publish the final calculated REBA score to a new topic on the network.
            pass
        except Exception as e:
            # If reba.py fails catch the error and keep the node running
            self.get_logger().error(f"An unexpected erroroccurred during REBA calculaion: {e}")

def main(args=None):
    # Initialize the ROS2 Python client library
    rclpy.init(args=args)

    # Create an instance of our node
    node = ErgonomicScorerNode()

    # Keep the node alive, alllowing it to continuously listen for callbacks
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()