#!/usr/bin/env python3
import requests
import json
from datetime import datetime, timezone
import rclpy
from rclpy.node import Node

from hri_ergonomic_msgs.msg import RebaAssessmentList

ORION_LD_URL = "http://localhost:1026/ngsi-ld/v1/entities"
CORE_CONTEXT = "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
HEADERS = {
    "Content-Type": "application/json",
    "Link": f'<{CORE_CONTEXT}>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'
}
REBA_TOPIC = "/humans/bodies/ergonomics/reba"


class FiwareRebaBridgeNode(Node):
    def __init__(self):
        super().__init__('fiware_reba_bridge_ld')

        self.subscription = self.create_subscription(
            RebaAssessmentList,
            REBA_TOPIC,
            self.reba_callback,
            10
        )

        self.get_logger().info("NGSI-LD & TRoE FIWARE Bridge Started.")

    def get_iso_time_from_header(self, header):
        """Generate an ISO 8601 UTC timestamp from a ROS header for Grafana."""
        sec = header.stamp.sec
        nanosec = header.stamp.nanosec
        dt = datetime.fromtimestamp(sec + nanosec / 1e9, tz=timezone.utc)
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def send_to_orion_ld(self, assessment, timestamp):
        """Send a single valid ergonomic assessment to Orion-LD."""
        entity_id = f"urn:ngsi-ld:ErgonomicAssessment:{assessment.key}"

        # NGSI-LD PATCH payload containing all REBA assessment attributes
        patch_payload = {
            "completeness": {
                "type": "Property",
                "value": float(assessment.completeness),
                "observedAt": timestamp
            },
            "scoreA": {
                "type": "Property",
                "value": int(assessment.score_a),
                "observedAt": timestamp
            },
            "scoreB": {
                "type": "Property",
                "value": int(assessment.score_b),
                "observedAt": timestamp
            },
            "scoreC": {
                "type": "Property",
                "value": int(assessment.score_c),
                "observedAt": timestamp
            },
            "riskLevel": {
                "type": "Property",
                "value": int(assessment.risk_level),
                "observedAt": timestamp
            },
            "groupAValid": {
                "type": "Property",
                "value": bool(assessment.group_a_valid),
                "observedAt": timestamp
            },
            "groupBValid": {
                "type": "Property",
                "value": bool(assessment.group_b_valid),
                "observedAt": timestamp
            },
            # --- Per-region REBA breakdown (Group A: body) ---
            "neckScore": {
                "type": "Property",
                "value": int(assessment.neck_score),
                "observedAt": timestamp
            },
            "trunkScore": {
                "type": "Property",
                "value": int(assessment.trunk_score),
                "observedAt": timestamp
            },
            "legScore": {
                "type": "Property",
                "value": int(assessment.leg_score),
                "observedAt": timestamp
            },
            # --- Per-region REBA breakdown (Group B: arm) ---
            "upperArmScore": {
                "type": "Property",
                "value": int(assessment.upper_arm_score),
                "observedAt": timestamp
            },
            "lowerArmScore": {
                "type": "Property",
                "value": int(assessment.lower_arm_score),
                "observedAt": timestamp
            },
            "wristScore": {
                "type": "Property",
                "value": int(assessment.wrist_score),
                "observedAt": timestamp
            },
            "neckConfidence": {
                "type": "Property",
                "value": float(assessment.body_confidence[0]),
                "observedAt": timestamp
            },
            "trunkConfidence": {
                "type": "Property",
                "value": float(assessment.body_confidence[2]),
                "observedAt": timestamp
            },
            "legConfidence": {
                "type": "Property",
                "value": float(assessment.body_confidence[4]),
                "observedAt": timestamp
            },
            "upperArmConfidence": {
                "type": "Property",
                "value": float(assessment.arm_confidence[0]),
                "observedAt": timestamp
            },
            "lowerArmConfidence": {
                "type": "Property",
                "value": float(assessment.arm_confidence[4]),
                "observedAt": timestamp
            },
        }

        patch_url = f"{ORION_LD_URL}/{entity_id}/attrs"

        try:
            response = requests.post(
                patch_url,
                data=json.dumps(patch_payload),
                headers=HEADERS
            )

            # If the entity does not exist yet, create it with a POST request
            if response.status_code == 404:
                create_payload = {
                    "id": entity_id,
                    "type": "ErgonomicAssessment",
                    **patch_payload
                }

                response = requests.post(
                    ORION_LD_URL,
                    data=json.dumps(create_payload),
                    headers=HEADERS
                )

                self.get_logger().info(f"Created new entity: {entity_id}")
                
            elif response.status_code not in [200, 201, 204]:
                self.get_logger().warn(f"Orion response: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Orion-LD connection error: {e}")

    def reba_callback(self, msg):
        """Process all REBA assessments received in the RebaAssessmentList message."""
        
        # Zaman damgasını rosbag'in kayıtlı ROS mesaj başlığından (header) al
        timestamp = self.get_iso_time_from_header(msg.header)

        for assessment in msg.assessments:
            if not assessment.key:
                continue
            
            # Doğru zaman damgasını fonksiyona parametre olarak ilet
            self.send_to_orion_ld(assessment, timestamp)


def main(args=None):
    rclpy.init(args=args)
    node = FiwareRebaBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()