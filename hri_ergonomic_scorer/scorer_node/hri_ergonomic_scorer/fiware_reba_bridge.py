#!/usr/bin/env python3
import math
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

# Must match RebaAssessment_Constants in the IDL.
SCORE_NOT_ASSESSED = 255
RISK_UNKNOWN = 255


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

    @staticmethod
    def _prop(value, timestamp):
        return {"type": "Property", "value": value, "observedAt": timestamp}

    def build_payload(self, assessment, timestamp):
        """
        Build the NGSI-LD attribute set for one assessment.

        Scores whose underlying item was never observed are OMITTED, not sent
        as their neutral value. Writing them anyway is what let a fully
        occluded body show up in Grafana as a solid "negligible risk" reading:
        Orion has no way to tell a measured 1 from a fabricated one.
        """
        payload = {
            "completeness": self._prop(float(assessment.completeness), timestamp),
            "groupAValid": self._prop(bool(assessment.group_a_valid), timestamp),
            "groupBValid": self._prop(bool(assessment.group_b_valid), timestamp),
            "scoreIsLowerBound": self._prop(bool(assessment.score_is_lower_bound), timestamp),
        }

        def add_score(name, value, assessed=True):
            if assessed and int(value) != SCORE_NOT_ASSESSED:
                payload[name] = self._prop(int(value), timestamp)

        # --- Final score and risk level ---
        add_score("scoreA", assessment.score_a, assessment.group_a_valid)
        add_score("scoreB", assessment.score_b, assessment.group_b_valid)
        if int(assessment.score_c) != SCORE_NOT_ASSESSED:
            payload["scoreC"] = self._prop(int(assessment.score_c), timestamp)
        if int(assessment.risk_level) != RISK_UNKNOWN:
            payload["riskLevel"] = self._prop(int(assessment.risk_level), timestamp)

        # --- Per-region REBA breakdown (Group A: body) ---
        add_score("neckScore", assessment.neck_score, assessment.neck_assessed)
        add_score("trunkScore", assessment.trunk_score, assessment.trunk_assessed)
        add_score("legScore", assessment.leg_score, assessment.legs_assessed)
        add_score("loadScore", assessment.load_score, assessment.load_known)

        # --- Per-region REBA breakdown (Group B: arm) ---
        add_score("upperArmScore", assessment.upper_arm_score, assessment.upper_arm_assessed)
        add_score("lowerArmScore", assessment.lower_arm_score, assessment.lower_arm_assessed)
        add_score("wristScore", assessment.wrist_score, assessment.wrist_assessed)
        add_score("couplingScore", assessment.coupling_score, assessment.coupling_known)

        # --- Observation flags, so a dashboard can render "not measured"
        #     differently from "measured and fine" ---
        payload["neckAssessed"] = self._prop(bool(assessment.neck_assessed), timestamp)
        payload["trunkAssessed"] = self._prop(bool(assessment.trunk_assessed), timestamp)
        payload["legsAssessed"] = self._prop(bool(assessment.legs_assessed), timestamp)
        payload["upperArmAssessed"] = self._prop(bool(assessment.upper_arm_assessed), timestamp)
        payload["lowerArmAssessed"] = self._prop(bool(assessment.lower_arm_assessed), timestamp)
        payload["wristAssessed"] = self._prop(bool(assessment.wrist_assessed), timestamp)

        # --- DEBA ---
        payload["debaValid"] = self._prop(bool(assessment.deba_valid), timestamp)
        if assessment.deba_valid and not math.isnan(assessment.deba_score):
            payload["debaScore"] = self._prop(float(assessment.deba_score), timestamp)

        # --- Confidence readouts (0 means the region backed no computation) ---
        for name, value in (
            ("neckConfidence", assessment.body_confidence[0]),
            ("trunkConfidence", assessment.body_confidence[2]),
            ("legConfidence", assessment.body_confidence[4]),
            ("upperArmConfidence", assessment.arm_confidence[0]),
            ("lowerArmConfidence", assessment.arm_confidence[4]),
        ):
            if value > 0.0:
                payload[name] = self._prop(float(value), timestamp)

        return payload

    def send_to_orion_ld(self, assessment, timestamp):
        """Send a single valid ergonomic assessment to Orion-LD."""
        entity_id = f"urn:ngsi-ld:ErgonomicAssessment:v2:{assessment.key}"
        patch_payload = self.build_payload(assessment, timestamp)
        patch_url = f"{ORION_LD_URL}/{entity_id}/attrs"

        try:
            response = requests.post(
                patch_url,
                data=json.dumps(patch_payload),
                headers=HEADERS,
                timeout=2.0
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
                    headers=HEADERS,
                    timeout=2.0
                )

                self.get_logger().info(f"Created new entity: {entity_id}")

            elif response.status_code not in [200, 201, 204]:
                self.get_logger().warn(f"Orion response: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Orion-LD connection error: {e}")

    def reba_callback(self, msg):
        """Process all REBA assessments received in the RebaAssessmentList message."""
        timestamp = self.get_iso_time_from_header(msg.header)

        for assessment in msg.assessments:
            if not assessment.key:
                continue
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
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
