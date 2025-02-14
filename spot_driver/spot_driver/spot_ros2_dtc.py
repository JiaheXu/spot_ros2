#!/usr/bin/env python3
# Debug
# from ros_helpers import *
import logging
import os
import tempfile
import threading
import time
import traceback
import typing
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import bdai_ros2_wrappers.process as ros_process
import builtin_interfaces.msg
import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros
from bdai_ros2_wrappers.node import Node
from bdai_ros2_wrappers.single_goal_action_server import (
    SingleGoalActionServer,
)
from bdai_ros2_wrappers.single_goal_multiple_action_servers import (
    SingleGoalMultipleActionServers,
)
from bosdyn.api import (
    geometry_pb2,
    gripper_camera_param_pb2,
    manipulation_api_pb2,
    robot_command_pb2,
    trajectory_pb2,
    world_object_pb2,
)
from bosdyn.api.geometry_pb2 import Quaternion, SE2VelocityLimit
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api.spot.choreography_sequence_pb2 import Animation, ChoreographySequence, ChoreographyStatusResponse
from bosdyn.client import math_helpers
from bosdyn.client.exceptions import InternalServerError
from bosdyn_api_msgs.math_helpers import bosdyn_localization_to_pose_msg
from bosdyn.client.math_helpers import Quat, SE2Pose, SE3Pose, math
from bosdyn_msgs.conversions import convert
from bosdyn_msgs.msg import (
    ArmCommandFeedback,
    Camera,
    FullBodyCommand,
    FullBodyCommandFeedback,
    GripperCommandFeedback,
    Logpoint,
    ManipulationApiFeedbackResponse,
    MobilityCommandFeedback,
    PtzDescription,
    RobotCommand,
    RobotCommandFeedback,
    RobotCommandFeedbackStatusStatus,
)

from google.protobuf import duration_pb2

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import (arm_command_pb2, geometry_pb2, robot_command_pb2, synchronized_command_pb2,
                        trajectory_pb2)

from bosdyn.client.frame_helpers import BODY_FRAME_NAME, VISION_FRAME_NAME
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient



from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    Twist,
    TransformStamped,
)
from rclpy import Parameter
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import CallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock
from rclpy.impl import rcutils_logger
from rclpy.publisher import Publisher
from rclpy.timer import Rate
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger

import spot_driver.robot_command_util as robot_command_util

from synchros2.tf_listener_wrapper import TFListenerWrapper
from synchros2.utilities import fqn, namespace_with

# DEBUG/RELEASE: RELATIVE PATH NOT WORKING IN DEBUG
# Release
from spot_driver.ros_helpers import (
    get_from_env_and_fall_back_to_param,
)
from spot_msgs.action import (  # type: ignore
    ExecuteDance,
    Manipulation,
    NavigateTo,
    Trajectory,
)
from spot_msgs.action import (  # type: ignore
    RobotCommand as RobotCommandAction,
)
from spot_msgs.msg import (  # type: ignore
    Feedback,
    LeaseArray,
    LeaseResource,
    Metrics,
    MobilityParams,
)
from spot_msgs.srv import (  # type: ignore
    ChoreographyRecordedStateToAnimation,
    ChoreographyStartRecordingState,
    ChoreographyStopRecordingState,
    ClearBehaviorFault,
    DeleteLogpoint,
    DeleteSound,
    Dock,
    GetChoreographyStatus,
    GetGripperCameraParameters,
    GetLEDBrightness,
    GetLogpointStatus,
    GetPtzPosition,
    GetVolume,
    GraphNavClearGraph,
    GraphNavGetLocalizationPose,
    GraphNavSetLocalization,
    GraphNavUploadGraph,
    InitializeLens,
    ListAllDances,
    ListAllMoves,
    ListCameras,
    ListGraph,
    ListLogpoints,
    ListPtz,
    ListSounds,
    ListWorldObjects,
    LoadSound,
    OverrideGraspOrCarry,
    PlaySound,
    RetrieveLogpoint,
    SetGripperAngle,
    SetGripperCameraParameters,
    SetLEDBrightness,
    SetLocomotion,
    SetPtzPosition,
    SetVelocity,
    SetVolume,
    StoreLogpoint,
    TagLogpoint,
    UploadAnimation,
    UploadSequence,
)
from spot_msgs.srv import (  # type: ignore
    RobotCommand as RobotCommandService,
)
from spot_wrapper.cam_wrapper import SpotCamCamera, SpotCamWrapper
from spot_wrapper.wrapper import SpotWrapper

MAX_DURATION = 1e6
COLOR_END = "\33[0m"
COLOR_GREEN = "\33[32m"
COLOR_YELLOW = "\33[33m"

from spot_driver.spot_ros2 import (
    set_node_parameter_from_parameter_list,
    WaitForGoal,
    GoalResponse,
    Response,
    Request,
    SpotROS
)

# from dtc_spot_msgs.srv import Anchor_3D

from geometry_msgs.msg import PointStamped, TwistStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import UInt8, Bool, String
from sensor_msgs.msg import Joy
# from airlab_msgs.msg import AIRLABModes
from std_msgs.msg import String, Float32, Int8, UInt8, Bool, UInt32MultiArray, Int32
from std_srvs.srv import SetBool
from rclpy.client import Client
from std_srvs.srv import Trigger
import rclpy
import time



from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d
import transforms3d as t3d
import math
from tf2_ros import TransformException

class SpotROS_DTC(SpotROS):

    def __init__(self, **kwargs):
        super().__init__( **kwargs )
        self.get_logger().info(COLOR_GREEN + "Hi from spot_driver_dtc." + COLOR_END)
        # _, __ = self.spot_wrapper.spot_arm.gripper_open()
        # self.command_client = self.spot_wrapper.ensure_client(RobotCommandClient.default_service_name)


        self.arm_status = False
        self.last_manual_mode = self.get_clock().now()
        self.manual_override_engaged = False
        
        self.claim_status = False 
        self.power_on_status = False
        self.stand_status = False

        self._robot_name = self.name
        self._body_frame_name = namespace_with(self._robot_name, BODY_FRAME_NAME)
        self._vision_frame_name = namespace_with(self._robot_name, VISION_FRAME_NAME)
        self._ee_frame_name = namespace_with(self._robot_name, "arm_link_wr1")

        self._tf_listener = TFListenerWrapper(self)
        # self._tf_listener.wait_for_a_tform_b(self._body_frame_name, self._vision_frame_name)
        # self._tf_listener.wait_for_a_tform_b( self._ee_frame_name, self._body_frame_name)

        self.estop_active = False
        self.last_estop_value = False
    
        self.has_anchor = False
        self.anchor = PointStamped()
        self.last_ee_pose = PoseStamped()

        # topic
        self.estop_sub = self.create_subscriber(Header, "estop", self.estop_callback, 1)
        self.look_at_sub = self.create_subscriber(PointStamped, "look_at_goal", self.look_at_callback, 1)

        self.startup()

        # output
        # self.armed_status_pub = self.create_publisher(Bool, "status/arm", 1)
        self.ee_pose_pub = self.create_publisher(PoseStamped, "ee_pose_in_body_frame", 1)
        self.body_pose_pub = self.create_publisher(PoseStamped, "body_pose_in_world_frame", 1)

        self.timer = self.create_timer(0.1, self.status_timer_callback)


    def echo(self, msg_str):
        self.get_logger().info(msg_str)


    def startup(self): # some behaviors before we move the robot
        while(True):
            success, msg_info = self.spot_wrapper.spot_arm.gripper_open()
            if(success):
                break
            self.get_logger().info(COLOR_GREEN + "Cannot open Griiper." + COLOR_END)

    def transformstamped_2_posestamped(self, transformstamped):
        
        posestamped = PoseStamped()
        posestamped.pose.position.x = transformstamped.transform.translation.x
        posestamped.pose.position.y = transformstamped.transform.translation.y
        posestamped.pose.position.z = transformstamped.transform.translation.z
        posestamped.pose.orientation = transformstamped.transform.rotation
        posestamped.header = transformstamped.header
        return posestamped

    def status_timer_callback(self):

        # self.get_logger().info(COLOR_GREEN + "status_timer_callback()!!!!!!!!!." + COLOR_END)
        world_t_robot = None
        try:
            world_t_robot = self._tf_listener.lookup_a_tform_b(self._vision_frame_name, self._body_frame_name)
        except TransformException as ex:
            self.get_logger().info("Could not get transform from world(vision) to body")

        if(world_t_robot is not None):
            self.last_body_pose = self.transformstamped_2_posestamped(world_t_robot)
            self.body_pose_pub.publish(self.last_body_pose)

        robot_t_ee = None
        try:
            robot_t_ee = self._tf_listener.lookup_a_tform_b(self._body_frame_name, self._ee_frame_name)
        except TransformException as ex:
            self.get_logger().info("Could not get transform from body to ee")

        if(robot_t_ee is not None):
            self.last_ee_pose = self.transformstamped_2_posestamped(robot_t_ee)
            self.ee_pose_pub.publish(self.last_ee_pose)


        return


    def estop_callback(self, msg):
        
        if(msg.frame_id == "True"):
            self.estop_active = True
        else:
            self.estop_active = False
        
        if(self.estop_active != self.last_estop_value):
            if(self.estop_active): # estop triggered
                self.trigger_estop()
            else:
                self.disengage_estop()

        self.last_estop_value = self.estop_active

    self trigger_estop(self):
        # save arm_pose
        # stow arm
        # trigger gentle steop

        self.saved_arm_pose = self.last_ee_pose

        stow_success, stow_message = self.spot_wrapper.spot_arm.arm_stow()
        estop_success, estop_message = self.spot_wrapper.assertEStop(False)

        return (stow_success and estop_success)

    self disengage_estop(self):
        # disentangle estop
        # stand
        # unstow arm
        # return to saved arm pose

        disengage_success, disengage_message = self.spot_wrapper.disengageEStop()
        stand_success, stand_message = self.spot_wrapper.stand()
        unstow_success, unstow_message = self.spot_wrapper.spot_arm.arm_unstow()

        pose_msg = PoseStamped()
        # timestamp = self.get_clock().now().to_msg()
        # self.saved_arm_pose
        pose_msg = self.saved_arm_pose.pose
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id="body"
        
        self.arm_pose_cmd_callback(self.saved_arm_pose)

        return (disengage_success and stand_success and unstow_success)

    def get_lookat_position(self, position, target):
        # we only rotate z and y axis

        forward = target - position
        forward = forward / np.linalg.norm(forward)
        x, y, z = forward[0], forward[1], forward[2]
        
        yaw = math.atan2(y, x)
        
        pitch = -math.atan2(z, math.sqrt(x*x + y*y) )

        # print("pitch: ", pitch)
        Rx = t3d.euler.euler2mat(0, 0, 0)
        Ry = t3d.euler.euler2mat(0, pitch, 0)
        Rz = t3d.euler.euler2mat(0, 0, yaw)
        R = Rz @ Ry
        # R = Ry 
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()
        trans = np.eye(4)
        trans[:3,:3] = R
        trans[0:3,3] = position
        return trans, quat
    
    def look_at_callback(self, ps_msg):

        target = np.array( [ps_msg.point.x, ps_msg.point.y, ps_msg.point.z] ) # in body frame
        position = np.array( [0.5, 0, 0.5] ) # in body frame
        goal, quat = self.get_lookat_position( position, target)
        pose_msg = PoseStamped()
        
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        timestamp = self.get_clock().now().to_msg()
        
        pose_msg.header.stamp=timestamp
        pose_msg.header.frame_id="body"
        
        self.arm_pose_cmd_callback(pose_msg)

        # move to goal 
    def gaze(self):
        

        # self.spot_wrapper is robot in examples
        # self._robot_command_client

        robot_state = self.spot_wrapper._robot_state_client.get_robot_state()
        # print("robot_state: ", robot_state)
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                         ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        # Look at a point 3 meters in front and 4 meters to the left.
        # We are not specifying a hand location, the robot will pick one.
        gaze_target_in_odom = odom_T_flat_body.transform_point(x=3.0, y=4.0, z=0)

        gaze_command = RobotCommandBuilder.arm_gaze_command(gaze_target_in_odom[0],
                                                            gaze_target_in_odom[1],
                                                            gaze_target_in_odom[2], ODOM_FRAME_NAME)
        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()

        gcode_origin_T_walk = SE3Pose(0.5, 0., 0., Quat(1, 0, 0, 0))

        odom_T_walk = gcode_origin_T_walk

        odom_T_walk_se2 = SE2Pose.flatten(odom_T_walk)
        # mobility_command = RobotCommandBuilder.synchro_se2_trajectory_command(odom_T_walk_se2.to_proto(), frame_name='odom')

        # Combine the arm and gripper commands into one RobotCommand
        # synchro_command = RobotCommandBuilder.build_synchro_command(mobility_command, gripper_command, gaze_command)
        synchro_command = RobotCommandBuilder.build_synchro_command(gripper_command, gaze_command)
        # Send the request
        # robot.logger.info('Requesting gaze.')
        gaze_command_id = self.spot_wrapper._robot_command_client.robot_command(synchro_command)

        block_until_arm_arrives(self.spot_wrapper._robot_command_client, gaze_command_id, 4.0)
    
    def walk(self) -> None:
        # self._logger.info("Walking forward")
        world_t_robot = self._tf_listener.lookup_a_tform_b(self._vision_frame_name, self._body_frame_name)
        world_t_robot_se2 = SE3Pose(
            world_t_robot.transform.translation.x,
            world_t_robot.transform.translation.y,
            world_t_robot.transform.translation.z,
            Quat(
                world_t_robot.transform.rotation.w,
                world_t_robot.transform.rotation.x,
                world_t_robot.transform.rotation.y,
                world_t_robot.transform.rotation.z,
            ),
        ).get_closest_se2_transform()

        ROBOT_T_GOAL = SE2Pose(0.5, 0.0, 0.0)
        world_t_goal = world_t_robot_se2 * ROBOT_T_GOAL
        proto_goal = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=world_t_goal.x,
            goal_y=world_t_goal.y,
            goal_heading=world_t_goal.angle,
            frame_name=VISION_FRAME_NAME,  # use Boston Dynamics' frame conventions
        )



# need to use customized frame in the future
# https://github.com/bdaiinstitute/spot_ros2/commit/6410c75e43a439cd22e3f51b60f7710c8338dfc2


@ros_process.main(prebaked=False)
def main(args: Optional[List[str]] = None) -> None:
    ros_process.spin(SpotROS_DTC)


if __name__ == "__main__":
    main()
