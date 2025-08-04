#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class SegmentationNode:
    def __init__(self):
        rospy.loginfo("🔁 Loading segmentation model...")
        model_path = "/home/mohit_iitian1/my_ws/src/segmentation_node/models/best_model.h5"
        self.model = load_model(model_path, compile=False)
        rospy.loginfo("✅ Model loaded successfully.")

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera_node/image_raw", Image, self.callback)

        self.mask_pub = rospy.Publisher("/seg/image_raw", Image, queue_size=1)
        self.overlay_pub = rospy.Publisher("/seg/image_color", Image, queue_size=1)

    def preprocess(self, image):
        resized = cv2.resize(image, (128, 128))
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=0)

    def postprocess(self, mask, original_shape):
        mask = (mask[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        return cv2.resize(mask, (original_shape[1], original_shape[0]))

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            input_tensor = self.preprocess(cv_image)

            prediction = self.model.predict(input_tensor)
            mask = self.postprocess(prediction, cv_image.shape)

            overlay = cv_image.copy()
            overlay[mask == 255] = [0, 0, 255]

            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")

            self.mask_pub.publish(mask_msg)
            self.overlay_pub.publish(overlay_msg)

            rospy.loginfo("📤 Published segmented image.")

        except Exception as e:
            rospy.logerr(f"❌ Error in callback: {e}")

if __name__ == '__main__':
    rospy.init_node('segmentation_node', anonymous=True)
    SegmentationNode()
    rospy.spin()
