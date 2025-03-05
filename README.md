# AI-Football-Analytics
Football is one of the most widely watched and analyzed sports globally, and the increasing demand for data-driven insights has led to significant advancements in AI-powered sports analytics. Traditionally, performance tracking in football has relied on manual tagging and GPS-based tracking systems, which are either time-consuming, expensive, or limited in accuracy.

This project introduces an AI-driven Football Analytics System that leverages computer vision and deep learning to track player movements, estimate speed and distance, and provide real-time insights into player performance. By integrating YOLO for object detection and ByteTrack for multi-object tracking, our model can efficiently distinguish and follow players across frames.

A unique aspect of our approach is the random assignment of well-known football player names to tracking IDs, demonstrating the model’s capability to differentiate players based on their movements and positioning. This enables a more intuitive representation of player tracking while ensuring consistent identity tracking throughout the match.

Furthermore, by incorporating view transformation techniques and camera movement adjustments, our system provides accurate movement metrics, ensuring that player statistics are computed relative to real-world positions. This AI-powered approach has significant implications for performance analysis, tactical evaluations, and sports broadcasting, making it a valuable tool for analysts, coaches, and enthusiasts.

The following sections detail our methodology, implementation, experimental setup, and results, highlighting the effectiveness of our system in real-time football analytics.
