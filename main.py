from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from player_names import get_player_name  # Import player names


def main():
    video_frames = read_video('/Users/macbook/Desktop/Football_Analysis_CV/detect/veo_field_video.mp4')

    tracker = Tracker('/Users/macbook/Desktop/Football_Analysis_CV/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssigner()

    # ✅ Check if players exist before assigning team colors
    if len(tracks["players"]) > 0 and len(tracks["players"][0]) > 0:
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    else:
        print("⚠️ Warning: No players detected in the first frame. Skipping team assignment.")

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            tracks['players'][frame_num][player_id]['player_name'] = get_player_name(player_id)

    player_assigner = PlayerBallAssigner()


    team_ball_control = []  # Initialize list

    for frame_num, player_track in enumerate(tracks['players']):
        if frame_num >= len(tracks['ball']) or 1 not in tracks['ball'][frame_num]:
            # If ball data is missing for this frame, continue safely
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            continue

        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # Append last known ball control team if available, else default to 0 (neutral)
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)

    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    save_video(output_video_frames, 'output_videos/output_video.avi')
    print("Output saved successfully.")

    



if __name__ == '__main__':
    main()

