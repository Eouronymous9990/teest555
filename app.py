import streamlit as st
import cv2
from PIL import Image
import os
from analyzer import analyze_videos
error test

st.set_page_config(page_title="Tactiq", layout="wide", initial_sidebar_state="collapsed", page_icon="icon.png")

st.title("Football Analysis")
st.markdown("Analyze football techniques with biomechanical insights from front and side view videos")

col1, col2 = st.columns(2)
with col1:
    front_video = st.file_uploader("Upload Front View Video", type=["mp4"])
with col2:
    side_video = st.file_uploader("Upload Side View Video", type=["mp4"])

if st.button("Analyze", key="analyze_button"):
    if front_video and side_video:
        with st.spinner("Processing videos... This may take a few minutes."):
            front_path = "temp_front.mp4"
            side_path = "temp_side.mp4"
            with open(front_path, "wb") as f:
                f.write(front_video.read())
            with open(side_path, "wb") as f:
                f.write(side_video.read())
            try:
                results = analyze_videos(front_path, side_path)
                for idx, result in enumerate(results, 1):
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.header(f"Detection {idx}")
                    analysis = result["analysis"]
                    
                    # Basic Information (No Evaluation)
                    st.subheader("Basic Information")
                    basic_info = [
                        {"Metric": "Receiving Foot", "Value": analysis["receiving_foot"].capitalize() if analysis["receiving_foot"] else "Not Detected"},
                        {"Metric": "Head Orientation", "Value": analysis["head_direction"] if analysis["head_direction"] else "Not Detected"}
                    ]
                    st.table(basic_info)
                    
                    # Front View Analysis
                    st.subheader("Front View Biomechanical Analysis")
                    front_metrics = []
                    
                    # Supporting Foot-Ball Distance (using heel_ball percentages)
                    if analysis.get("heel_ball_percentage_y") is not None:
                        front_metrics.append({
                            "Metric": "Heel-Ball Distance Y",
                            "Value": f"{analysis['heel_ball_percentage_y']:.1f}%"
                        })
                    
                    if analysis.get("heel_ball_percentage_x") is not None:
                        front_metrics.append({
                            "Metric": "Heel-Ball Distance X",
                            "Value": f"{analysis['heel_ball_percentage_x']:.1f}%"
                        })
                    
                    # Ankle-Ball Distance in cm
                    front_metrics.append({
                        "Metric": "Ankle-Ball Distance (cm)",
                        "Value": f"{analysis['ankle_ball_dist_cm']:.1f} cm" if analysis.get('ankle_ball_dist_cm') is not None else "N/A"
                    })
                    
                    # Pelvis Angle
                    front_metrics.append({
                        "Metric": "Pelvis Angle",
                        "Value": f"{analysis['front_pelvis_angle']}°" if analysis["front_pelvis_angle"] is not None else "N/A"
                    })
                    
                    # Torso Angle
                    front_metrics.append({
                        "Metric": "Torso Angle",
                        "Value": f"{analysis['front_torso_angle']}°" if analysis["front_torso_angle"] is not None else "N/A"
                    })
                    
                    # Ankle Y Position
                    if analysis["normalized_receiving_ankle_position"] and analysis["normalized_receiving_ankle_position"].get("Y_position_percentage") is not None:
                        front_metrics.append({
                            "Metric": "Ankle Y Position (% of Body)",
                            "Value": f"{analysis['normalized_receiving_ankle_position']['Y_position_percentage']}%"
                        })
                    else:
                        front_metrics.append({
                            "Metric": "Ankle Y Position (% of Body)",
                            "Value": "N/A"
                        })
                    
                    # Ankle X Position (absolute value)
                    # Ankle X Position (% of Body) = abs(100 - X_percentage)
                    if analysis["normalized_receiving_ankle_position"] and analysis["normalized_receiving_ankle_position"].get("X_position_percentage") is not None:
                        x_pct = analysis['normalized_receiving_ankle_position']['X_position_percentage']
                        x_adjusted = abs(100 - x_pct)  # 👈 هذه هي الصيغة الصحيحة: المسافة من الطرف المقابل
                        front_metrics.append({
                            "Metric": "Ankle X Position (% of Body)",
                            "Value": f"{x_adjusted:.1f}%"
                        })
                    else:
                        front_metrics.append({
                            "Metric": "Ankle X Position (% of Body)",
                            "Value": "N/A"
                        })
                    
                    
                    st.table(front_metrics)
                    
                    # Display Ankle Values
                    st.markdown("**Ankle Values (for reference):**")
                    ankle_ref = []
                    if analysis.get("right_ankle_angle") is not None or analysis.get("right_ankle_distance") is not None:
                        ankle_ref.append({
                            "Side": "Right Ankle",
                            "Angle": f"{analysis['right_ankle_angle']}°" if analysis.get("right_ankle_angle") else "N/A",
                            "Distance (px)": f"{analysis['right_ankle_distance']} px" if analysis.get("right_ankle_distance") else "N/A"
                        })
                    if analysis.get("left_ankle_angle") is not None or analysis.get("left_ankle_distance") is not None:
                        ankle_ref.append({
                            "Side": "Left Ankle",
                            "Angle": f"{analysis['left_ankle_angle']}°" if analysis.get("left_ankle_angle") else "N/A",
                            "Distance (px)": f"{analysis['left_ankle_distance']} px" if analysis.get("left_ankle_distance") else "N/A"
                        })
                    if ankle_ref:
                        st.table(ankle_ref)
                    
                    # Side View Analysis
                    st.subheader("Side View Biomechanical Analysis")
                    side_metrics = []
                    
                    # Supporting Knee Angle
                    side_metrics.append({
                        "Metric": "Supporting Knee Angle",
                        "Value": f"{analysis['side_supporting_knee_angle']}°" if analysis.get("side_supporting_knee_angle") is not None else "N/A"
                    })
                    
                    # Side Torso Angle
                    side_metrics.append({
                        "Metric": "Torso Angle",
                        "Value": f"{analysis['side_torso_angle']}°" if analysis.get("side_torso_angle") is not None else "N/A"
                    })
                    
                    st.table(side_metrics)
                    
                    # Timestamp
                    st.markdown(f"**Analysis Timestamp:** {analysis['timestamp']}")
                    
                    # Display Images
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.subheader("Front View Frame")
                        front_img = Image.fromarray(cv2.cvtColor(result["front_frame"], cv2.COLOR_BGR2RGB))
                        st.image(front_img, use_container_width=True)
                    with col_img2:
                        st.subheader("Side View Frame")
                        side_img = Image.fromarray(cv2.cvtColor(result["side_frame"], cv2.COLOR_BGR2RGB))
                        st.image(side_img, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
            finally:
                if os.path.exists(front_path):
                    os.remove(front_path)
                if os.path.exists(side_path):
                    os.remove(side_path)
    else:
        st.warning("Please upload both front and side videos to proceed.")
