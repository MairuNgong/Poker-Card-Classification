import cv2
import os
import numpy as np

# ตั้งค่าโฟลเดอร์ input ที่เก็บไฟล์วิดีโอ
# ตัวอย่าง: raw_video/club
VIDEO_INPUT_FOLDER = "Raw_Video_data\Dl_raw"

# ตั้งค่าโฟลเดอร์ output หลัก
ROOT_OUTPUT_FOLDER = "pokerdataset"

# นามสกุลไฟล์วิดีโอที่รองรับ
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.MOV', '.MP4')

def get_output_path(filename, root_output):
    """
    แปลงชื่อไฟล์เป็น output path
    เช่น: 2_diamond.mp4 -> pokerdataset/diamond/2
          ace_spade.mp4 -> pokerdataset/spade/ace
          2_diamond_phet.mp4 -> pokerdataset/diamond/2  (support suffix)
    """
    # ตัดนามสกุลออก เช่น "2_diamond_phet.mp4" -> "2_diamond_phet"
    name_without_ext = os.path.splitext(filename)[0]
    
    # แยกชื่อไฟล์ด้วย "_" เช่น "2_diamond_phet" -> ["2", "diamond", "phet"]
    parts = name_without_ext.split("_")
    
    # กำหนด suit ที่ถูกต้อง
    valid_suits = ['diamond', 'spade', 'heart', 'club']

    if len(parts) >= 2:
        # เช็คว่า part ที่ 2 เป็น suit หรือไม่ (parts[1])
        # เช่น ["2", "diamond", "phet"] -> parts[1] คือ "diamond"
        if parts[1] in valid_suits:
             card_value = parts[0]
             suit = parts[1]
             # ไม่สน suffix ข้างหลัง (เช่น phet) ในการสร้าง folder
        else:
            # Fallback for compatibility or unknown format
            card_value = parts[0]
            suit = "_".join(parts[1:])
        
        # สร้าง path: pokerdataset/diamond/2
        output_path = os.path.join(root_output, suit, card_value)
    else:
        # ถ้าชื่อไม่มี "_" ให้ใช้ชื่อเดิม
        output_path = os.path.join(root_output, name_without_ext)
    
    return output_path

def get_videos_from_folder(folder_path):
    """ดึงไฟล์วิดีโอทั้งหมดจาก folder"""
    videos = []
    if not os.path.exists(folder_path):
        print(f"[!] Folder not found: {folder_path}")
        return videos
    
    for filename in os.listdir(folder_path):
        if filename.endswith(VIDEO_EXTENSIONS):
            full_path = os.path.join(folder_path, filename)
            videos.append(full_path)
    
    return videos

def extract_frames(video_path, target_frames=100):
    filename = os.path.basename(video_path)
    card_name = os.path.splitext(filename)[0]
    target_dir = get_output_path(filename, ROOT_OUTPUT_FOLDER)
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # คำนวณรายชื่อเฟรมที่จะเอาไว้ล่วงหน้า (กระจายให้ทั่วทั้งวิดีโอ)
    # เช่น ถ้ามี 269 เฟรม จะเลือกมา 100 ตัวเลขที่กระจายกันตั้งแต่ 0 ถึง 268
    if total_frames >= target_frames:
        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    else:
        frame_indices = range(total_frames) # ถ้าวิดีโอสั้นกว่า 100 ก็เอาเท่าที่มี

    saved_count = 0
    print(f" -> Processing '{card_name}': {total_frames} frames found.")

    for frame_id in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        
        if not ret:
            # ถ้าอ่านพลาด ลองขยับถอยหลังมา 1 เฟรม (เผื่อเจอจุดสิ้นสุดไฟล์)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
            ret, frame = cap.read()
            if not ret: break

        save_name = f"{card_name}_{saved_count:03d}.jpg"
        cv2.imwrite(os.path.join(target_dir, save_name), frame)
        saved_count += 1

    cap.release()
    print(f" -> Done. Saved {saved_count} images to '{target_dir}'\n")

# --- ส่วนหลักของการทำงาน ---
if __name__ == "__main__":
    print("--- Poker Video to Image Dataset ---")
    print(f"Input folder: {VIDEO_INPUT_FOLDER}")
    print(f"Output folder: {ROOT_OUTPUT_FOLDER}")
    print("-" * 40)
    
    # 1. ดึงไฟล์วิดีโอทั้งหมดจาก folder
    video_files = get_videos_from_folder(VIDEO_INPUT_FOLDER)
    
    if not video_files:
        print("No video files found. Exiting.")
    else:
        print(f"Found {len(video_files)} video(s):")
        for v in video_files:
            print(f"  - {os.path.basename(v)}")
        print("-" * 40)
        
        # 2. วนลูปทำทีละไฟล์
        for vid in video_files:
            extract_frames(vid, target_frames=100)
            
        print("-" * 40)
        print(f"All Completed! Check your folder: {os.path.abspath(ROOT_OUTPUT_FOLDER)}")