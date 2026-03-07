"""
สคริปต์สำหรับ apply transforms ให้กับภาพใน folder train
จะสร้างภาพที่ผ่านการ transform แล้วเก็บไว้ใน folder ใหม่
"""
import os
from PIL import Image
import cv2
import numpy as np
from augment import add_salt_and_pepper
from torchvision import transforms
import torch

# กำหนด paths
# ใช้ path หลักของ dataset (ตรวจสอบ path ให้ถูกต้องตามโครงสร้างจริงของคุณ)
# เช่น C:\Users\ssupa\Code\DL\pokerdataset_phet
dataset_root = r"C:\Users\ssupa\Code\DL\pokerdataset_phet\pokerdataset_phet"
output_root = r"C:\Users\ssupa\Code\DL\augmented_poker_data"

# สร้าง output folder ถ้ายังไม่มี
os.makedirs(output_root, exist_ok=True)

# สำหรับ Training (ปรับให้เหมาะกับไพ่โป๊กเกอร์ + YOLO Classification)
train_transform = transforms.Compose([
    # 1. ย่อรูปให้ใหญ่กว่า 224 เพื่อให้หมุน/เปลี่ยนมุมแล้วไม่เกิดมุมดำ
    transforms.Resize(280), 
    
    # 2. หมุน + ขยับ + ซูม โดยใช้สีขาว fill
    transforms.RandomAffine(
        degrees=15,          # หมุน ±15 องศา (ไพ่ไม่หลุดเฟรม)
        translate=(0.05, 0.05),  # ขยับนิดหน่อย
        scale=(0.9, 1.1),    # ซูมเข้าออกนิดหน่อย
        fill=255             # fill สีขาว (ไม่มีมุมดำ)
    ),
    
    # 3. จำลองมุมกล้องเอียง (ถ่ายไพ่จากมุมต่างๆ)
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3, fill=255),
    
    # 4. ตัดตรงกลางให้ได้ 224x224 (ตรงกับ YOLO classification input)
    transforms.CenterCrop(224),
    
    # 5. ปรับแสง สี ความสว่าง เพื่อจำลองสภาพแสงต่างๆ
    transforms.ColorJitter(
        brightness=0.2, 
        contrast=0.2, 
        saturation=0.2, 
        hue=0.1
    ),
    
    # 6. สุ่มเบลอเล็กน้อย จำลองกล้องไม่คมชัด
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
    # สุ่ม salt-and-pepper noise เล็กน้อย
    transforms.RandomApply([transforms.Lambda(lambda img: add_salt_and_pepper(np.array(img), prob=0.02))], p=0.2),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # ตัดตรงกลางเป๊ะๆ ให้ได้ 224x224
])

# Transform สำหรับบันทึก (ไม่รวม ToTensor และ Normalize เพราะจะบันทึกเป็นภาพ)
# ถ้าต้องการบันทึกเป็น tensor ค่อย apply ทีหลังตอน train

def save_transformed_image(img_tensor_or_pil, output_path):
    """บันทึกภาพที่ผ่าน transform แล้ว"""
    if isinstance(img_tensor_or_pil, torch.Tensor):
        # ถ้าเป็น tensor ก็แปลงกลับเป็น PIL
        img = transforms.ToPILImage()(img_tensor_or_pil)
    elif isinstance(img_tensor_or_pil, np.ndarray):
        # ถ้าเป็น numpy array (เช่นจาก cv2 หรือ salt_and_pepper func) แปลงเป็น PIL
        img = Image.fromarray(img_tensor_or_pil)
    else:
        img = img_tensor_or_pil
    img.save(output_path)

def apply_transforms_to_folder(input_dir, output_dir, transform, num_augments=3):
    """
    Apply transforms ให้กับทุกภาพใน folder
    num_augments: จำนวนครั้งที่จะสุ่ม augment ต่อ 1 ภาพ (เพื่อสร้างข้อมูลเพิ่ม)
    """
    # รับรายการไฟล์ภาพทั้งหมด
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(image_extensions)]
    
    print(f"พบภาพทั้งหมด {len(image_files)} ภาพ")
    print(f"กำลังสร้าง {num_augments} รูปต่อ 1 ภาพต้นฉบับ...")
    
    total_created = 0
    
    for idx, filename in enumerate(image_files):
        input_path = os.path.join(input_dir, filename)
        
        try:
            # เปิดภาพ
            img = Image.open(input_path).convert('RGB')
            
            # สร้าง augmented images หลายรูป
            for aug_idx in range(num_augments):
                # Apply transform (จะสุ่มใหม่ทุกครั้ง)
                transformed_img = transform(img)
                
                # สร้างชื่อไฟล์ใหม่
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_aug{aug_idx}{ext}"
                output_path = os.path.join(output_dir, new_filename)
                
                # บันทึก
                save_transformed_image(transformed_img, output_path)
                total_created += 1
                
            if (idx + 1) % 20 == 0:
                print(f"ประมวลผลแล้ว {idx + 1}/{len(image_files)} ภาพ")
                
        except Exception as e:
            print(f"เกิดข้อผิดพลาดกับไฟล์ {filename}: {e}")
    
    print(f"\n✅ เสร็จสิ้น! สร้างภาพทั้งหมด {total_created} ภาพ")
    print(f"📁 บันทึกไว้ที่: {output_dir}")

if __name__ == "__main__":
    print("=" * 50)
    if not os.path.exists(dataset_root):
        print(f"❌ ไม่พบโฟลเดอร์: {dataset_root}")
        exit()

    # วนลูปทุก suit และ rank
    for suit in os.listdir(dataset_root):
        suit_path = os.path.join(dataset_root, suit)
        if not os.path.isdir(suit_path):
            continue
            
        for rank in os.listdir(suit_path):
            rank_path = os.path.join(suit_path, rank)
            if not os.path.isdir(rank_path):
                continue

            # สร้าง path ปลายทางให้ตรงกับโครงสร้างเดิม
            target_dir = os.path.join(output_root, suit, rank)
            os.makedirs(target_dir, exist_ok=True)

            print(f"📂 กำลังประมวลผล: {suit}/{rank}")

            # Apply train transform
            apply_transforms_to_folder(
                rank_path, 
                target_dir, 
                train_transform,
                num_augments=3  # เปลี่ยนได้ตามต้องการ
            )
    
    print("\n✅ เสร็จสิ้นกระบวนการทั้งหมด!")
