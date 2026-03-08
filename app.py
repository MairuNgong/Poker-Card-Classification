import streamlit as st
from PIL import Image
import os
import time
import pandas as pd
import torch
from torchvision import transforms
import timm
from ultralytics import YOLO


st.set_page_config(
    page_title="Poker Card Classifier",
    page_icon="🃏",
    layout="centered" # เปลี่ยนเป็น centered เพื่อให้อ่านง่ายขึ้น ไม่กว้างเกินไป
)


SUIT_INFO = {
    "club": "ดอกจิก (Club) ♣️",
    "diamond": "ข้าวหลามตัด (Diamond) ♦️",
    "heart": "โพแดง (Heart) ♥️",
    "spade": "โพดำ (Spade) ♠️",
}

RANK_INFO = {
    "a": "A (Ace)", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9", "10": "10",
    "j": "J (Jack)", "q": "Q (Queen)", "k": "K (King)",
}

IMG_SIZE = 224


YOLO_SUIT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolomodel", "suit_yolo.pt")
YOLO_RANK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolomodel", "rank_yolo.pt")
EFF_SUIT_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "effmodel", "suit_eff.pth")
EFF_RANK_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "effmodel", "rank_eff.pth")


@st.cache_resource
def load_yolo_suit_model():
    return YOLO(YOLO_SUIT_MODEL_PATH, task="classify")

@st.cache_resource
def load_yolo_rank_model():
    return YOLO(YOLO_RANK_MODEL_PATH, task="classify")

@st.cache_resource
def load_eff_suit_model():
    device = torch.device('cpu')
    ckpt = torch.load(EFF_SUIT_MODEL_PATH, map_location=device)
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=len(ckpt["class_names"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["class_names"]

@st.cache_resource
def load_eff_rank_model():
    device = torch.device('cpu')
    ckpt = torch.load(EFF_RANK_MODEL_PATH, map_location=device)
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=len(ckpt["class_names"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["class_names"]


def predict_yolo(model, image):
    start = time.perf_counter()
    results = model(image, imgsz=224)
    elapsed = (time.perf_counter() - start) * 1000

    result = results[0]
    top1_idx = result.probs.top1
    top1_conf = result.probs.top1conf.item()
    predicted_class = result.names[top1_idx]
    
    all_probs = {result.names[i]: p for i, p in enumerate(result.probs.data.tolist())}
    return predicted_class, top1_conf, all_probs, elapsed

def predict_effnet(model, class_names, image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    elapsed = (time.perf_counter() - start) * 1000

    probs_list = probs.tolist()
    top1_conf = max(probs_list)
    predicted_class = class_names[probs_list.index(top1_conf)]
    
    all_probs = {class_names[i]: p for i, p in enumerate(probs_list)}
    return predicted_class, top1_conf, all_probs, elapsed


def render_prediction_block(predicted_class, conf, all_probs, elapsed, info_dict):
    display_name = info_dict.get(predicted_class, predicted_class.upper())
    
    st.metric(label="Predicted", value=display_name, delta=f"{conf*100:.1f}% Confidence", delta_color="normal")
    st.caption(f"Inference time: {elapsed:.1f} ms")
    
    with st.expander("รายละเอียดความน่าจะเป็น"):
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for cls_name, p in sorted_probs[:5]: # แสดงแค่ Top 5
            label = info_dict.get(cls_name, cls_name.upper())
            st.write(f"{label} ({p*100:.1f}%)")
            st.progress(float(p))


st.title("Poker Card Classifier")
st.write("เปรียบเทียบประสิทธิภาพการทำนายไพ่ระหว่างโมเดล **YOLO** และ **EfficientNetV2-S**")
st.divider()

uploaded_file = st.file_uploader("อัปโหลดรูปภาพไพ่ (JPG, PNG)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col_img1, col_img2, col_img3 = st.columns([1,2,1])
    with col_img2:
        st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)

    with st.spinner("กำลังประมวลผล..."):
        yolo_suit_model = load_yolo_suit_model()
        yolo_rank_model = load_yolo_rank_model()
        eff_suit_model, eff_suit_classes = load_eff_suit_model()
        eff_rank_model, eff_rank_classes = load_eff_rank_model()

        y_suit_cls, y_suit_conf, y_suit_probs, y_suit_time = predict_yolo(yolo_suit_model, image)
        y_rank_cls, y_rank_conf, y_rank_probs, y_rank_time = predict_yolo(yolo_rank_model, image)
        e_suit_cls, e_suit_conf, e_suit_probs, e_suit_time = predict_effnet(eff_suit_model, eff_suit_classes, image)
        e_rank_cls, e_rank_conf, e_rank_probs, e_rank_time = predict_effnet(eff_rank_model, eff_rank_classes, image)

    st.divider()

  
    st.subheader("ผลการทำนาย")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### YOLO")
        st.markdown("**ดอกไพ่ (Suit)**")
        render_prediction_block(y_suit_cls, y_suit_conf, y_suit_probs, y_suit_time, SUIT_INFO)
        st.markdown("**เลขไพ่ (Rank)**")
        render_prediction_block(y_rank_cls, y_rank_conf, y_rank_probs, y_rank_time, RANK_INFO)

    with col2:
        st.markdown("### EfficientNetV2-S")
        st.markdown("**ดอกไพ่ (Suit)**")
        render_prediction_block(e_suit_cls, e_suit_conf, e_suit_probs, e_suit_time, SUIT_INFO)
        st.markdown("**เลขไพ่ (Rank)**")
        render_prediction_block(e_rank_cls, e_rank_conf, e_rank_probs, e_rank_time, RANK_INFO)

    st.divider()


    st.subheader("ตารางสรุปการเปรียบเทียบ (Comparison Summary)")
    
    suit_match = "ตรงกัน" if y_suit_cls == e_suit_cls else "ไม่ตรงกัน"
    rank_match = "ตรงกัน" if y_rank_cls == e_rank_cls else "ไม่ตรงกัน"
    
    if suit_match == "ตรงกัน" and rank_match == "ตรงกัน":
        st.success("โมเดลทั้งสองทำนายผลตรงกันทั้งหมด")
    else:
        st.warning("มีผลการทำนายบางส่วนที่โมเดลทั้งสองให้ผลไม่ตรงกัน")

    df_summary = pd.DataFrame({
        "หัวข้อ": ["ทำนายดอกไพ่", "ความมั่นใจ (ดอก)", "ทำนายเลขไพ่", "ความมั่นใจ (เลข)", "เวลาประมวลผลรวม"],
        "YOLO": [
            SUIT_INFO.get(y_suit_cls, y_suit_cls),
            f"{y_suit_conf*100:.2f}%",
            RANK_INFO.get(y_rank_cls, y_rank_cls),
            f"{y_rank_conf*100:.2f}%",
            f"{(y_suit_time + y_rank_time):.1f} ms"
        ],
        "EfficientNetV2-S": [
            SUIT_INFO.get(e_suit_cls, e_suit_cls),
            f"{e_suit_conf*100:.2f}%",
            RANK_INFO.get(e_rank_cls, e_rank_cls),
            f"{e_rank_conf*100:.2f}%",
            f"{(e_suit_time + e_rank_time):.1f} ms"
        ]
    })
    
    df_summary.set_index("หัวข้อ", inplace=True)
    st.table(df_summary)

else:
    st.info("กรุณาอัปโหลดรูปภาพไพ่เพื่อเริ่มต้นการทดสอบโมเดล")