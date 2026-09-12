import sys
import os

# --- حل مشكلة basicsr مع تحديثات torchvision بدون التعديل على الملفات ---
import torchvision.transforms.functional as F

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    pass

if 'torchvision.transforms.functional_tensor' not in sys.modules:
    import types
    sys.modules['torchvision.transforms.functional_tensor'] = types.ModuleType('torchvision.transforms.functional_tensor')

sys.modules['torchvision.transforms.functional_tensor'].rgb_to_grayscale = F.rgb_to_grayscale
# ---------------------------------------------------------------------

import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import requests
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# إعداد واجهة Streamlit
st.set_page_config(page_title="تحسين جودة الصور", page_icon="🖼️", layout="centered")

st.title("🖼️ تحسين جودة الصور وإزالة النغمشة")
st.markdown("هذا التطبيق يستخدم **Real-ESRGAN** لتكبير حجم الصور وتحسين جودتها 4 أضعاف، بالإضافة إلى إزالة النغمشة (Noise) منها.")

@st.cache_resource
def load_model():
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    model_path = "RealESRGAN_x4plus.pth"
    
    if not os.path.exists(model_path):
        with st.spinner("جاري تحميل موديل الذكاء الاصطناعي (أول مرة فقط)..."):
            response = requests.get(model_url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=64, # تقليل أكثر للذاكرة (السيرفر المجاني إمكانياته محدودة)
        tile_pad=10,
        pre_pad=0,
        half=(device == "cuda")
    )
    return upsampler

try:
    upsampler = load_model()
except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل الموديل: {e}")
    st.stop()

uploaded_file = st.file_uploader("ارفع صورة هنا (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # نحول الألوان لعرضها بشكل صحيح في Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="الصورة الأصلية", use_container_width=True)
    
    if st.button("🚀 تحسين الجودة الآن"):
        with st.spinner("جاري المعالجة... قد يستغرق هذا دقيقة أو دقيقتين على السيرفر المجاني."):
            # إزالة النغمشة
            img_denoised = cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21)
            
            try:
                # الرفع x4
                output, _ = upsampler.enhance(img_denoised, outscale=4)
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                output_pil = Image.fromarray(output_rgb)
                
                st.success("تم تحسين الصورة بنجاح! ✅")
                st.image(output_rgb, caption="الصورة بعد التحسين (x4)", use_container_width=True)
                
                import io
                buf = io.BytesIO()
                output_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="💾 تحميل الصورة المُحسّنة",
                    data=byte_im,
                    file_name=f"enhanced_{uploaded_file.name}",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"خطأ أثناء معالجة الصورة. ربما حجم الصورة ضخم ويستهلك الذاكرة بالكامل: {e}")
