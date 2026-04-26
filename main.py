import os
import cv2
import numpy as np
import streamlit as st

TEMPLATES_FOLDER = 'templates'
st.set_page_config(page_title = 'Object Analyzer', layout = 'wide')

st.markdown('''
    <style>
    [data-testid='stImage'] img {
        max-height: 70vh;
        object-fit: contain;
    }
    </style>
    ''', unsafe_allow_html = True)

def get_features(img : np.ndarray, mask : np.ndarray) -> tuple:

    assert mask.dtype == bool

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(np.float32) / 255.0
    mask3 = mask[:, :, None]
    mean = (lab * mask3).sum(axis = (0, 1)) / (mask.sum() + 1e-10)
    lab = lab - mean

    M = cv2.moments(mask.astype(np.uint8) * 255)
    area =  mask.sum()
    radius = np.sqrt(area / (2 * np.sqrt(3))) * 0.95
    center = (int(M['m01'] / M['m00']), int(M['m10'] / M['m00']))

    vec_a = np.empty((10, ), dtype = np.float32)
    vec_b = np.empty((10, ), dtype = np.float32)

#    img = cv2.circle(img, center[::-1], int(radius), (255, 0, 0), -1)
#    cv2.imwrite('temp.jpg', img * mask[:, :, None])

    for i in range(10):
        r = int(radius * (i + 1) / 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
        center_crop = lab[center[0] - r : center[0] + r + 1, center[1] - r : center[1] + r + 1]
        a, b = (center_crop * kernel[:, :, None]).sum(axis = (0, 1))

        vec_a[i] = a
        vec_b[i] = b

    for i in reversed(range(10)):
        r1 = int(radius * (i + 0) / 10)
        r2 = int(radius * (i + 1) / 10)
        area = np.pi * ((r2 - r1) ** 2)

        if i == 0:
            vec_a[i] = vec_a[i] / area
            vec_b[i] = vec_b[i] / area
        else:
            vec_a[i] = (vec_a[i] - vec_a[i - 1]) / area
            vec_b[i] = (vec_b[i] - vec_b[i - 1]) / area

    features = np.concatenate((vec_a, vec_b))
    features /= np.linalg.norm(features) + 1e-10

    return features, center[::-1]

def erode_with_border_strategy(image, kernel_size=(5, 5), iterations=1):
    border_size = max(kernel_size) * iterations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    padded = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value = 0)
    eroded = cv2.erode(padded, kernel, iterations=iterations)
    return eroded[border_size:-border_size, border_size:-border_size]

def get_mask_with_steps(img):
    blur = cv2.medianBlur(img, 5)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_hull = np.zeros_like(gray)

    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100]
    if not areas: return edges, mask_hull, mask_hull

    med = np.median(areas)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100 and med * 0.8 < area < med * 1.2:
            hull = cv2.convexHull(contour)
            cv2.drawContours(mask_hull, [hull], -1, 255, thickness=cv2.FILLED)

    final_mask = erode_with_border_strategy(mask_hull, (5, 5))
    return edges, mask_hull, final_mask

@st.cache_data
def load_templates():
    templates = []
    if not os.path.exists(TEMPLATES_FOLDER):
        return templates
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    for file in os.listdir(TEMPLATES_FOLDER):
        if file.lower().endswith(valid_extensions):
            path = os.path.join(TEMPLATES_FOLDER, file)
            img = cv2.imread(path)
            if img is None: continue
            _, _, mask = get_mask_with_steps(img)
            features, _ = get_features(img, mask != 0)
            if features is not None:
                templates.append((file, features))
    return templates

st.title('👀 Поиск Тетрариума (Террариума?) (Тирамису?)')

templates = load_templates()

uploaded_file = st.sidebar.file_uploader('Загрузить изображение', type=['jpg', 'png', 'jpeg', 'bmp'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    if img_bgr is None:
        st.error('Не удалось прочитать файл. Возможно, формат поврежден.')
    else:
        edges, mask_hull, final_mask = get_mask_with_steps(img_bgr)
        count, labeled_mask = cv2.connectedComponents(final_mask)
        
        result_img = img_bgr.copy()
        
        for i in range(1, count):
            m = (labeled_mask == i)
            features, center = get_features(img_bgr, m)
            
            if features is not None:
                best = (None, -1e10)
                for temp_name, temp_feat in templates:
                    sim = np.dot(temp_feat, features)
                    score = (sim - 0.8) * 500
                    if score > best[1]:
                        best = (temp_name, score)
                
                t_name, t_score = best
                obj_id = os.path.splitext(t_name)[0] if t_name else '?'

                color = [int(x) for x in np.random.randint(100, 255, 3)]
                result_img[m] = result_img[m] * 0.5 + np.array(color) * 0.5

                label = f'{obj_id} [{t_score:.0f}%]'
                cv2.putText(result_img, label, (center[0] - 50, center[1]), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        st.subheader('Этапы обработки')
        cols = st.columns(4)
        titles = ['Оригинал', '1. Края (Canny)', '2. Контуры (Hull)', '3. Эрозия']
        imgs = [img_bgr, edges, mask_hull, final_mask]
        
        for col, title, im in zip(cols, titles, imgs):
            with col:
                st.caption(title)
                display_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if len(im.shape) == 3 else im
                st.image(display_im, use_container_width=True)

        st.divider()

        st.subheader(f'Финальный результат (Фишек: {count-1})')
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)

else:
    if not templates:
        st.warning("Папка 'templates' пуста. Добавьте туда фишки")
    st.info('Ожидание загрузки файла...')
