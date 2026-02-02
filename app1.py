import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_sortable import sortable

TARGET_W = 900  # 고정 폭


# -------------------------
# Helpers
# -------------------------
def safe_base(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"[^\w\-.가-힣]+", "_", base).strip("_")
    return base[:120] if base else "detail"


def is_image_file(fn: str) -> bool:
    fn = fn.lower()
    return fn.endswith((".jpg", ".jpeg", ".png", ".webp"))


def pil_open_rgb(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def resize_to_width(img: Image.Image, width: int) -> Image.Image:
    """왜곡 없이 폭만 맞춤(비율 유지)."""
    w, h = img.size
    if w == width:
        return img
    new_h = int(round(h * (width / w)))
    return img.resize((width, new_h), Image.LANCZOS)


# -------------------------
# Spacing analysis (reference image)
# -------------------------
@dataclass
class SpacingRule:
    top_px: int
    between_px: int
    bottom_px: int


def analyze_spacing_from_reference(ref_img: Image.Image) -> SpacingRule:
    img = ref_img.convert("RGB")
    arr = np.array(img).astype(np.uint8)
    gray = arr.mean(axis=2)

    white_thr = 245
    ratio_thr = 0.98
    row_white_ratio = (gray > white_thr).mean(axis=1)

    runs = []
    in_run = False
    start = 0
    for i, r in enumerate(row_white_ratio):
        if r >= ratio_thr:
            if not in_run:
                in_run = True
                start = i
        else:
            if in_run:
                length = i - start
                if length >= 5:
                    runs.append((start, i - 1, length))
                in_run = False
    if in_run:
        length = len(row_white_ratio) - start
        if length >= 5:
            runs.append((start, len(row_white_ratio) - 1, length))

    # fallback: 폭 비율 기본값
    if len(runs) < 2:
        w = img.size[0]
        return SpacingRule(
            top_px=max(10, int(round(w * 0.25))),
            between_px=max(10, int(round(w * 0.47))),
            bottom_px=max(10, int(round(w * 0.19))),
        )

    top = runs[0][2]
    bottom = runs[-1][2]
    mids = [r[2] for r in runs[1:-1]] or [int(round((top + bottom) / 2))]
    between = int(round(float(np.median(mids))))

    return SpacingRule(top_px=top, between_px=between, bottom_px=bottom)


def scale_spacing_to_target(rule: SpacingRule, ref_width: int, target_width: int) -> SpacingRule:
    if ref_width <= 0:
        return rule
    s = target_width / ref_width
    return SpacingRule(
        top_px=max(0, int(round(rule.top_px * s))),
        between_px=max(0, int(round(rule.between_px * s))),
        bottom_px=max(0, int(round(rule.bottom_px * s))),
    )


# -------------------------
# Build detail image (NO crop / NO enhance)
# -------------------------
def build_detail_image(
    images: List[Image.Image],
    spacing: SpacingRule,
    out_width: int = TARGET_W,
    background=(255, 255, 255),
) -> Image.Image:
    resized = [resize_to_width(im, out_width) for im in images]
    heights = [im.size[1] for im in resized]

    total_h = spacing.top_px + spacing.bottom_px
    if resized:
        total_h += sum(heights)
        total_h += spacing.between_px * (len(resized) - 1)

    canvas = Image.new("RGB", (out_width, total_h), background)

    y = spacing.top_px
    for i, im in enumerate(resized):
        canvas.paste(im, (0, y))
        y += im.size[1]
        if i != len(resized) - 1:
            y += spacing.between_px

    return canvas


# -------------------------
# ZIP handling
# -------------------------
def read_images_from_zip(uploaded_zip, sort_mode: str) -> List[Tuple[str, Image.Image]]:
    zbytes = uploaded_zip.read()
    zf = zipfile.ZipFile(io.BytesIO(zbytes))

    infos = [i for i in zf.infolist() if (not i.is_dir()) and is_image_file(i.filename)]
    if sort_mode == "filename":
        infos.sort(key=lambda x: os.path.basename(x.filename).lower())

    items = []
    for info in infos:
        fn = os.path.basename(info.filename)
        try:
            data = zf.read(info)
            im = pil_open_rgb(data)
            items.append((fn, im))
        except Exception:
            continue

    return items


# -------------------------
# Session state helpers
# -------------------------
def ensure_state():
    if "items" not in st.session_state:
        # items: list of dict {id, name, img}
        st.session_state.items = []
    if "include" not in st.session_state:
        # include[id] = bool
        st.session_state.include = {}
    if "order" not in st.session_state:
        # order: list of ids
        st.session_state.order = []


def reset_items():
    st.session_state.items = []
    st.session_state.include = {}
    st.session_state.order = []


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="미샵 상세페이지 생성기", layout="wide")
st.title("상세페이지 생성기")
st.caption("업로드한 이미지를 세로로 배열해 상세페이지용 JPG 1장 생성 (폭 900 / 여백 자동 / 크롭·보정 금지)")

ensure_state()

# Sidebar (minimal)
with st.sidebar:
    st.header("설정")

    st.subheader("여백 기준(레퍼런스)")
    ref = st.file_uploader(
        "레퍼런스 상세페이지 이미지(선택)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        key="ref",
    )

    if ref:
        ref_img = Image.open(ref).convert("RGB")
        raw = analyze_spacing_from_reference(ref_img)
        auto = scale_spacing_to_target(raw, ref_img.size[0], TARGET_W)
        st.success("레퍼런스 분석 완료")
    else:
        auto = SpacingRule(
            top_px=int(round(TARGET_W * 0.25)),
            between_px=int(round(TARGET_W * 0.47)),
            bottom_px=int(round(TARGET_W * 0.19)),
        )
        st.info("레퍼런스 미등록 시 기본 여백 적용")

    st.subheader("여백 미세조정")
    use_manual = st.toggle("수동 조정", value=False)
    if use_manual:
        top_px = st.slider("상단(px)", 0, 600, auto.top_px, 5)
        between_px = st.slider("사이(px)", 0, 900, auto.between_px, 5)
        bottom_px = st.slider("하단(px)", 0, 600, auto.bottom_px, 5)
        spacing = SpacingRule(top_px=top_px, between_px=between_px, bottom_px=bottom_px)
    else:
        spacing = auto

    st.divider()
    st.subheader("ZIP 정렬")
    zip_sort_label = st.radio(
        "ZIP 이미지 순서",
        ["ZIP 내부 순서(권장)", "파일명 기준 정렬"],
        index=0,
    )
    zip_sort_mode = "zip_order" if "ZIP 내부" in zip_sort_label else "filename"

    st.divider()
    if st.button("업로드 목록 초기화", use_container_width=True):
        reset_items()
        st.rerun()


# Main layout
left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.subheader("1) 업로드")
    mode = st.radio("업로드 방식", ["JPG 여러 장", "ZIP 파일", "JPG + ZIP 혼합"], horizontal=True)

    add_btn = st.button("업로드 반영하기", type="primary")
    st.caption("※ 이미지를 올린 뒤 ‘업로드 반영하기’를 눌러 목록에 추가하세요.")

    jpgs = None
    zips = None
    if mode in ["JPG 여러 장", "JPG + ZIP 혼합"]:
        jpgs = st.file_uploader(
            "이미지 업로드(JPG/PNG/WebP)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="jpgs",
        )

    if mode in ["ZIP 파일", "JPG + ZIP 혼합"]:
        zips = st.file_uploader(
            "ZIP 업로드(자동 압축 해제)",
            type=["zip"],
            accept_multiple_files=True,
            key="zips",
        )

    if add_btn:
        new_items = []

        if jpgs:
            for f in jpgs:
                try:
                    img = Image.open(f).convert("RGB")
                    new_items.append((f.name, img))
                except Exception:
                    continue

        if zips:
            for z in zips:
                new_items.extend(read_images_from_zip(z, sort_mode=zip_sort_mode))

        # append into session
        if new_items:
            for fn, img in new_items:
                # unique id
                uid = f"{len(st.session_state.items)+1:04d}_{safe_base(fn)}"
                st.session_state.items.append({"id": uid, "name": fn, "img": img})
                st.session_state.include[uid] = True
                st.session_state.order.append(uid)

            st.success(f"추가 완료: {len(new_items)}장")
            st.rerun()
        else:
            st.warning("추가할 이미지가 없습니다. 업로드 후 다시 눌러주세요.")

    if st.session_state.items:
        st.divider()
        st.subheader("2) 순서 변경 + 제외 선택")

        # ---- Drag sortable list ----
        # show labels with hidden id
        id_to_name = {it["id"]: it["name"] for it in st.session_state.items}
        current_ids = st.session_state.order[:]

        labels = [f"{i+1:02d}. {id_to_name[_id]}  ⟪{_id}⟫" for i, _id in enumerate(current_ids)]
        reordered_labels = sortable(labels, direction="vertical", key="sortable_list")

        # parse ids back
        new_order = []
        for lb in reordered_labels:
            m = re.search(r"⟪(.+?)⟫$", lb)
            if m:
                new_order.append(m.group(1))

        # if changed, update
        if new_order and new_order != st.session_state.order:
            st.session_state.order = new_order

        # ---- Exclude checkboxes with thumbnail grid ----
        st.caption("아래에서 특정 컷은 체크 해제하면 제외됩니다.")
        cols = st.columns(4)
        # build id->item
        id_to_item = {it["id"]: it for it in st.session_state.items}

        for idx, _id in enumerate(st.session_state.order):
            it = id_to_item[_id]
            with cols[idx % 4]:
                # small preview
                prev = resize_to_width(it["img"], 220)
                st.image(prev, use_container_width=True)
                st.checkbox(
                    f"포함 ({it['name']})",
                    value=st.session_state.include.get(_id, True),
                    key=f"inc_{_id}",
                    on_change=lambda _id=_id: st.session_state.include.__setitem__(_id, st.session_state[f"inc_{_id}"]),
                )

        included_count = sum(1 for _id in st.session_state.order if st.session_state.include.get(_id, True))
        st.info(f"현재 포함: {included_count}장 / 전체: {len(st.session_state.order)}장")

    else:
        st.info("아직 업로드된 이미지가 없습니다.")

with right:
    st.subheader("3) 생성 & 다운로드")
    st.write("**출력 규칙**")
    st.write(f"- 폭: **{TARGET_W}px 고정**")
    st.write("- 크롭/보정/왜곡: **없음** (비율 유지 리사이즈만)")
    st.write(f"- 여백: 상단 {spacing.top_px}px / 사이 {spacing.between_px}px / 하단 {spacing.bottom_px}px")

    if not st.session_state.items:
        st.info("왼쪽에서 이미지를 업로드하세요.")
    else:
        id_to_item = {it["id"]: it for it in st.session_state.items}
        selected_ids = [_id for _id in st.session_state.order if st.session_state.include.get(_id, True)]

        if not selected_ids:
            st.warning("포함된 이미지가 없습니다. 체크를 켜주세요.")
        else:
            base_name = safe_base(id_to_item[selected_ids[0]]["name"])
            out_name = f"{base_name}_detail_900.jpg"

            if st.button("상세페이지 이미지 생성", type="primary", use_container_width=True):
                with st.spinner("생성 중…"):
                    imgs = [id_to_item[_id]["img"] for _id in selected_ids]
                    detail = build_detail_image(imgs, spacing=spacing, out_width=TARGET_W)

                st.success("생성 완료!")
                st.write(f"결과 크기: **{detail.size[0]} × {detail.size[1]} px**")
                st.image(detail, caption="미리보기(축소)", width=320)

                buf = io.BytesIO()
                detail.save(buf, format="JPEG", quality=95)
                buf.seek(0)

                st.download_button(
                    "JPG 다운로드",
                    data=buf,
                    file_name=out_name,
                    mime="image/jpeg",
                    use_container_width=True,
                )

                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr(out_name, buf.getvalue())
                zbuf.seek(0)

                st.download_button(
                    "ZIP으로 다운로드",
                    data=zbuf,
                    file_name=f"{base_name}_outputs.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
