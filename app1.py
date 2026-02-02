import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import streamlit as st
from PIL import Image

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
    """왜곡 없이 폭만 맞춤(비율 유지). 크롭/보정 없음."""
    w, h = img.size
    if w == width:
        return img
    new_h = int(round(h * (width / w)))
    return img.resize((width, new_h), Image.LANCZOS)


# -------------------------
# Spacing analysis (reference)
# -------------------------
@dataclass
class SpacingRule:
    top_px: int
    between_px: int
    bottom_px: int


def analyze_spacing_from_reference(ref_img: Image.Image) -> SpacingRule:
    """레퍼런스 상세페이지 이미지에서 흰 여백 띠를 찾아 여백(px) 추정."""
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

    # fallback: 폭 비율 기본값(샘플 기반)
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
# Build detail image
# -------------------------
def build_detail_image(images: List[Image.Image], spacing: SpacingRule) -> Image.Image:
    resized = [resize_to_width(im, TARGET_W) for im in images]
    heights = [im.size[1] for im in resized]

    total_h = spacing.top_px + spacing.bottom_px
    if resized:
        total_h += sum(heights)
        total_h += spacing.between_px * (len(resized) - 1)

    canvas = Image.new("RGB", (TARGET_W, total_h), (255, 255, 255))
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
# Session State  (IMPORTANT: avoid key name 'items')
# -------------------------
def ensure_state():
    if "img_items" not in st.session_state:
        # list of dict {id, name, img}
        st.session_state["img_items"] = []
    if "img_include" not in st.session_state:
        # dict: id -> bool
        st.session_state["img_include"] = {}


def reset_items():
    st.session_state["img_items"] = []
    st.session_state["img_include"] = {}


def add_item(name: str, img: Image.Image):
    uid = f"{len(st.session_state['img_items'])+1:04d}_{safe_base(name)}"
    st.session_state["img_items"].append({"id": uid, "name": name, "img": img})
    st.session_state["img_include"][uid] = True


def move_item(index: int, direction: int):
    """direction: -1 (up), +1 (down)"""
    items = st.session_state["img_items"]
    j = index + direction
    if j < 0 or j >= len(items):
        return
    items[index], items[j] = items[j], items[index]


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="미샵 상세페이지 생성기", layout="wide")
st.title("상세페이지 생성기")
st.caption("폭 900 고정 · 업로드 순서 배열 · 여백 자동 · 크롭/보정/왜곡 금지")

ensure_state()

# ---- Sidebar: Settings ----
with st.sidebar:
    st.header("설정")

    ref = st.file_uploader(
        "레퍼런스 상세페이지 이미지(선택)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
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
        st.info("기본 여백 적용")

    use_manual = st.toggle("여백 수동 조정", value=False)
    if use_manual:
        top_px = st.slider("상단(px)", 0, 600, auto.top_px, 5)
        between_px = st.slider("사이(px)", 0, 900, auto.between_px, 5)
        bottom_px = st.slider("하단(px)", 0, 600, auto.bottom_px, 5)
        spacing = SpacingRule(top_px=top_px, between_px=between_px, bottom_px=bottom_px)
    else:
        spacing = auto

    st.divider()
    zip_sort_label = st.radio(
        "ZIP 이미지 순서",
        ["ZIP 내부 순서(권장)", "파일명 기준 정렬"],
        index=0,
    )
    zip_sort_mode = "zip_order" if "ZIP 내부" in zip_sort_label else "filename"

    st.divider()
    if st.button("목록 초기화", use_container_width=True):
        reset_items()
        st.rerun()

# ---- Main: Upload -> Arrange -> Generate ----
tab1, tab2 = st.tabs(["업로드", "순서/제외 & 생성"])

with tab1:
    st.subheader("1) 업로드")
    mode = st.radio("업로드 방식", ["JPG 여러 장", "ZIP 파일", "JPG + ZIP 혼합"], horizontal=True)

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

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("업로드 반영하기", type="primary", use_container_width=True):
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

            if new_items:
                for fn, img in new_items:
                    add_item(fn, img)
                st.success(f"추가 완료: {len(new_items)}장")
                st.rerun()
            else:
                st.warning("추가할 이미지가 없습니다.")

    with colB:
        st.metric("현재 업로드", f"{len(st.session_state['img_items'])} 장")

    if st.session_state["img_items"]:
        st.divider()
        st.subheader("업로드 미리보기 (최대 12장)")
        preview_count = min(12, len(st.session_state["img_items"]))
        previews = [resize_to_width(it["img"], 300) for it in st.session_state["img_items"][:preview_count]]
        st.image(previews, width=140)
    else:
        st.info("이미지를 업로드하고 ‘업로드 반영하기’를 눌러주세요.")

with tab2:
    st.subheader("2) 순서 변경 / 제외 / 생성")

    items = st.session_state["img_items"]
    include = st.session_state["img_include"]

    if not items:
        st.info("먼저 업로드 탭에서 이미지를 추가하세요.")
    else:
        st.caption("⬆⬇ 버튼으로 순서를 조정하고, 체크를 끄면 해당 컷은 제외됩니다.")

        top_row = st.columns([1, 1, 1])
        with top_row[0]:
            if st.button("전체 포함", use_container_width=True):
                for it in items:
                    include[it["id"]] = True
                st.rerun()
        with top_row[1]:
            if st.button("전체 제외", use_container_width=True):
                for it in items:
                    include[it["id"]] = False
                st.rerun()
        with top_row[2]:
            st.write(f"포함: **{sum(1 for it in items if include.get(it['id'], True))} / {len(items)}**")

        st.divider()

        for idx, it in enumerate(items):
            uid = it["id"]
            name = it["name"]
            inc = include.get(uid, True)

            row = st.columns([0.14, 0.56, 0.15, 0.15])
            with row[0]:
                prev = resize_to_width(it["img"], 160)
                st.image(prev, use_container_width=True)
            with row[1]:
                st.write(f"**{idx+1:02d}. {name}**")
                st.caption(f"{it['img'].size[0]}×{it['img'].size[1]}")
                st.checkbox("포함", value=inc, key=f"inc_{uid}")
                include[uid] = st.session_state[f"inc_{uid}"]
            with row[2]:
                st.write("")
                if st.button("⬆ 위로", key=f"up_{uid}", use_container_width=True, disabled=(idx == 0)):
                    move_item(idx, -1)
                    st.rerun()
                if st.button("⬇ 아래로", key=f"down_{uid}", use_container_width=True, disabled=(idx == len(items) - 1)):
                    move_item(idx, +1)
                    st.rerun()
            with row[3]:
                st.write("")
                if st.button("삭제", key=f"del_{uid}", use_container_width=True):
                    st.session_state["img_items"] = [x for x in items if x["id"] != uid]
                    if uid in include:
                        del include[uid]
                    st.rerun()

            st.markdown("---")

        selected = [it for it in st.session_state["img_items"] if include.get(it["id"], True)]

        st.subheader("3) 다운로드")
        st.write(f"- 폭: **{TARGET_W}px**")
        st.write(f"- 여백: 상단 {spacing.top_px}px / 사이 {spacing.between_px}px / 하단 {spacing.bottom_px}px")

        if not selected:
            st.warning("포함된 이미지가 없습니다. 체크를 켜주세요.")
        else:
            base_name = safe_base(selected[0]["name"])
            out_name = f"{base_name}_detail_900.jpg"

            if st.button("상세페이지 이미지 생성", type="primary", use_container_width=True):
                with st.spinner("생성 중…"):
                    imgs = [it["img"] for it in selected]
                    detail = build_detail_image(imgs, spacing=spacing)

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
