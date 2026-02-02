import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import List, Tuple

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
    """왜곡 없이 폭만 맞춤(비율 유지)."""
    w, h = img.size
    if w == width:
        return img
    new_h = int(round(h * (width / w)))
    return img.resize((width, new_h), Image.LANCZOS)


# -------------------------
# Spacing analysis from reference detail image
# (find horizontal white bands)
# -------------------------
@dataclass
class SpacingRule:
    top_px: int
    between_px: int
    bottom_px: int


def analyze_spacing_from_reference(ref_img: Image.Image) -> SpacingRule:
    """
    레퍼런스 상세페이지 이미지에서
    '거의 흰색' 가로 띠를 찾아
    상단/중간/하단 여백(px)을 추정.
    """
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

    # runs가 적으면 기본값 리턴
    if len(runs) < 2:
        # fallback: "폭 대비 비율" 기반 기본값(샘플 분석값 기반)
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
    """레퍼런스 폭 기준 px 여백을, 목표 폭(900)에 비례 스케일."""
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
def build_detail_image(
    images: List[Image.Image],
    spacing: SpacingRule,
    out_width: int = TARGET_W,
    background=(255, 255, 255),
) -> Image.Image:
    """
    업로드된 이미지를 순서대로:
    - 각 이미지를 폭 900에 맞춰 리사이즈(비율 유지)
    - 사이사이 흰 여백(between_px)
    - 최상단/최하단 흰 여백(top/bottom)
    """
    resized = [resize_to_width(im, out_width) for im in images]
    heights = [im.size[1] for im in resized]

    total_h = spacing.top_px + spacing.bottom_px
    if len(resized) > 0:
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
def read_images_from_zip(uploaded_zip) -> List[Tuple[str, Image.Image]]:
    """
    ZIP 안의 이미지를 'ZIP 내부 순서(ZipInfo 순서)'로 읽음.
    (업로드 순서 요구에 가장 근접)
    """
    zbytes = uploaded_zip.read()
    zf = zipfile.ZipFile(io.BytesIO(zbytes))

    items = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        fn = info.filename
        if not is_image_file(fn):
            continue
        data = zf.read(info)
        try:
            im = pil_open_rgb(data)
            items.append((os.path.basename(fn), im))
        except Exception:
            continue

    return items


# -------------------------
# UI
# -------------------------
st.set_page_config(layout="wide")
st.title(미샵 상세페이지 생성기)

st.markdown(
    """
- 업로드한 이미지를 **업로드 순서대로** 세로로 배열해 **상세페이지용 JPG 1장**을 만듭니다.  
- 각 이미지는 **폭 900으로 비율 유지 리사이즈만** 하며, **크롭/왜곡/보정은 하지 않습니다.**  
- 이미지 사이/상단/하단 흰 여백은 **레퍼런스 상세페이지 이미지**로부터 자동 분석합니다.
"""
)

colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.subheader("1) 여백 기준(레퍼런스) 설정")
    ref = st.file_uploader(
        "레퍼런스 상세페이지 이미지 업로드(권장) — 여백 자동 분석용",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        key="ref",
    )

    if ref:
        ref_img = Image.open(ref).convert("RGB")
        rule_raw = analyze_spacing_from_reference(ref_img)
        rule = scale_spacing_to_target(rule_raw, ref_img.size[0], TARGET_W)
        st.success("레퍼런스 분석 완료!")
        st.write(f"- (레퍼런스 폭 {ref_img.size[0]} 기준) 상단 {rule_raw.top_px}px / 사이 {rule_raw.between_px}px / 하단 {rule_raw.bottom_px}px")
        st.write(f"- (900폭 적용) 상단 {rule.top_px}px / 사이 {rule.between_px}px / 하단 {rule.bottom_px}px")
        st.image(ref_img, caption="레퍼런스 미리보기", width=220)
    else:
        # 샘플 분석 기반 기본 비율 (폭 대비)
        # top≈0.25W, between≈0.47W, bottom≈0.19W
        rule = SpacingRule(
            top_px=int(round(TARGET_W * 0.25)),
            between_px=int(round(TARGET_W * 0.47)),
            bottom_px=int(round(TARGET_W * 0.19)),
        )
        st.info("레퍼런스를 올리지 않으면, 기본 여백(샘플 기반 비율)로 적용합니다.")
        st.write(f"- 기본(900폭) 상단 {rule.top_px}px / 사이 {rule.between_px}px / 하단 {rule.bottom_px}px")

    st.divider()
    st.subheader("2) 입력 방식 선택")
    mode = st.radio("업로드 방식", ["JPG 여러 장 업로드", "ZIP 업로드(자동 압축해제)", "JPG + ZIP 혼합"], horizontal=False)

with colB:
    st.subheader("3) 이미지 업로드")
    uploaded_images: List[Tuple[str, Image.Image]] = []

    if mode in ["JPG 여러 장 업로드", "JPG + ZIP 혼합"]:
        jpgs = st.file_uploader(
            "JPG(또는 PNG/WebP) 여러 장 업로드 — 업로드 순서대로 정렬",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="jpgs",
        )
        if jpgs:
            for f in jpgs:
                try:
                    uploaded_images.append((f.name, Image.open(f).convert("RGB")))
                except Exception:
                    continue

    if mode in ["ZIP 업로드(자동 압축해제)", "JPG + ZIP 혼합"]:
        zips = st.file_uploader(
            "ZIP 업로드(여러 개 가능) — ZIP 내부 파일 순서대로 사용",
            type=["zip"],
            accept_multiple_files=True,
            key="zips",
        )
        if zips:
            for z in zips:
                for fn, im in read_images_from_zip(z):
                    uploaded_images.append((fn, im))

    if not uploaded_images:
        st.warning("이미지를 업로드하면 여기서 생성/다운로드가 활성화됩니다.")
    else:
        st.success(f"총 {len(uploaded_images)}장 업로드 완료")

        # 미리보기(업로드 순서)
        st.subheader("업로드 순서 미리보기(상위 12장)")
        previews = [resize_to_width(im, 300) for _, im in uploaded_images[:12]]
        st.image(previews, width=150)

        st.divider()
        st.subheader("4) 생성 및 다운로드")

        # 파일명 규칙: 업로드한 파일명을 이용
        # - ZIP만 업로드: zip 이름을 쓰고 싶지만, 혼합/다중 zip도 있어서
        #   기본은 첫 이미지 파일명 기반으로 생성
        base_name = safe_base(uploaded_images[0][0])
        out_name = f"{base_name}_detail_900.jpg"

        if st.button("상세페이지 이미지 생성", type="primary"):
            imgs_only = [im for _, im in uploaded_images]
            detail = build_detail_image(imgs_only, spacing=rule, out_width=TARGET_W)

            st.success("생성 완료!")
            st.write(f"- 결과 크기: {detail.size[0]} × {detail.size[1]} px")
            st.image(detail, caption="결과 미리보기(축소)", width=260)

            # 단일 JPG 다운로드
            buf = io.BytesIO()
            detail.save(buf, format="JPEG", quality=95)
            buf.seek(0)

            st.download_button(
                "JPG 다운로드",
                data=buf,
                file_name=out_name,
                mime="image/jpeg",
            )

            # 여러 작업을 대비해 "한꺼번에" ZIP 다운로드도 제공(현재는 1개지만 규격 맞춤)
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(out_name, buf.getvalue())
            zbuf.seek(0)

            st.download_button(
                "ZIP으로 한꺼번에 다운로드",
                data=zbuf,
                file_name=f"{base_name}_outputs.zip",
                mime="application/zip",
            )
