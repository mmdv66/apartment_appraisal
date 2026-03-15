import pandas as pd
import ast
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW_CSV = ROOT / "data" / "raw" / "flats_data.csv"
OUT_CSV = ROOT / "data" / "dataset.csv"
OUT_IMAGES_CSV = ROOT / "data" / "dataset_images.csv"

df = pd.read_csv(RAW_CSV)
print(f"Загружено: {len(df)} объявлений")

# =====================
# 1. Дубликаты
# =====================
df = df.drop_duplicates(subset=["flat_id"]).copy()
print(f"После дедупликации: {len(df)}")

# =====================
# 2. Цена → число
# =====================
df["price_rub"] = (
    df["price"]
    .astype(str)
    .str.replace(r"[^\d]", "", regex=True)
    .pipe(pd.to_numeric, errors="coerce")
)
df = df[(df["price_rub"] >= 1_000_000) & (df["price_rub"] <= 2_000_000_000)]
print(f"После чистки цены: {len(df)}")

# =====================
# 3. Площади → float
# =====================
def clean_area(val):
    if pd.isna(val):
        return None
    return pd.to_numeric(
        re.sub(r"[^\d,\.]", "", str(val)).replace(",", "."),
        errors="coerce"
    )

for col in ["total_area", "living_area", "kitchen_area"]:
    if col in df.columns:
        df[col] = df[col].apply(clean_area)

if "total_area" in df.columns:
    df = df[(df["total_area"].isna()) | ((df["total_area"] >= 10) & (df["total_area"] <= 1000))]

# =====================
# 4. Этаж → два числа
# =====================
def parse_floor(val):
    if pd.isna(val):
        return None, None
    nums = re.findall(r"\d+", str(val))
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    elif len(nums) == 1:
        return int(nums[0]), None
    return None, None

if "floor" in df.columns:
    df[["floor_num", "total_floors"]] = df["floor"].apply(
        lambda x: pd.Series(parse_floor(x))
    )
    df = df.drop(columns=["floor"], errors="ignore")

# =====================
# 5. Год постройки → int
# =====================
if "build_year" in df.columns:
    df["build_year"] = pd.to_numeric(
        df["build_year"].astype(str).str.extract(r"(\d{4})")[0],
        errors="coerce"
    )
    df = df[(df["build_year"].isna()) | ((df["build_year"] >= 1900) & (df["build_year"] <= 2030))]

# =====================
# 6. Категории → числа
# =====================
if "building_type" in df.columns:
    df["building_type"] = df["building_type"].astype(str).str.strip()
    df["building_type_code"] = pd.Categorical(df["building_type"]).codes

if "renovation" in df.columns:
    df["renovation"] = df["renovation"].astype(str).str.strip()
    df["renovation_code"] = pd.Categorical(df["renovation"]).codes

if "rooms" in df.columns:
    df["rooms"] = pd.to_numeric(
        df["rooms"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )

# =====================
# 7. Описание (NLP)
# =====================
df = df[df["description"].notna()].copy()
df["description"] = df["description"].str.strip()
df = df[df["description"].str.len() > 20]
print(f"После чистки описания: {len(df)}")

# =====================
# 8. Фотки
# =====================
df["image_paths"] = df["image_paths"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) and x != "[]" else []
)
df["image_paths"] = df["image_paths"].apply(
    lambda paths: [p for p in paths if Path(p).exists()]
)
df = df[df["image_paths"].apply(len) > 0].copy()
print(f"После проверки фоток: {len(df)}")

# =====================
# 9. Итог
# =====================
print(f"\nИтого чистых объявлений: {len(df)}")
print(f"Колонки: {list(df.columns)}")
print(f"\nСтатистика:")
for col in ["price_rub", "total_area", "living_area", "kitchen_area", "floor_num", "build_year", "rooms"]:
    if col in df.columns:
        print(f"  {col}: min={df[col].min()}, max={df[col].max()}, nulls={df[col].isna().sum()}")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n>>> dataset.csv сохранён: {len(df)} строк")

# Датасет для CV (одна строка = одно фото)
df_images = df.explode("image_paths").rename(columns={"image_paths": "image_path"})
df_images = df_images[df_images["image_path"].notna()].copy()
df_images.to_csv(OUT_IMAGES_CSV, index=False, encoding="utf-8-sig")
print(f">>> dataset_images.csv сохранён: {len(df_images)} строк (фото)")