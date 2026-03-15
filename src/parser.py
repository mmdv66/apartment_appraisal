import asyncio
import os
import re
import httpx
import pandas as pd
from playwright.async_api import async_playwright
from tqdm import tqdm

DATA_DIR = "data/raw"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


async def download_images(flat_id: str, image_urls: list) -> list:
    saved_paths = []
    flat_img_dir = os.path.join(IMAGES_DIR, str(flat_id))
    os.makedirs(flat_img_dir, exist_ok=True)

    async with httpx.AsyncClient(timeout=15) as client:
        for i, url in enumerate(image_urls[:15]):
            try:
                r = await client.get(url, headers={"Referer": "https://www.cian.ru/"})
                ext = url.split(".")[-1].split("?")[0] or "jpg"
                path = os.path.join(flat_img_dir, f"{i}.{ext}")
                with open(path, "wb") as f:
                    f.write(r.content)
                saved_paths.append(path)
            except Exception as e:
                print(f"    [!] Фото {i}: {e}")
    return saved_paths


async def scrape_flat(page, url: str) -> dict | None:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)

        data = {"url": url}
        m = re.search(r"/(\d+)/?$", url)
        data["flat_id"] = m.group(1) if m else url.split("/")[-2]

        # --- Цена ---
        el = await page.query_selector("[data-name='PriceInfo']")
        data["price"] = (await el.inner_text()).strip() if el else None

        # --- Описание (NLP) ---
        el = await page.query_selector("[data-name='Description']")
        data["description"] = (await el.inner_text()).strip() if el else None

        # --- Характеристики (ObjectFactoids) ---
        factoids = {}
        items = await page.query_selector_all("[data-name='ObjectFactoids'] [class*='item']")
        for item in items:
            text = (await item.inner_text()).strip()
            parts = [p.strip() for p in text.split("\n") if p.strip()]
            if len(parts) >= 2:
                factoids[parts[0]] = parts[1]  # ключ=название, значение=данные

        # --- ML-фичи из factoids ---
        for search_key, field in [
            ("Общая", "total_area"),
            ("Жилая", "living_area"),
            ("Кухня", "kitchen_area"),
            ("Этаж", "floor"),
            ("Год постройки", "build_year"),
            ("Тип дома", "building_type"),
            ("Ремонт", "renovation"),
            ("Количество комнат", "rooms"),
        ]:
            for k, v in factoids.items():
                if search_key.lower() in k.lower():
                    data[field] = v
                    break

        # --- Адрес ---
        for sel in ["[data-name='Geo']", "[class*='geo']", "address"]:
            el = await page.query_selector(sel)
            if el:
                data["address"] = (await el.inner_text()).strip().replace("\n", ", ")
                break
        else:
            data["address"] = None

        # --- Картинки ---
        img_urls = []

        try:
            await page.wait_for_selector("[data-name='OfferGallery']", timeout=5000)
        except:
            pass

        await page.evaluate("window.scrollBy(0, 300)")
        await asyncio.sleep(1)

        imgs = await page.query_selector_all("[data-name='OfferGallery'] img")
        for img in imgs:
            src = await img.get_attribute("src") or await img.get_attribute("data-src")
            if src and src.startswith("http") and src not in img_urls:
                src = re.sub(r"/\d+x\d+/", "/", src)
                img_urls.append(src)

        if len(img_urls) <= 1:
            try:
                gallery = await page.query_selector("[data-name='OfferGallery']")
                if gallery:
                    await gallery.click()
                    await asyncio.sleep(2)

                    for sel in [
                        "img[class*='slide']",
                        "img[class*='fullscreen']",
                        "img[class*='gallery']",
                        "[data-name='OfferGallery'] img",
                    ]:
                        imgs = await page.query_selector_all(sel)
                        for img in imgs:
                            src = await img.get_attribute("src") or await img.get_attribute("data-src")
                            if src and src.startswith("http") and src not in img_urls:
                                src = re.sub(r"/\d+x\d+/", "/", src)
                                img_urls.append(src)
                        if len(img_urls) > 1:
                            break

                    for close_sel in ["button[class*='close']", "[data-name='CloseButton']", "button[aria-label*='закр']"]:
                        close = await page.query_selector(close_sel)
                        if close:
                            await close.click()
                            break
            except Exception as e:
                print(f"    [!] Галерея: {e}")

        data["image_urls"] = img_urls
        data["image_count"] = len(img_urls)
        data["image_paths"] = await download_images(data["flat_id"], img_urls) if img_urls else []

        print(f"  [+] {data.get('address', url[:50])} — {data.get('price')} | фото: {len(img_urls)} | площадь: {data.get('total_area')} | этаж: {data.get('floor')}")
        return data

    except Exception as e:
        print(f"  [!] Ошибка на {url}: {e}")
        return None


async def get_listing_urls(page, search_url: str, max_pages: int = 5) -> list:
    all_urls = []
    for page_num in range(1, max_pages + 1):
        paginated = f"{search_url}&p={page_num}"
        print(f"  Страница поиска {page_num}...")
        try:
            await page.goto(paginated, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(4)
            await page.wait_for_selector("a[href*='/sale/flat/']", timeout=20000)
            links = await page.query_selector_all("a[href*='/sale/flat/']")
            for link in links:
                href = await link.get_attribute("href")
                if href and re.search(r"/sale/flat/\d{7,}/", href):
                    full = href if href.startswith("http") else "https://www.cian.ru" + href
                    all_urls.append(full)
            print(f"    Найдено: {len(set(all_urls))} уник. ссылок")
        except Exception as e:
            print(f"  [!] Ошибка стр.{page_num}: {e}")
            break
    return list(set(all_urls))


async def main():
    SEARCH_URL = "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&region=1"
    MAX_SEARCH_PAGES = 30
    MAX_FLATS = None  # Ограничение на количество объявлений (None = без лимита)
    DOWNLOAD_IMAGES = True
    CONCURRENCY = 2

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="ru-RU",
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )

        search_page = await context.new_page()
        print(">>> Открываем ЦИАН (реши капчу если появится)...")
        await search_page.goto(SEARCH_URL, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(5)

        print(">>> Собираем ссылки...")
        urls = await get_listing_urls(search_page, SEARCH_URL, MAX_SEARCH_PAGES)
        await search_page.close()

        if MAX_FLATS:
            urls = urls[:MAX_FLATS]
        print(f">>> Итого: {len(urls)} объявлений\n")

        results = []
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def scrape_with_sem(url):
            async with semaphore:
                p = await context.new_page()
                try:
                    result = await scrape_flat(p, url)
                    if result:
                        results.append(result)
                finally:
                    await p.close()
                    await asyncio.sleep(2)

        tasks = [scrape_with_sem(u) for u in urls]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Парсинг"):
            await coro

        await browser.close()

    # Сохраняем
    if results:
        df_new = pd.DataFrame(results)

        # Сохраняем как строки, не разворачиваем
        for col in ["characteristics", "factoids"]:
            if col in df_new.columns:
                df_new[col] = df_new[col].apply(
                    lambda x: str(x) if isinstance(x, dict) else x
                )

        out = os.path.join(DATA_DIR, "flats_data.csv")

        if os.path.exists(out):
            df_old = pd.read_csv(out)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["flat_id"])
            df_combined.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"\n>>> Готово! Добавлено {len(df_new)}, итого {len(df_combined)} → {out}")
        else:
            df_new.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"\n>>> Готово! {len(df_new)} объявлений → {out}")
    else:
        print("\n>>> Данные не собраны.")


if __name__ == "__main__":
    asyncio.run(main())