import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def _clean(s):
    return " ".join(s.split()) if s else ""

def _parse_reviews(html):
    s = BeautifulSoup(html, "html.parser")
    scope = s.select_one('section.styles_reviewListContainer__2bg_p[data-nosnippet="false"]') or s
    rows = []
    for card in scope.select('div[data-testid="service-review-card-v2"]'):
        try:
            header = card.select_one('div.styles_reviewCardInnerHeader__8Xqy8')
            name = _clean(header.select_one('[data-consumer-name-typography="true"]').get_text(strip=True)) if header else ""
            country = _clean(header.select_one('[data-consumer-country-typography="true"]').get_text(strip=True)) if header else ""
            t = header.select_one('time[data-service-review-date-time-ago="true"], time') if header else None
            date_txt = _clean(t.get_text(strip=True) if t else "") or _clean(t.get("title") if t else "") or _clean(t.get("datetime") if t else "")

            score_el = card.select_one('img.CDS_StarRating_starRating__614d2e')
            review_score = ""
            if score_el and score_el.get("src"):
                # src example: .../stars/stars-5.svg
                src = score_el["src"]
                if "stars-" in src:
                    review_score = src.split("stars-")[-1].split(".")[0]

            body = card.select_one('div.styles_reviewContent__tuXiN[data-review-content="true"]') or card
            title = body.select_one('h2[data-service-review-title-typography="true"]')
            text = body.select_one('p[data-service-review-text-typography="true"]')

            rows.append({
                "name": name,
                "country": country,
                "date": date_txt,
                "review_score": review_score,
                "review_header": _clean(title.get_text(strip=True) if title else ""),
                "review_body": _clean(text.get_text(" ", strip=True) if text else ""),
            })
        except Exception as e:
            print(f"Error scraping review: {e}")
            continue
    print(f"Scraped {len(rows)} reviews")
    return rows

def _iter_pages(base_url, max_pages, delay, timeout):
    session = requests.Session()
    session.headers.update(HEADERS)
    page = 1
    scraped = 0
    while True:
        r = session.get(f"{base_url}?page={page}", timeout=timeout)
        if r.status_code == 404:
            return
        r.raise_for_status()
        yield r.text
        scraped += 1
        if max_pages and scraped >= max_pages:
            return
        page += 1
        if delay:
            time.sleep(delay)

def scrape_reviews_paginated(
    base_url="https://www.trustpilot.com/review/scaleway.com",
    max_pages=None,
    delay_seconds=2,
    timeout=20,
):
    rows = []
    for html in _iter_pages(base_url, max_pages, delay_seconds, timeout):
        page_rows = _parse_reviews(html)
        if not page_rows:
            break
        rows.extend(page_rows)

    return {k: [r[k] for r in rows] for k in ("name", "country", "date", "review_score", "review_header", "review_body")}


data = scrape_reviews_paginated()
df = pd.DataFrame(data=data)
df.to_csv("data/reviews/reviews.csv", index=False, encoding="utf-8")
