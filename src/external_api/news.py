import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import finnhub


class FinnhubNewsCollector:
    """Minimal OO Finnhub collector for pipeline integration.

    Features:
    - Defaults: start_date=now-30 days, end_date=now, entities=['AAPL']
    - Saves one JSONL per entity in output_dir
    - Returns all saved records as list[dict]
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_pool_file: str = "src/external_api/finnhub_api_pool.json",
        output_dir: str = "src/external_apis/finnhub_news_data",
        paginate: bool = True,
        page_delay_seconds: float = 1.0,
    ) -> None:
        self.api_pool_file = api_pool_file
        self.output_dir = output_dir
        self.paginate = paginate
        self.page_delay_seconds = page_delay_seconds

        self.api_key = api_key or self._load_first_active_api_key()
        if not self.api_key:
            raise ValueError(
                "No Finnhub API key provided and no active key found in finnhub_api_pool.json"
            )

        self.client = finnhub.Client(api_key=self.api_key)
        os.makedirs(self.output_dir, exist_ok=True)

    def collect(
        self,
        entities: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save: bool = False,
        fields: Optional[Dict] = None,
    ) -> List[Dict]:
        """Collect news for entities, save JSONL files, and return list of records.

        Args:
            entities: List of symbols (default ['AAPL'])
            start_date: 'YYYY-MM-DD' (default now - 30 days)
            end_date: 'YYYY-MM-DD' (default now)
            fields: Dictionary of fields to include in each record

        Returns:
            List[Dict]: Flat list of all saved records across entities.
        """
        entities = entities or ["AAPL"]
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        start_date = start_date or (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        all_records: List[Dict] = []

        for entity in entities:
            raw_items = (
                self._fetch_paginated(entity, start_date, end_date)
                if self.paginate
                else self._fetch_once(entity, start_date, end_date)
            )

            records = self._to_records(entity, raw_items)
            if fields is not None:
                records = [{k: r.get(k, "") for k in fields} for r in records]
            if save:
                self._save_entity_jsonl(entity, records)
            all_records.extend(records)
        
        return all_records

    def _fetch_once(self, entity: str, start_date: str, end_date: str) -> List[Dict]:
        try:
            return self.client.company_news(entity, _from=start_date, to=end_date) or []
        except Exception:
            return []

    def _fetch_paginated(self, entity: str, start_date: str, end_date: str) -> List[Dict]:
        """Paginate backwards in time using end-date cursor."""
        from_ts = datetime.strptime(start_date, "%Y-%m-%d").timestamp()
        current_to = end_date
        seen_urls = set()
        out: List[Dict] = []

        for _ in range(30):
            batch = self._fetch_once(entity, start_date, current_to)
            if not batch:
                break

            filtered = [item for item in batch if item.get("datetime", 0) >= from_ts]
            if not filtered:
                break

            added_this_page = 0
            for item in filtered:
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    out.append(item)
                    added_this_page += 1

            if added_this_page == 0 or len(filtered) < 50:
                break

            oldest_ts = min(i.get("datetime", 0) for i in filtered)
            if not oldest_ts or oldest_ts <= from_ts:
                break

            oldest_date = datetime.fromtimestamp(oldest_ts)
            next_to = (oldest_date - timedelta(days=1)).strftime("%Y-%m-%d")
            if next_to <= start_date:
                break

            current_to = next_to
            time.sleep(self.page_delay_seconds)

        return out

    def _to_records(self, entity: str, items: List[Dict]) -> List[Dict]:
        records: List[Dict] = []
        for item in items:
            dt_value = item.get("datetime", 0)
            if not dt_value:
                continue

            article = (item.get("headline", "") + "\n\n" + item.get("summary", "")).strip()
            if len(article) <= 50:
                continue

            records.append(
                {
                    "Date": datetime.fromtimestamp(dt_value).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "Url": item.get("url", ""),
                    "Article": article,
                    "Stock_symbol": entity.lower(),
                    "Article_title": item.get("headline", ""),
                    "Publisher": item.get("source", "Unknown"),
                    "Category": item.get("category", "General"),
                }
            )

        records.sort(key=lambda x: x["Date"], reverse=True)
        return records

    def _save_entity_jsonl(self, entity: str, records: List[Dict]) -> None:
        file_path = os.path.join(self.output_dir, f"{entity}.jsonl")
        with open(file_path, "w", encoding="utf-8") as f:
            for row in records:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")

    def _load_first_active_api_key(self) -> Optional[str]:
        if not os.path.exists(self.api_pool_file):
            return None

        try:
            with open(self.api_pool_file, "r", encoding="utf-8") as f:
                pool = json.load(f)
        except Exception:
            return None

        finnhub_keys = pool.get("finnhub", {})
        for key, meta in finnhub_keys.items():
            if meta.get("status") == "active":
                return key
        return None


if __name__ == "__main__":
    # Quick standalone usage:
    # python news.py
    collector = FinnhubNewsCollector()
    data = collector.collect(fields=["Date", "Article"])
    print(f"Saved and returned {len(data)} records.")
    for record in data[:5]:
        print(record)
    
