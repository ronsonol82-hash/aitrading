# get_instruments.py
import os
import requests
import json
from dotenv import load_dotenv

# Грузим токен
load_dotenv()
TOKEN = os.getenv("TINKOFF_API_TOKEN")

if not TOKEN:
    print("❌ Токен не найден в .env!")
    exit(1)

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# Обновленный список (15 старых + 1 новая акция + 1 старая валюта + 4 новые валюты)
TARGET_TICKERS = [
    # --- АКЦИИ (16 шт) ---
    "SBER", "LKOH", "GAZP", "GMKN", "TATN", 
    "NVTK", "SNGS", "ROSN", "PLZL", "MGNT",
    "NLMK", "CHMF", "ALRS", "MOEX", "IMOEX",
    "T",      # Т-Технологии (бывший TCSG)
    "SNGSP",  # <--- НОВАЯ (Сургут Преф)

    # --- ВАЛЮТЫ И МЕТАЛЛЫ (5 шт) ---
    "CNYRUB_TOM", # Юань
    "HKDRUB_TOM", # <--- НОВАЯ (Гонконг)
    "TRYRUB_TOM", # <--- НОВАЯ (Лира)
    "KZTRUB_TOM", # <--- НОВАЯ (Тенге)
    "GLDRUB_TOM"  # <--- НОВАЯ (Золото)
]

def find_figi(ticker, class_code="TQBR"):
    """
    Ищет инструмент через API V2 InstrumentsService/FindInstrument
    TQBR - основной режим торгов акциями (Т+1)
    """
    url = "https://invest-public-api.tinkoff.ru/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/FindInstrument"
    
    payload = {
        "query": ticker,
        "instrumentKinds": ["INSTRUMENT_TYPE_SHARE", "INSTRUMENT_TYPE_CURRENCY"]
    }
    
    try:
        resp = requests.post(url, headers=HEADERS, data=json.dumps(payload))
        if resp.status_code != 200:
            print(f"Error {resp.status_code}: {resp.text}")
            return None
            
        data = resp.json()
        instruments = data.get("instruments", [])
        
        for item in instruments:
            # Для акций ищем совпадение тикера и classCode TQBR (основной рынок)
            if item['ticker'] == ticker:
                # Фильтр для акций РФ (TQBR)
                if item['classCode'] == "TQBR":
                    return item['figi']
                # Фильтр для валют (CETS)
                if item['classCode'] == "CETS":
                    return item['figi']
                
        # Если не нашли точное совпадение по классу, берем первое попавшееся (аккуратно!)
        if instruments:
            return instruments[0]['figi']
            
    except Exception as e:
        print(f"Exception: {e}")
    return None

print("🔎 Поиск FIGI для топ-листа...\n")
print("TINKOFF_FIGI_MAP = {")

found_count = 0
for t in TARGET_TICKERS:
    # Пробуем найти
    figi = find_figi(t)
    if figi:
        print(f'    "{t}": "{figi}",')
        found_count += 1
    else:
        print(f'    # "{t}": "NOT_FOUND",')

print("}")
print(f"\n✅ Найдено: {found_count} из {len(TARGET_TICKERS)}")