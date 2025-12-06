# telegram_scraper.py
import asyncio
import os
import re
import json
import pandas as pd
from datetime import datetime, timedelta
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from transformers import pipeline
import torch
import redis

# Импортируем конфиг для настроек Redis
try:
    from config import Config
except ImportError:
    # Заглушка, если запускаем скрипт отдельно от проекта
    class Config:
        USE_REDIS = True
        REDIS_HOST = 'localhost'
        REDIS_PORT = 6379

# --- КЛЮЧИ ---
# Берем из конфига, если он загрузился, или из переменных окружения
API_ID = getattr(Config, 'TG_API_ID', os.getenv('TELEGRAM_API_ID'))
API_HASH = getattr(Config, 'TG_API_HASH', os.getenv('TELEGRAM_API_HASH'))

if not API_ID or not API_HASH:
    raise ValueError("❌ Не найдены API_ID или API_HASH! Проверь .env или config.py")

CHANNELS = [
    'tree_of_alpha', 'unusual_whales', 'WatcherGuru', 'Tier10k', 'WalterBloomberg',
    'Cointelegraph', 'CryptoTownEU'
]

OUTPUT_FILE = 'data_cache/news_sentiment.csv'
RAW_FILE = 'data_cache/news_raw_bert.csv'
DAYS_BACK = 100 # Сколько дней истории грузить (уменьшил дефолт для скорости)

# --- ИНИЦИАЛИЗАЦИЯ AI ---
# Проверяем GPU
device = 0 if torch.cuda.is_available() else -1
device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
print(f"🧠 Загрузка FinBERT на {device_name}...")

# truncation=True и max_length=512 спасают от ошибок длинных текстов
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="ProsusAI/finbert", 
    device=device
)

# --- ИНИЦИАЛИЗАЦИЯ REDIS ---
redis_client = None
if Config.USE_REDIS:
    try:
        redis_client = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=0)
        redis_client.ping()
        print("⚡ Redis подключен.")
    except Exception as e:
        print(f"⚠️ Redis недоступен: {e}")

def clean_text(text):
    if not text: return ""
    # Убираем ссылки
    text = re.sub(r'http\S+', '', text)
    # Убираем лишние пробелы и переносы
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_bert_sentiment_batch(texts, batch_size=16):
    """
    Обрабатывает список текстов пачкой.
    batch_size=16 или 32 - зависит от твоей VRAM. 
    Если вылетит CUDA OOM, уменьшай до 8.
    """
    clean_texts = [t[:512] for t in texts] # Обрезаем по длине токенов BERT
    results = []
    
    # Защита от пустого списка
    if not clean_texts:
        return []

    try:
        # Pipeline сам умеет в батчи, если передать список
        predictions = sentiment_pipeline(clean_texts, truncation=True, batch_size=batch_size)
        
        for p in predictions:
            score = p['score']
            if p['label'] == 'negative':
                score = -score
            elif p['label'] == 'neutral':
                score = 0.0
            results.append(score)
            
    except Exception as e:
        print(f"🔥 GPU Batch Error: {e}")
        # Fallback: если батч упал, возвращаем нули
        return [0.0] * len(texts)
        
    return results

async def scrape_channel(client, channel_name, cutoff_date):
    print(f"   🕵️‍♂️ Канал: @{channel_name}...")
    
    # Сюда копим готовые данные
    final_data = []
    
    offset_id = 0
    limit = 100 # Сколько сообщений запрашиваем у Телеги за раз
    consecutive_old_messages = 0
    
    while True:
        try:
            history = await client(GetHistoryRequest(
                peer=channel_name, offset_id=offset_id, offset_date=None, 
                add_offset=0, limit=limit, max_id=0, min_id=0, hash=0
            ))
            
            if not history.messages: 
                break
            
            # 1. Сначала собираем "сырые" кандидаты из текущего куска истории
            batch_candidates = [] 
            
            for message in history.messages:
                if not message.date: continue
                msg_date = message.date.replace(tzinfo=None)
                
                if msg_date < cutoff_date:
                    consecutive_old_messages += 1
                    if consecutive_old_messages > 5: # Даем шанс 5 старым сообщениям (вдруг пины)
                        return final_data
                    continue 
                else:
                    consecutive_old_messages = 0 # Сброс, если нашли свежее

                if message.message:
                    text = clean_text(message.message)
                    if len(text) >= 10: # Только если текст достаточно длинный
                        batch_candidates.append({
                            'datetime': msg_date,
                            'text': text,
                            'channel': channel_name
                        })

            # 2. Если есть кандидаты, прогоняем их через BERT одной пачкой
            if batch_candidates:
                texts_to_process = [item['text'] for item in batch_candidates]
                
                # --- GPU BLAST ---
                scores = get_bert_sentiment_batch(texts_to_process, batch_size=32)
                
                # 3. Собираем результаты
                for i, item in enumerate(batch_candidates):
                    score = scores[i]
                    
                    # Фильтр шума
                    if abs(score) > Config.MIN_EDGE: # Берем порог из конфига или 0.01
                        entry = item.copy()
                        entry['sentiment'] = score
                        # Обрезаем текст для экономии памяти, полный текст нам не нужен для торговли
                        entry['text'] = entry['text'][:100] 
                        
                        final_data.append(entry)

                        # --- REDIS PUSH ---
                        # Пишем в Redis, если новость свежая (24ч)
                        if Config.USE_REDIS and redis_client:
                            if (datetime.now() - entry['datetime']).days < 1:
                                redis_entry = entry.copy()
                                redis_entry['datetime'] = str(redis_entry['datetime'])
                                try:
                                    redis_client.lpush("news_sentiment", json.dumps(redis_entry))
                                    redis_client.ltrim("news_sentiment", 0, 500)
                                except Exception as e:
                                    print(f"Redis Error: {e}")

            # Обновляем offset для следующего запроса к API Телеграма
            offset_id = history.messages[-1].id
            
            # Пауза, чтобы Дуров не забанил
            await asyncio.sleep(0.5)

        except Exception as e:
            print(f"    ⚠️ Ошибка с {channel_name}: {e}")
            break
            
    return final_data

async def main():
    print(f"🚀 Запуск News Hunter v3.0 (GPU + Redis)")
    if not os.path.exists('data_cache'): os.makedirs('data_cache')

    client = TelegramClient('anon_session', API_ID, API_HASH)
    await client.start()
    
    cutoff_date = datetime.now() - timedelta(days=DAYS_BACK)
    cutoff_date = cutoff_date.replace(tzinfo=None)
    
    all_news = []
    for channel in CHANNELS:
        news = await scrape_channel(client, channel, cutoff_date)
        print(f"    ✅ {channel}: {len(news)} записей.")
        all_news.extend(news)
        
    if not all_news: 
        print("❌ Новостей не найдено за указанный период.")
        return

    print(f"\n💾 Обработка данных...")
    df = pd.DataFrame(all_news)
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # Сохраняем сырые данные (полезно для отладки)
    df.to_csv(RAW_FILE)
    
    # Ресемплинг для торгового бота (агрегация по 15 минут)
    # Используем mean() для усреднения настроения за 15 минут
    df_resampled = df['sentiment'].resample('15min').mean().fillna(0)
    
    # Сглаживание EMA (Exponential Moving Average), чтобы убрать резкие пики
    df_resampled_ema = df_resampled.ewm(span=12).mean() # 3 часа
    
    # Сохраняем в формат, который ждет data_loader.py
    final_df = pd.DataFrame({
        'sentiment': df_resampled,
        'sentiment_ema': df_resampled_ema
    })
    
    final_df.to_csv(OUTPUT_FILE)
    print(f"🎉 Готово! \n   Сырые данные: {RAW_FILE} \n   Для бота: {OUTPUT_FILE}")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Скрапинг остановлен.")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")