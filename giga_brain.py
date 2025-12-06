# giga_brain.py
import requests
import json
import uuid

class GigaTrader:
    def __init__(self, auth_token):
        self.url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {auth_token}'
        }

    def analyze_logs(self, log_summary):
        """
        Отправляет статистику торгов в LLM для ревью.
        """
        prompt = f"""
        Ты - старший риск-менеджер хедж-фонда. 
        Посмотри на статистику моего торгового бота и дай жесткую критику.
        
        Статистика:
        {log_summary}
        
        Вопросы:
        1. Почему WinRate такой низкий?
        2. Есть ли перекос в сторону StopLoss?
        3. Что изменить в параметрах (SL, TP)?
        
        Отвечай кратко и по делу.
        """
        
        payload = {
            "model": "GigaChat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(payload), verify=False)
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"LLM Error: {e}"

# Пример использования в main.py:
# advisor = GigaTrader("ТВОЙ_ТОКЕН")
# print(advisor.analyze_logs(trade_stats_string))