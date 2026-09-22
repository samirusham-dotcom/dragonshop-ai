"""Dragon Shop AI — API поиска и умного помощника."""

import logging
import os
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:  # Позволяет запустить поиск даже до установки AI-зависимости.
    OpenAI = None


logger = logging.getLogger(__name__)

app = FastAPI(title="Dragon Shop AI - MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для MVP; в production ограничьте домен фронтенда.
    allow_methods=["*"],
    allow_headers=["*"],
)


class Product(BaseModel):
    title: str
    price_cny: float
    price_kzt: float
    price_with_delivery_kzt: float
    rating: float
    reviews_count: int
    image_url: str
    source: str
    product_url: str


class ChatMessage(BaseModel):
    """One previous turn supplied by the browser."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2_000)
    history: List[ChatMessage] = Field(default_factory=list, max_length=12)


class ChatResponse(BaseModel):
    reply: str


CNY_TO_KZT = 68.0
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
DRAGON_AI_INSTRUCTIONS = """
Ты — Dragon AI, доброжелательный и практичный помощник интернет-магазина Dragon Shop AI.
Отвечай по-русски, если пользователь не выбрал другой язык. Помогай формулировать
поисковые запросы, сравнивать товары, объяснять цены и пользоваться сайтом. Можешь
поддержать обычный разговор, но отвечай кратко и по делу. Не выдавай тестовые товары
за реальные, не придумывай наличие, сроки доставки, скидки или статусы заказов. Если
для ответа нужна актуальная карточка товара, предложи пользователю выполнить поиск.
""".strip()


def mock_search(query: str) -> List[Product]:
    """Temporary catalog data until a supplier API is connected."""
    fake_results = [
        {"title": f"{query} — вариант 1 (Pinduoduo)", "price_cny": 89, "rating": 4.8, "reviews_count": 1240, "image_url": "https://placehold.co/600x600/f27258/fff?text=Product+1", "source": "Pinduoduo", "product_url": "https://example.com/product1"},
        {"title": f"{query} — вариант 2 (Taobao)", "price_cny": 112, "rating": 4.6, "reviews_count": 530, "image_url": "https://placehold.co/600x600/d9ef75/202422?text=Product+2", "source": "Taobao", "product_url": "https://example.com/product2"},
        {"title": f"{query} — вариант 3 (JD)", "price_cny": 104, "rating": 4.9, "reviews_count": 89, "image_url": "https://placehold.co/600x600/d9e8f1/202422?text=Product+3", "source": "JD", "product_url": "https://example.com/product3"},
    ]
    products = []
    for item in fake_results:
        price_kzt = item["price_cny"] * CNY_TO_KZT
        products.append(Product(
            **item,
            price_kzt=round(price_kzt),
            price_with_delivery_kzt=round(price_kzt * 1.25),
        ))
    return products


@app.get("/")
def root():
    return {"status": "Dragon Shop AI backend работает"}


@app.get("/search", response_model=List[Product])
def search(
    query: str = Query(..., min_length=1, description="Например: чёрные кроссовки"),
    max_price_kzt: Optional[float] = Query(None, gt=0, description="Максимальная цена в тенге"),
):
    results = mock_search(query)
    if max_price_kzt is not None:
        results = [item for item in results if item.price_with_delivery_kzt <= max_price_kzt]
    return sorted(results, key=lambda item: item.rating, reverse=True)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a short, privacy-preserving conversation to the model."""
    if OpenAI is None:
        raise HTTPException(
            status_code=503,
            detail="AI-модуль не установлен. Выполните: pip install -r requirements.txt",
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Dragon AI ещё не настроен: задайте переменную OPENAI_API_KEY на сервере.",
        )

    messages = [
        {"role": item.role, "content": item.content}
        for item in request.history[-12:]
    ]
    messages.append({"role": "user", "content": request.message})

    try:
        response = OpenAI(api_key=api_key).responses.create(
            model=CHAT_MODEL,
            instructions=DRAGON_AI_INSTRUCTIONS,
            input=messages,
            max_output_tokens=500,
            store=False,
            text={"verbosity": "low"},
        )
        reply = response.output_text.strip()
    except Exception:
        logger.exception("Dragon AI request failed")
        raise HTTPException(
            status_code=502,
            detail="Dragon AI временно недоступен. Попробуйте ещё раз через минуту.",
        )

    if not reply:
        raise HTTPException(
            status_code=502,
            detail="Dragon AI не вернул текстовый ответ. Попробуйте переформулировать вопрос.",
        )
    return ChatResponse(reply=reply)
