import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class HistoryMessage(BaseModel):
    """Одна реплика диалога, передаваемая вызывающей стороной как контекст.

    Роль `system` намеренно не принимается: содержимое истории в конечном счёте пишет
    пользователь, и возможность передать системную реплику означала бы возможность
    переопределить системный промпт со всеми его правилами безопасности.
    """

    role: Literal["user", "assistant"]
    content: str


class Delivery(BaseModel):
    """Поля, нужные только для доставки ответа через WS-мост в chat-svc.

    Вынесены из корня запроса намеренно: это первичные ключи схемы `chat` нашей БД, и
    потребителю без chat-svc (Telegram-бот соседней команды) незачем их придумывать.
    """

    assistant_message_id: str
    message_id: str | None = None


class GenerateRequest(BaseModel):
    """Запрос на генерацию ответа.

    Обязательным является только `question`. Всё остальное опционально, поэтому один и тот же
    контракт обслуживает и наш веб-чат (`response_mode="ws_bridge"`, ответ уходит потоком в
    chat-svc), и Telegram-бот соседней команды (`response_mode="sync"`, ответ возвращается
    прямо в HTTP-ответе). См. README, §10-11.
    """

    question: str

    # Куда отдавать результат. Стриминг (SSE) не реализован — соседней команде он не нужен,
    # а нашему чату хватает WS-моста (README, §12.3, вопрос 2).
    response_mode: Literal["sync", "ws_bridge"] = "sync"
    delivery: Delivery | None = None

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    user_id: str | None = None
    # Заполняется сервером из заголовка X-Service-Name, а не вызывающей стороной:
    # для аналитики важно, чей это трафик, и подделывать его в теле запроса не должно быть можно.
    consumer: str | None = Field(default=None, exclude=True)

    profile: dict[str, Any] = Field(default_factory=dict)
    # История диалога. Ограничения на количество сообщений нет: обрабатываются все, пока
    # суммарно укладываются в MAX_HISTORY_CHARS, иначе отбрасываются самые старые.
    history: list[HistoryMessage] = Field(default_factory=list)

    # Ретривал выполняется внутри сервиса; поле заполняется в process_job, а не вызывающей
    # стороной (в отличие от старого контракта llm-svc, где чанки заранее клал rag-svc).
    rag: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_delivery_for_bridge(self) -> "GenerateRequest":
        if self.response_mode == "ws_bridge":
            if self.delivery is None:
                raise ValueError("response_mode='ws_bridge' требует объект delivery")
            if not self.session_id:
                raise ValueError("response_mode='ws_bridge' требует session_id")
        return self


class CancelRequest(BaseModel):
    """Отмена по `request_id` — идентификатору, который вызывающая сторона уже знает."""

    request_id: str
