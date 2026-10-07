# ─────────────────────────────────────────────────────────────
#  Bölüm:    API Gateway (OpenRouter tarzı anahtar yönetimi)
#  Dosya:    app/gateway.py
#  Amaç:     Tüm /v1/* isteklerine uygulanan kimlik doğrulama, yaşam döngüsü
#            (süre/iptal), hız sınırı (RPM/TPM), model allowlist, aylık token
#            kotası, prepaid bakiye ve kullanım kaydı katmanını sunar.
#  Mekanik:  - Anahtarlar yalnızca SHA-256 hash olarak saklanır; düz metin
#              oluşturma anında BİR KEZ döner.
#            - api_auth: FastAPI dependency; her istekte doğrulama + istek
#              sayacı + son kullanım günceller.
#            - ApiContext: route içinde enforce_model() ve charge() çağrılır.
#  Kullanım: router = APIRouter(dependencies=[Depends(gateway.api_auth)])
#            async def uç(ctx: ApiContext = Depends(gateway.api_auth)): ...
# ─────────────────────────────────────────────────────────────

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import Header, HTTPException, status

from app import auth
from app.pricing import compute_cost
from app.rate_limit import RateLimitExceeded, get_limiter


def _store():
    """Depo tekiline auth üzerinden erişir (test ortamlarının monkeypatch'ı uyumlu)."""
    return auth.get_store()

# /v1/* yanıtlarındaki OpenAI-uyumlu hata biçimi için
ERROR_BODY_META = {"type": "invalid_request_error", "param": None}


def _gateway_error(status_code: int, message: str, code: Optional[str] = None) -> HTTPException:
    """OpenAI hata yapısında (detail.error) HTTPException üretir."""
    return HTTPException(
        status_code=status_code,
        detail={"error": {"message": message, "type": "invalid_request_error", "param": None, "code": code}},
    )


def hash_key(key: str) -> str:
    """API anahtarının SHA-256 özetini döner (depoda yalnızca bu saklanır)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_key() -> tuple[str, str, str]:
    """Yeni API anahtarı üretir: (düz_metin, prefix, hash).

    Düz metin yalnızca oluşturma anında bir kez döner; daha sonra asla
    depodan okunamaz.
    """
    raw = secrets.token_hex(32)
    return raw, raw[:12], hash_key(raw)


class ApiContext:
    """Doğrulanmış API anahtarına dair timeout bağlamı."""

    def __init__(self, record: dict) -> None:
        self.record = record
        self.key_id = int(record["id"])

    # Yaygın alanlara kısa erişim
    @property
    def model_allowlist(self) -> list:
        raw = self.record.get("model_allowlist") or "[]"
        if isinstance(raw, list):
            return raw
        try:
            import json
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, TypeError):
            return []

    def enforce_model(self, model_id: str) -> None:
        """İzin verilen modeller listesini kontrol eder (403)."""
        allow = self.model_allowlist
        if allow and model_id not in allow:
            raise _gateway_error(
                403,
                f"'{model_id}' modeli bu anahtar için izinli değil. "
                f"İzinli modeller: {', '.join(allow) or '(listeleme)'}",
                code="model_not_allowed",
            )

    def charge(
        self,
        task: str,
        model_id: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        images: int = 0,
        videos: int = 0,
        audio_sec: float = 0.0,
        music: int = 0,
    ) -> float:
        """İsteğin kullanımını kota/bakiye/TPM hesaplarına işler; maliyeti döner."""
        self.enforce_model(model_id)
        total_tokens = int(prompt_tokens) + int(completion_tokens)
        tpm = int(self.record.get("rate_limit_tpm") or 0)
        if tpm > 0:
            try:
                get_limiter().check_and_record_tokens(str(self.key_id), tpm, total_tokens)
            except RateLimitExceeded as exc:
                raise _gateway_error(
                    429,
                    f"{exc.kind} sınırı aşıldı ({exc.limit}). Lütfen bir süre bekleyin.",
                    code="rate_limit_exceeded",
                ) from exc

        cost = compute_cost(task, prompt_tokens, completion_tokens, images, videos, audio_sec, music)
        store = _store()
        store.record_usage(
            key_id=self.key_id,
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            requests=0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            images=images,
            videos=videos,
            audio_sec=audio_sec,
            music=music,
            cost=cost,
        )
        store.spend_balance(self.key_id, cost)
        return cost


# ------------------------------------------------------------------
# Doğrulama ve FastAPI dependency
# ------------------------------------------------------------------

def authenticate(candidate: str) -> ApiContext:
    """API anahtarını doğrular; geçerliyse ApiContext, değilse hata fırlatır.

    Kontroller (sırayla): bulunma -> iptal -> süre -> bakiye -> RPM -> kota.
    """
    store = _store()
    if not candidate:
        raise _gateway_error(status.HTTP_401_UNAUTHORIZED, "API anahtarı gerekli.", code="missing_api_key")

    record = store.get_api_key_by_hash(hash_key(candidate)) if candidate else None
    if record is None:
        # Geriye dönük: düz metin saklanan miras anahtarlar (bkz. create_api_key)
        legacy = _match_legacy_key(store, candidate)
        if legacy is None:
            raise _gateway_error(status.HTTP_401_UNAUTHORIZED, "API anahtarı geçersiz.", code="invalid_api_key")
        record = legacy

    if int(record.get("revoked") or 0):
        raise _gateway_error(status.HTTP_403_FORBIDDEN, "API anahtarı iptal edilmiş.", code="api_key_revoked")

    expires = record.get("expires_at")
    if expires:
        try:
            parsed = datetime.fromisoformat(expires)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if parsed < datetime.now(timezone.utc):
                raise _gateway_error(status.HTTP_401_UNAUTHORIZED, "API anahtarının süresi dolmuş.", code="api_key_expired")
        except ValueError:
            pass  # bozuk değer esnetilir

    balance = float(record.get("balance_credits") or 0.0)
    if balance < 0:
        raise _gateway_error(status.HTTP_402_PAYMENT_REQUIRED, "Bakiye yetersiz. Kredi yükleyin.", code="insufficient_credits")

    key_id = int(record["id"])
    # RPM hız sınırı
    rpm = int(record.get("rate_limit_rpm") or 0)
    if rpm > 0:
        try:
            get_limiter().try_take_request(str(key_id), rpm)
        except RateLimitExceeded as exc:
            raise _gateway_error(
                429,
                f"RPM sınırı aşıldı ({exc.limit}). {exc.retry_after} sn sonra tekrar deneyin.",
                code="rate_limit_exceeded",
            ) from exc

    # Aylık token kotası (ay başında sıfırlanır)
    monthly = int(record.get("monthly_token_quota") or 0)
    if monthly > 0:
        now = datetime.now(timezone.utc)
        since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        used = _month_usage_tokens(store, key_id, since)
        if used >= monthly:
            raise _gateway_error(
                status.HTTP_402_PAYMENT_REQUIRED,
                f"Aylık token kotası doldu ({used}/{monthly}).",
                code="monthly_quota_exceeded",
            )

    ctx = ApiContext(record)
    # İstek sayacı + son kullanım (her doğrulanmış istek sayılır)
    store.record_usage(key_id, datetime.now(timezone.utc).strftime("%Y-%m-%d"), requests=1)
    store.touch_api_key(key_id)
    return ctx


def _match_legacy_key(store, candidate: str) -> Optional[dict]:
    """Düz metin saklanan miras anahtarlarla eşleşmeyi sabit zamanlı dener."""
    from app.auth import verify_api_key

    for row in store.get_all_api_keys():
        if verify_api_key(candidate, row):
            rec = store.get_api_key_by_hash(hash_key(candidate))
            return rec if rec else None
    return None


def _month_usage_tokens(store, key_id: int, since_iso: str) -> int:
    """Ay başından bu yana tüketilen toplam token sayısını döner."""
    u = store.usage_since(key_id, since_iso)
    return int(u.get("prompt_tokens") or 0) + int(u.get("completion_tokens") or 0)


def api_auth(authorization: str | None = Header(default=None)) -> ApiContext:
    """FastAPI dependency: /v1/* isteklerini gateway üzerinden doğrular."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise _gateway_error(
            status.HTTP_401_UNAUTHORIZED,
            "Geçersiz yetkilendirme. 'Authorization: Bearer <anahtar>' gönderin.",
            code="missing_api_key",
        )
    candidate = authorization.split(" ", 1)[1].strip()
    return authenticate(candidate)