# ─────────────────────────────────────────────────────────────
#  Bölüm:    Hız Sınırlayıcı (Rate Limiting)
#  Dosya:    app/rate_limit.py
#  Amaç:     API anahtarı başına RPM (dakikada istek) ve TPM (dakikada
#            token) kotalarını bellek içi, iş parçacığı güvenli şekilde
#            uygular.
#  Mekanik:  - RPM: token kovası (token bucket) — 1 istek = 1 jet/başlık.
#            - TPM: 60 sn'lik kayan pencere sayaçları (sonraki isteklerin
#              token akışı bu pencerede toplanır).
#            - Limite takılan çağrıda Retry-After bilgisi verilir.
#            - Oturum boyunca bellek içinde tutulur; sunucu yeniden
#              başlayınca sayaçlar sıfırlanır (kabul edilebilir varsayım).
# ─────────────────────────────────────────────────────────────

import threading
import time
from typing import Optional


class RateLimitExceeded(Exception):
    """Hız sınırı aşıldı; HTTP 429 ile sonuçlanır."""

    def __init__(self, limit: int, retry_after: int, kind: str) -> None:
        super().__init__(f"{kind} limiti aşıldı ({limit}).")
        self.limit = limit
        self.retry_after = retry_after
        self.kind = kind


class RateLimiter:
    """Anahtar başına RPM (kıvılcım) ve TPM (kayan pencere) takibi."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # RPM token kovası: key_id -> (kapasite, tokenler, son_dolşum_zamanı)
        self._buckets: dict[str, tuple[float, float, float]] = {}
        # TPM kayan pencere: key_id -> [(epoch, token)]
        self._tokens: dict[str, list[tuple[float, float]]] = {}

    def try_take_request(self, key_id: str, rpm: int) -> None:
        """RPM kontrolü: 1 istek için kovadan bir jeton alır.

        rpm <= 0 sınırsız demektir.
        """
        if rpm <= 0:
            return
        now = time.monotonic()
        with self._lock:
            cap = float(rpm)
            refill = cap / 60.0  # saniyede dolan jeton
            cap_b, tokens, last = self._buckets.get(key_id, (cap, cap, now))
            cap = cap_b
            # Geçen süre kadar jeton doldur (tavana kadar)
            tokens = min(cap, tokens + (now - last) * refill)
            if tokens < 1.0:
                # Kaç saniye sonra 1 jeton dolacak
                retry_after = max(1, int((1.0 - tokens) / refill))
                self._buckets[key_id] = (cap, tokens, now)
                raise RateLimitExceeded(rpm, retry_after, "RPM")
            self._buckets[key_id] = (cap, tokens - 1.0, now)

    def check_and_record_tokens(self, key_id: str, tpm: int, tokens: float) -> None:
        """TPM kontrolü: 60 sn'lik pencerede verilen token sayısını dener.

        Pencere doluysa RateLimitExceeded; değilse tokenları kaydeder.
        """
        if tpm <= 0:
            return
        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            window = [t for t in self._tokens.get(key_id, []) if t[0] > cutoff]
            used = sum(t[1] for t in window)
            if used + tokens > tpm:
                retry_after = max(1, int(60 - (now - (window[0][0] if window else cutoff))))
                raise RateLimitExceeded(tpm, retry_after, "TPM")
            window.append((now, tokens))
            self._tokens[key_id] = window

    def peek_tokens(self, key_id: str) -> float:
        """Son 60 sn'de tüketilen token toplamını döner (raporlama)."""
        cutoff = time.time() - 60.0
        with self._lock:
            return sum(t[1] for t in self._tokens.get(key_id, []) if t[0] > cutoff)


# Uygulama genelinde tek sınırlayıcı örneği
_limiter: Optional[RateLimiter] = None
_limiter_lock = threading.Lock()


def get_limiter() -> RateLimiter:
    """Tek RateLimiter örneğini döner (istenç güvenli)."""
    global _limiter
    if _limiter is None:
        with _limiter_lock:
            if _limiter is None:
                _limiter = RateLimiter()
    return _limiter