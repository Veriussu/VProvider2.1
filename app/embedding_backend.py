# ─────────────────────────────────────────────────────────────
#  Bölüm:    Embedding Backend
#  Dosya:    app/embedding_backend.py
#  Amaç:     Metin gömme (embeddings) işini tembel (lazy) ve tek tekil
#            (singleton) model üzerinden yürütür; ağır sentence-transformers
#            paketi yalnızca ilk çağrıda yüklenir (import-guard).
#  Mekanik:  - EmbeddingBackend(path): model yolu ya da HF model kimliği.
#            - is_available: paket kurulu + yapay zeka hub'ı açık mı.
#            - embed(texts): girdi listesi için normalize vektörler (list).
#            - dimensions(): modelin gömme boyutu.
#            - get_backend(path): uygulama genelinde tek örnek.
#            Testler factory ile gerçek paket gerektirmeden çalışır.
#  Kullanım: backend = embedding_backend.get_backend("models/embeddings/x")
# ─────────────────────────────────────────────────────────────

import threading
from typing import Callable, Optional

from app import ai_hub
from app.config import settings

# Kodcuyu üreten fonksiyon: (path: str) -> SentenceTransformer benzeri
EncoderFactory = Callable[[str], object]


class EmbeddingBackend:
    """SentenceTransformer üzerinde gecikmeli (lazy) bir sarmalayıcı."""

    def __init__(self, path: str, factory: Optional[EncoderFactory] = None) -> None:
        self.path = path
        self._factory = factory
        self._model: Optional[object] = None
        self._lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        """Paket kurulu + hub açık mı? (gerçek yükleme yapmaz)."""
        return ai_hub.available("embedding")

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if not self.is_available:
                raise RuntimeError(
                    "Embedding desteği kapalı: sentence-transformers kurulu değil "
                    "veya AI hub devre dışı (AI_HUB_ENABLED=true olmalı)."
                )
            if self._factory is not None:
                self._model = self._factory(self.path)
            else:
                st = ai_hub.get("sentence_transformers")
                if st is None:
                    raise RuntimeError("sentence-transformers yüklenemedi.")
                self._model = st.SentenceTransformer(self.path)
            return self._model

    def dimensions(self) -> int:
        """Modelin gömme boyutunu döner (ölçülemezse örnek vektörle bulur)."""
        model = self._load()
        # Yeni ad (get_embedding_dimension), eski ad (get_sentence_embedding_dimension)
        for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            getter = getattr(model, name, None)
            if callable(getter):
                try:
                    dims = getter()
                except TypeError:
                    continue
                if dims:
                    return int(dims)
        return len(self.embed([""])[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Metin listesini normalize vektör dizisine çevirir."""
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


# ------------------------------------------------------------------
# Singleton erişimi
# ------------------------------------------------------------------

_backend: Optional[EmbeddingBackend] = None


def get_backend(
    path: Optional[str] = None,
    factory: Optional[EncoderFactory] = None,
) -> EmbeddingBackend:
    """Uygulama genelinde tek embedding arka ucu döner.

    path verilirse ve mevcut arka uç farklı bir yola işaret ediyorsa
    yeniden oluşturulur (testlerde "_reset" ile temiz planlanabilir).
    """
    global _backend
    requested = path or settings.embedding_model
    if _backend is None or (requested and _backend.path != requested):
        _backend = EmbeddingBackend(requested or "", factory=factory)
    return _backend


def _reset() -> None:
    """Tekil örneği sıfırlar (yalnızca testler için)."""
    global _backend
    _backend = None