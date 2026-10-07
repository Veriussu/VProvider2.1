# ─────────────────────────────────────────────────────────────
#  Bölüm:    Veri Katmanı
#  Dosya:    app/user_store.py
#  Amaç:     Kullanıcı, oturum, uygulama ayarları ve site bilgilerini
#            SQLite veritabanında tutar.
#  Mekanik:  - UserStore sınıfı tek bir .db dosyasını iş parçacığı
#              güvenli (kilitli) şekilde yönetir.
#            - Tablolar: users (kullanıcılar), sessions (oturumlar),
#              settings (anahtar-değer ayarlar: api_key, setup_done),
#              site_info (proje adı, github/domain/mail, logo/favicon BLOB).
#            - Logo ve favicon, dosya yolu yerine BLOB olarak saklanır;
#              böylece klasör taşıma sorunları oluşmaz.
#  Kullanım: get_store() tek örneği (singleton) verir; tüm modüller
#            bu fonksiyon üzerinden erişir.
# ─────────────────────────────────────────────────────────────

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import SITE_IDENTITY, settings

# Oturum geçerlilik süresi (gün)
SESSION_TTL_DAYS = 7

# Site bilgileri için varsayılanlar (ilk oluşturmada kullanılır)


# settings tablosundaki özel anahtarlar
KEY_API_KEY = "api_key"
KEY_SETUP_DONE = "setup_done"


class UserStore:
    """SQLite üzerinde çalışan basit ve güvenli veri deposu."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()  # birden fazla istek aynı anda yazmasın

    # ------------------------------------------------------------------
    # İç yardımcılar
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Veritabanı bağlantısını açar; klasörü gerekirse oluşturur."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _run(self, sql: str, params: tuple = (), fetch: str | None = None) -> Any:
        """Kilitli şekilde tek sorgu çalıştırır; sonucu döner."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(sql, params)
                if fetch == "one":
                    return cur.fetchone()
                if fetch == "all":
                    return cur.fetchall()
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def _now(self) -> str:
        """UTC zamanını ISO biçiminde üretir."""
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Kurulum (şema + varsayılanlar)
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Tablo yapısını oluşturur ve varsayılan kayıtları ekler."""
        self._run(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL
            )
            """
        )
        self._run(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        self._run(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        self._run(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                name                TEXT NOT NULL,
                key                 TEXT UNIQUE,
                key_hash            TEXT UNIQUE,
                prefix              TEXT,
                created_at          TEXT NOT NULL,
                expires_at          TEXT,
                revoked             INTEGER NOT NULL DEFAULT 0,
                rate_limit_rpm      INTEGER NOT NULL DEFAULT 0,
                rate_limit_tpm      INTEGER NOT NULL DEFAULT 0,
                monthly_token_quota INTEGER NOT NULL DEFAULT 0,
                balance_credits     REAL    NOT NULL DEFAULT 0,
                model_allowlist     TEXT    NOT NULL DEFAULT '[]',
                last_used           TEXT
            )
            """
        )
        self._run(
            """
            CREATE TABLE IF NOT EXISTS usage_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id            INTEGER NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
                date              TEXT NOT NULL,
                requests          INTEGER NOT NULL DEFAULT 0,
                prompt_tokens     INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                images            INTEGER NOT NULL DEFAULT 0,
                videos            INTEGER NOT NULL DEFAULT 0,
                audio_sec         REAL    NOT NULL DEFAULT 0,
                music             INTEGER NOT NULL DEFAULT 0,
                cost              REAL    NOT NULL DEFAULT 0,
                UNIQUE(key_id, date)
            )
            """
        )
        self._migrate_api_keys()
        # Varsayılan ayar anahtarları: yalnızca YOKSA eklenir,
        # mevcut değerler (örn. API anahtarı) asla sıfırlanmaz
        # api_key için varsayılan değer yerleştirilmez; kullanıcı her zaman kendi anahtarını oluşturmalı
        self._run("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (KEY_SETUP_DONE, "0"))

    # ------------------------------------------------------------------
    # Şema yükseltme (eski veritabanları yeni sütunlara geçirilir)
    # ------------------------------------------------------------------

    def _migrate_api_keys(self) -> None:
        """api_keys tablosuna eksik sütunları ekler; miras anahtarları hash'ler.

        Daha önce oluşturulmuş veritabanları yalnızca (id, name, key, created_at)
        içerir; yeni alanlar ALTER ile eklenir ve key değerleri hash'lenerek
        düz metin kalıntısından güvenle taşınır.
        """
        added: list[str] = []
        with self._lock:
            conn = self._connect()
            try:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(api_keys)").fetchall()}
                for name, decl in (
                    ("key_hash", "TEXT"),
                    ("prefix", "TEXT"),
                    ("expires_at", "TEXT"),
                    ("revoked", "INTEGER NOT NULL DEFAULT 0"),
                    ("rate_limit_rpm", "INTEGER NOT NULL DEFAULT 0"),
                    ("rate_limit_tpm", "INTEGER NOT NULL DEFAULT 0"),
                    ("monthly_token_quota", "INTEGER NOT NULL DEFAULT 0"),
                    ("balance_credits", "REAL NOT NULL DEFAULT 0"),
                    ("model_allowlist", "TEXT NOT NULL DEFAULT '[]'"),
                    ("last_used", "TEXT"),
                ):
                    if name not in cols:
                        conn.execute(f"ALTER TABLE api_keys ADD COLUMN {name} {decl}")
                        added.append(name)
                # Miras anahtarlar: key varsa hash+prefix doldur
                rows = conn.execute(
                    "SELECT id, key FROM api_keys WHERE key IS NOT NULL AND key_hash IS NULL"
                ).fetchall()
                for row_id, plain in rows:
                    if plain:
                        conn.execute(
                            "UPDATE api_keys SET key_hash = ?, prefix = ? WHERE id = ?",
                            (hashlib.sha256(plain.encode()).hexdigest(), plain[:12], row_id),
                        )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Kullanıcılar
    # ------------------------------------------------------------------

    def user_count(self) -> int:
        """Veritabanındaki kullanıcı sayısını döner."""
        row = self._run("SELECT COUNT(*) AS n FROM users", fetch="one")
        return int(row["n"]) if row else 0

    def create_user(self, username: str, password_hash: str) -> bool:
        """Yeni kullanıcı oluşturur. Kullanıcı adı daha önce varsa False döner."""
        try:
            self._run(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, self._now()),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_user_by_username(self, username: str) -> Optional[dict]:
        """Kullanıcı adına göre kullanıcı kaydını sözlük olarak döner."""
        row = self._run(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username,),
            fetch="one",
        )
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Oturumlar
    # ------------------------------------------------------------------

    def create_session(self, token: str, user_id: int, ttl_days: int = SESSION_TTL_DAYS) -> str:
        """Kullanıcı için yeni bir oturum kaydı oluşturur."""
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(days=ttl_days)).isoformat()
        self._run(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now.isoformat(), expires),
        )
        return token

    def get_session(self, token: str) -> Optional[dict]:
        """Geçerli (süresi dolmamış) oturumu döner; süresi dolmuşsa siler."""
        row = self._run("SELECT * FROM sessions WHERE token = ?", (token,), fetch="one")
        if row is None:
            return None
        if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
            self.delete_session(token)
            return None
        return dict(row)

    def delete_session(self, token: str) -> None:
        """Oturumu siler (çıkış işlemi)."""
        self._run("DELETE FROM sessions WHERE token = ?", (token,))

    # ------------------------------------------------------------------
    # Anahtar-değer ayarları
    # ------------------------------------------------------------------

    def set_setting(self, key: str, value: str) -> None:
        """Tek bir ayarı yazar (yoksa ekler, varsa günceller)."""
        self._run(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def get_setting(self, key: str, default: str = "") -> str:
        """Tek bir ayarı okur; yoksa varsayılanı döner."""
        row = self._run("SELECT value FROM settings WHERE key = ?", (key,), fetch="one")
        return row["value"] if row else default

    def is_setup_done(self) -> bool:
        """İlk kurulumun yapılıp yapılmadığını döner."""
        return self.get_setting(KEY_SETUP_DONE, "0") == "1"

    def mark_setup_done(self) -> None:
        self.set_setting(KEY_SETUP_DONE, "1")

    def set_api_key(self, api_key: str) -> None:
        """OpenAI uyumlu API anahtarını kaydeder."""
        self.set_setting(KEY_API_KEY, api_key)

    def get_api_key(self) -> str:
        """Kayıtlı API anahtarını döner."""
        return self.get_setting(KEY_API_KEY)

    # ------------------------------------------------------------------
    # İsimlendirilmiş API anahtarları (birden çok /v1* anahtarı)
    # ------------------------------------------------------------------

    def list_api_keys(self) -> list[dict]:
        """Tüm isimlendirilmiş API anahtarlarını (ad, anahtar, tarih) döner."""
        rows = self._run("SELECT id, name, key, created_at FROM api_keys ORDER BY id", fetch="all")
        return [dict(r) for r in rows]

    def create_api_key(self, name: str, key: str) -> dict:
        """Yeni isimlendirilmiş API anahtarı ekler ve kaydı döner.

        Geriye dönük uyumlu bellek katmanı: key düz metin olarak da saklanır
        (yeni üretimler yalnızca hash+prefix ile saklanır; bkz.
        create_api_key_with_options).
        """
        row_id = self._run(
            "INSERT INTO api_keys (name, key, key_hash, prefix, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, key, hashlib.sha256(key.encode()).hexdigest(), key[:12], self._now()),
        )
        row = self._run(
            "SELECT id, name, key, created_at FROM api_keys WHERE id = ?", (row_id,), fetch="one"
        )
        return dict(row)

    def get_api_key_by_id(self, key_id: int) -> Optional[dict]:
        """Anahtar kimliğine göre kaydı döner; yoksa None."""
        row = self._run(
            "SELECT id, name, key, created_at FROM api_keys WHERE id = ?", (key_id,), fetch="one"
        )
        return dict(row) if row else None

    def delete_api_key(self, key_id: int) -> None:
        """İsimlendirilmiş anahtarı siler; son kalana kadar miras api_key temizler."""
        self._run("DELETE FROM api_keys WHERE id = ?", (key_id,))
        remaining = self.get_all_api_keys()
        if not remaining:
            self.set_setting(KEY_API_KEY, "")

    def get_all_api_keys(self) -> list[str]:
        """Doğrulama için tüm isimlendirilmiş anahtar değerlerini döner."""
        rows = self._run("SELECT key FROM api_keys WHERE key IS NOT NULL", fetch="all")
        return [r["key"] for r in rows]

    # ------------------------------------------------------------------
    # Anahtar-diskli yaşam döngüsü (gateway): yalnızca hash saklanır
    # ------------------------------------------------------------------

    _MASKED_FIELDS = (
        "id", "name", "prefix", "created_at", "expires_at", "revoked",
        "rate_limit_rpm", "rate_limit_tpm", "monthly_token_quota",
        "balance_credits", "model_allowlist", "last_used",
    )

    @staticmethod
    def _parse_allowlist(raw: str) -> list:
        """JSON model allowlist'ini listeye çevirir (bozuk ise boş liste)."""
        try:
            parsed = json.loads(raw or "[]")
            return parsed if isinstance(parsed, list) else []
        except (ValueError, TypeError):
            return []

    def create_api_key_with_options(
        self,
        name: str,
        key_hash: str,
        prefix: str,
        expires_at: Optional[str] = None,
        rate_limit_rpm: int = 0,
        rate_limit_tpm: int = 0,
        monthly_token_quota: int = 0,
        balance_credits: float = 0.0,
        model_allowlist: Optional[list] = None,
    ) -> dict:
        """Yeni API anahtarı ekler; düz metin anahtar ASLA saklanmaz.

        key geçen çağrıda bir kez döner (raw); depoda yalnızca key_hash+prefix.
        """
        allow = json.dumps(model_allowlist or [])
        # UNIQUE kısıtlaması için benzersiz placeholder (düz metin key saklanmaz)
        placeholder = f"hash-only-{uuid.uuid4().hex}"
        row_id = self._run(
            "INSERT INTO api_keys (name, key, key_hash, prefix, created_at, expires_at, "
            "rate_limit_rpm, rate_limit_tpm, monthly_token_quota, balance_credits, model_allowlist) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, placeholder, key_hash, prefix, self._now(), expires_at,
             int(rate_limit_rpm), int(rate_limit_tpm), int(monthly_token_quota),
             float(balance_credits), allow),
        )
        return self.get_api_key_record(row_id) or {}

    def get_api_key_record(self, key_id: int) -> Optional[dict]:
        """Anahtar kaydını (maskeli alanlarla) döner; yoksa None."""
        row = self._run(f"SELECT {', '.join(self._MASKED_FIELDS)} FROM api_keys WHERE id = ?",
                        (key_id,), fetch="one")
        if row is None:
            return None
        rec = dict(row)
        rec["model_allowlist"] = self._parse_allowlist(rec.get("model_allowlist") or "[]")
        rec["revoked"] = bool(rec.get("revoked"))
        return rec

    def get_api_key_by_hash(self, key_hash: str) -> Optional[dict]:
        """Hash'e göre tam kaydı (key dahil, doğrulama için) döner."""
        row = self._run(
            "SELECT id, name, key, key_hash, prefix, created_at, expires_at, revoked, "
            "rate_limit_rpm, rate_limit_tpm, monthly_token_quota, balance_credits, "
            "model_allowlist, last_used FROM api_keys WHERE key_hash = ?",
            (key_hash,), fetch="one",
        )
        if row is None:
            return None
        return dict(row)

    def list_api_key_records(self) -> list[dict]:
        """Panel için tüm anahtar kayıtlarını (maskeli) döner."""
        rows = self._run(f"SELECT {', '.join(self._MASKED_FIELDS)} FROM api_keys ORDER BY id",
                         fetch="all")
        out = []
        for r in rows:
            rec = dict(r)
            rec["model_allowlist"] = self._parse_allowlist(rec.get("model_allowlist") or "[]")
            rec["revoked"] = bool(rec.get("revoked"))
            out.append(rec)
        return out

    def update_api_key(self, key_id: int, **fields) -> Optional[dict]:
        """Anahtarın limit/ayar alanlarını günceller; kaydı (maskeli) döner."""
        allowed = {
            "name", "expires_at", "rate_limit_rpm", "rate_limit_tpm",
            "monthly_token_quota", "balance_credits", "model_allowlist",
        }
        sets, params = [], []
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k == "model_allowlist":
                v = json.dumps(v or [])
            sets.append(f"{k} = ?")
            params.append(v)
        if not sets:
            return self.get_api_key_record(key_id)
        params.append(key_id)
        self._run(f"UPDATE api_keys SET {', '.join(sets)} WHERE id = ?", tuple(params))
        return self.get_api_key_record(key_id)

    def revoke_api_key(self, key_id: int) -> None:
        """Anahtarı anında geçersiz kılar (revoked=1)."""
        self._run("UPDATE api_keys SET revoked = 1 WHERE id = ?", (key_id,))

    def set_api_key_revoked(self, key_id: int, revoked: bool) -> None:
        """Anahtarın iptal durumunu ayarlar (True=iptal, False=yeniden etkin)."""
        self._run("UPDATE api_keys SET revoked = ? WHERE id = ?", (1 if revoked else 0, key_id))

    def touch_api_key(self, key_id: int) -> None:
        """Anahtarın son kullanım zamanını günceller."""
        self._run("UPDATE api_keys SET last_used = ? WHERE id = ?", (self._now(), key_id))

    def spend_balance(self, key_id: int, amount: float) -> None:
        """Prepaid bakiyeden harcama yapar (bileşik ve atomiktir).

        Semantik:
          - balance == 0      : sınırsız anahtar; düşüm yapılmaz.
          - balance > 0       : prepaid kalan; harcanır, sıfır veya altına inerse
                                -0.000001 "borç" işaretçisiyle kilitlenir
                                (sonraki kimlik doğrulamada 402 alınır).
          - balance < 0       : zaten kilitli (borç); dokunulmaz.
        """
        self._run(
            """
            UPDATE api_keys SET balance_credits = CASE
                WHEN balance_credits = 0 THEN 0
                WHEN balance_credits < 0 THEN balance_credits
                WHEN balance_credits - ? <= 0 THEN -0.000001
                ELSE balance_credits - ?
            END WHERE id = ?
            """,
            (float(amount), float(amount), key_id),
        )

    # ------------------------------------------------------------------
    # Kullanım kaydı (usage_log): günlük granüler sayaçlar
    # ------------------------------------------------------------------

    def record_usage(
        self,
        key_id: int,
        date: str,
        requests: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        images: int = 0,
        videos: int = 0,
        audio_sec: float = 0.0,
        music: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Tek günün kullanım satırını (varsa arttırarak) günceller."""
        self._run(
            """
            INSERT INTO usage_log (key_id, date, requests, prompt_tokens, completion_tokens,
                images, videos, audio_sec, music, cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key_id, date) DO UPDATE SET
                requests = requests + excluded.requests,
                prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                completion_tokens = completion_tokens + excluded.completion_tokens,
                images = images + excluded.images,
                videos = videos + excluded.videos,
                audio_sec = audio_sec + excluded.audio_sec,
                music = music + excluded.music,
                cost = cost + excluded.cost
            """,
            (key_id, date, int(requests), int(prompt_tokens), int(completion_tokens),
             int(images), int(videos), float(audio_sec), int(music), float(cost)),
        )

    def usage_rows(self, key_id: int, days: int = 30) -> list[dict]:
        """Son `days` günün kullanım kayıtlarını (eski->yeni) döner."""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = self._run(
            "SELECT date, requests, prompt_tokens, completion_tokens, images, videos, "
            "audio_sec, music, cost FROM usage_log "
            "WHERE key_id = ? AND date >= ? ORDER BY date",
            (key_id, since), fetch="all",
        )
        return [dict(r) for r in rows]

    def usage_since(self, key_id: int, since_iso: str) -> dict:
        """Belirli bir zamandan bu yana toplam kullanımı döner (kota kontrolleri)."""
        row = self._run(
            "SELECT SUM(requests) AS requests, SUM(prompt_tokens) AS prompt_tokens, "
            "SUM(completion_tokens) AS completion_tokens, SUM(images) AS images, "
            "SUM(videos) AS videos, SUM(audio_sec) AS audio_sec, SUM(music) AS music, "
            "SUM(cost) AS cost FROM usage_log WHERE key_id = ? AND date >= ?",
            (key_id, since_iso), fetch="one",
        )
        if row is None:
            return {}
        return dict(row) or {}

    def usage_totals(self, key_id: int) -> dict:
        """Anahtarın TÜM geçmiş kullanım toplamlarını döner (panel özeti)."""
        row = self._run(
            "SELECT COALESCE(SUM(requests),0) AS requests, "
            "COALESCE(SUM(prompt_tokens),0) AS prompt_tokens, "
            "COALESCE(SUM(completion_tokens),0) AS completion_tokens, "
            "COALESCE(SUM(images),0) AS images, COALESCE(SUM(videos),0) AS videos, "
            "COALESCE(SUM(audio_sec),0) AS audio_sec, COALESCE(SUM(music),0) AS music, "
            "COALESCE(SUM(cost),0) AS cost FROM usage_log WHERE key_id = ?",
            (key_id,), fetch="one",
        )
        return dict(row) if row else {}

    # ------------------------------------------------------------------
    # Site bilgileri (proje kimliği + logo/favicon)
    # ------------------------------------------------------------------

    def get_site_info(self) -> dict:
        """Proje kimliğini döner.

        Değerler DB'den değil, kod içine gömülü SITE_IDENTITY sabitinden
        gelir; sistem ayarları değiştirilemez olarak tasarlanmıştır.
        """
        return dict(SITE_IDENTITY)


# ------------------------------------------------------------------
# Singleton erişimi: tüm modüller bu tek örnek üzerinden çalışır.
# İlk çağrıda veritabanı dosyası oluşturulur ve şema hazırlanır.
# ------------------------------------------------------------------

_default_store: Optional[UserStore] = None
_store_lock = threading.Lock()


def get_store() -> UserStore:
    """Uygulama genelinde tek UserStore örneğini döner."""
    global _default_store
    if _default_store is None:
        with _store_lock:
            if _default_store is None:
                _default_store = UserStore(settings.database_file)
                _default_store.init()
    return _default_store