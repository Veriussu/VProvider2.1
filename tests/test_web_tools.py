# ─────────────────────────────────────────────────────────────
#  Bölüm:    İnternet Erişimi (web araçları) Testleri
#  Dosya:    tests/test_web_tools.py
#  Amaç:     web_tools modülünü (araç şemaları, SSRF engeli, argüman
#            ayrıştırma, hata yolları) ve model yöneticisindeki sunucu-
#            tarafı web araç döngüsünü sahte motorla doğrular.
#  Mekanik:  Ağ gerektiren testler ayrılan (network) işaretli testlerdir;
#            birim testler ağa dokunmaz, yalnızca saf işlevleri çalıştırır.
# ─────────────────────────────────────────────────────────────

import asyncio

import pytest

from app import user_store as user_store_mod
from app import web_tools
from app.model_manager import ModelManager


# ------------------------------------------------------------------
# Yardımcı: aç/kapa durumunu test veritabanına bağlar
# ------------------------------------------------------------------

@pytest.fixture()
def web_store(store):
    """web_tools is_enabled/set_enabled okumasını test DB'sine bağlar."""
    user_store_mod.get_store = lambda: store
    return store


class ScriptedEngine:
    """Web araç döngüsü testleri için betikli sahte motor.

    script adımları: 'web' = web_search aracı çağrısı üret, 'custom' =
    dış araç çağrısı üret, 'stop' = doğrudan yanıt. Her run_chat bir adım
    ilerler; çalıştırma sayısı ve görülen mesaj/parametreler kaydedilir.
    """

    def __init__(self, path, script):
        self.path = path
        self.script = list(script)
        self._loaded = False
        self.run_calls = 0
        self.seen_messages = None
        self.seen_params = None
        self.last_tool_calls = None
        self.last_finish_reason = "stop"

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    def tool_call_info(self):
        return self.last_tool_calls, self.last_finish_reason

    def _call(self, name, call_id, args):
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": args},
        }

    def run_chat(self, messages, **params):
        self.run_calls += 1
        self.seen_messages = list(messages)
        self.seen_params = dict(params)
        step = self.script[self.run_calls - 1] if self.run_calls <= len(self.script) else "stop"
        if step == "web":
            self.last_tool_calls = [self._call("web_search", f"w{self.run_calls}", '{"query":"Ankara"}')]
            self.last_finish_reason = "tool_calls"
            return ""
        if step == "gemma":
            # Gemma tarzı: yapılandırılmış araç yok; içerikte metin işaretçisi
            self.last_tool_calls = None
            self.last_finish_reason = "stop"
            return f"<|tool_call>call:web_search{{query:<|\"|>Ankara<|\"|>}}<tool_call|>"
        if step == "custom":
            self.last_tool_calls = [self._call("bash", "custom1", '{"command":"ls"}')]
            self.last_finish_reason = "tool_calls"
            return ""
        self.last_tool_calls = None
        self.last_finish_reason = "stop"
        return "Nihai yanıt."

    def stream_chat(self, messages, _yield_events=False, **params):
        self.seen_messages = list(messages)
        self.seen_params = dict(params)
        if _yield_events:
            yield {"type": "content", "text": "Akışta"}
            yield {"type": "finish", "reason": "stop"}
        else:
            yield "Akışta"


def _web_manager(tmp_path, script, web_store, memory_mode="keep"):
    """Sahte betikli motorlu bir ModelManager kurar."""
    manager = ModelManager(
        models_dir=tmp_path,
        engine_factory=lambda info: ScriptedEngine(str(info.path), script),
        memory_mode=memory_mode,
    )
    (tmp_path / "model-a.gguf").write_bytes(b"data")
    return manager


# ------------------------------------------------------------------
# Durum (aç/kapa) ve şemalar
# ------------------------------------------------------------------

def test_default_enabled_uses_settings(web_store, monkeypatch):
    """DB ayarı boşken .env ayarı (varsayılan açık) geçerli olur."""
    assert web_tools.is_enabled() is True


def test_set_enabled_persists(web_store, monkeypatch):
    """Toggle DB'de kalıcıdır; sonraki okumalarda kullanılır."""
    web_tools.set_enabled(False)
    assert web_tools.is_enabled() is False
    web_tools.set_enabled(True)
    assert web_tools.is_enabled() is True


def test_tool_specs_shape():
    """Şemalar OpenAI function biçiminde ve beklenen alanlara sahip."""
    specs = web_tools.tool_specs()
    names = {s["function"]["name"] for s in specs}
    assert names == {"web_search", "fetch_url"}
    for s in specs:
        assert s["type"] == "function"
        assert s["function"]["parameters"]["type"] == "object"
        assert "required" in s["function"]["parameters"]
        assert s["function"]["description"]


def test_is_builtin():
    """Yerleşik araç adları tanınır; dış araçlar tanınmaz."""
    assert web_tools.is_builtin({"function": {"name": "web_search"}})
    assert web_tools.is_builtin({"function": {"name": "fetch_url"}})
    assert not web_tools.is_builtin({"function": {"name": "bash"}})
    assert not web_tools.is_builtin({})


# ------------------------------------------------------------------
# Gemma tarzı metin araç çağrısı ayrıştırıcısı
# ------------------------------------------------------------------

def test_parse_gemma_tool_call():
    """Metin işaretçisi OpenAI biçimine çevrilir; argümanlar sözlüktür."""
    content = "<|tool_call>call:web_search{query:<|\"|>Türkiye'nin başkenti?<|\"|>}<tool_call|>"
    calls = web_tools.parse_tool_calls(content)
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "web_search"
    assert fn["arguments"] == {"query": "Türkiye'nin başkenti?"}


def test_parse_gemma_multiple_and_fetch():
    """Birden çok çağrı ve fetch_url argümanı ayrıştırılır."""
    content = (
        "x<|tool_call>call:web_search{query:<|\"|>hava<|\"|>}<tool_call|>"
        "<|tool_call>call:fetch_url{url:<|\"|>https://example.com/<|\"|>}<tool_call|>"
    )
    calls = web_tools.parse_tool_calls(content)
    assert [c["function"]["name"] for c in calls] == ["web_search", "fetch_url"]
    assert calls[1]["function"]["arguments"]["url"] == "https://example.com/"


def test_parse_gemma_none_when_no_marker():
    """İşaretçi yoksa boş liste döner."""
    assert web_tools.parse_tool_calls("Sıradan bir yanıt.") == []
    assert web_tools.parse_tool_calls("") == []
    assert web_tools.parse_tool_calls(None) == []


# ------------------------------------------------------------------
# SSRF koruması (ağ gerektirmez)
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9055/",
        "http://localhost/",
        "http://10.0.0.5/admin",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://0.0.0.0/x",
        "http://[::1]/",
    ],
)
def test_ssrf_private_urls_blocked(url):
    blocked, reason = web_tools._ssrf_blocked(url)
    assert blocked, url
    assert reason


@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8/",
        "http://1.1.1.1/",
        "https://example.com/",
    ],
)
def test_ssrf_public_urls_allowed(url):
    blocked, _ = web_tools._ssrf_blocked(url)
    assert not blocked, url


def test_ssrf_invalid_scheme():
    blocked, reason = web_tools._ssrf_blocked("file:///etc/passwd")
    assert blocked and "http" in reason


# ------------------------------------------------------------------
# Hata yolları (ağ erişimi gerekmez)
# ------------------------------------------------------------------

def test_fetch_url_empty_or_badscheme():
    assert "boş" in web_tools.fetch_url("")
    assert "boş" in web_tools.fetch_url("   ")
    assert "http" in web_tools.fetch_url("ftp://example.com/x")


def test_fetch_url_private_blocked():
    r = web_tools.fetch_url("http://127.0.0.1:9055/admin")
    assert "yasak" in r


def test_web_search_empty_query():
    assert "boş" in web_tools.web_search("")


def test_run_tool_unknown_and_badargs():
    r = web_tools.run_tool({"function": {"name": "foo", "arguments": "{}"}})
    assert "bilinmeyen" in r
    r2 = web_tools.run_tool({"function": {"name": "web_search", "arguments": ""}})
    assert "boş" in r2


# ------------------------------------------------------------------
# Sunucu-tarafı web araç döngüsü (sahte motor)
# ------------------------------------------------------------------

def test_web_loop_executes_then_answers(tmp_path, web_store, monkeypatch):
    """Model web aracı çağırıp sonra yanıtlarsa döngü tamamlanır."""
    def fake_run_tool(tc):
        return "Arama sonucu: Ankara'dır."

    monkeypatch.setattr(web_tools, "run_tool", fake_run_tool)
    manager = _web_manager(tmp_path, ["web", "web", "stop"], web_store)
    manager.set_memory_mode("model-a", "keep")

    answer = _run(manager.chat("model-a", [{"role": "user", "content": "Türkiye'nin başkenti?"}]))

    engine = manager._engines["model-a"]
    assert answer == "Nihai yanıt."
    assert engine.run_calls == 3  # 2 web turu + 1 (döngü içi son tur) -> döngü zaten bitirdi

    tool_msgs = [m for m in engine.seen_messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert all(m["content"] == "Arama sonucu: Ankara'dır." for m in tool_msgs)
    assert any("web_search" in t["function"]["name"] for t in engine.seen_params["tools"])


def test_web_loop_keeps_custom_tool_for_client(tmp_path, web_store, monkeypatch):
    """Dış (custom) araç görülünce döngü durur; çağrı istemciye bırakılır."""
    calls = []
    monkeypatch.setattr(web_tools, "run_tool", lambda tc: calls.append(tc) or "x")
    manager = _web_manager(tmp_path, ["web", "custom"], web_store)
    manager.set_memory_mode("model-a", "keep")

    answer = _run(manager.chat("model-a", [{"role": "user", "content": "komut çalıştır"}]))
    engine = manager._engines["model-a"]

    assert answer == ""  # içerik boş; araç çağrısı motorda bekliyor
    c, finish = engine.tool_call_info()
    assert c and c[0]["function"]["name"] == "bash"
    assert finish == "tool_calls"
    assert len(calls) == 1  # yalnızca web aracı çalıştı


def test_web_loop_disabled_is_transparent(tmp_path, web_store, monkeypatch):
    """Kapalıyken döngü hiçbir ek iş yapmaz; yalnızca araç yoktur."""
    web_store.set_setting("web_tools_enabled", "0")
    monkeypatch.setattr(web_tools, "run_tool", lambda tc: "X")
    manager = _web_manager(tmp_path, ["stop"], web_store)
    manager.set_memory_mode("model-a", "keep")

    answer = _run(manager.chat("model-a", [{"role": "user", "content": "merhaba"}]))
    engine = manager._engines["model-a"]

    assert answer == "Nihai yanıt."
    assert engine.run_calls == 1
    assert "web_search" not in str(engine.seen_params.get("tools"))  # araç enjekte edilmedi
    assert not any(m["role"] == "tool" for m in engine.seen_messages)


async def _collect(agen):
    return [item async for item in agen]


def test_web_loop_stream_transparent_when_disabled(tmp_path, web_store):
    """Akış yolu kapalıyken ağ işlemine girmez; olayları olduğu gibi akar."""
    web_store.set_setting("web_tools_enabled", "0")
    manager = _web_manager(tmp_path, ["stop"], web_store)
    manager.set_memory_mode("model-a", "keep")

    evs = _run(_collect(manager.chat_stream_events("model-a", [{"role": "user", "content": "merhaba"}])))
    texts = [e["text"] for e in evs if e.get("type") == "content"]
    assert texts == ["Akışta"]


def test_web_loop_gemma_text_tool_calls(tmp_path, web_store, monkeypatch):
    """Yapılandırılmış araç olmadığında metin işaretçisi ayrıştırılıp çalıştırılır."""
    monkeypatch.setattr(web_tools, "run_tool", lambda tc: "Arama sonucu: Ankara'dır.")
    manager = _web_manager(tmp_path, ["gemma", "stop"], web_store)
    manager.set_memory_mode("model-a", "keep")

    answer = _run(manager.chat("model-a", [{"role": "user", "content": "başkent?"}]))

    engine = manager._engines["model-a"]
    assert answer == "Nihai yanıt."
    tool_msgs = [m for m in engine.seen_messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "Arama sonucu: Ankara'dır."


def _run(coro):
    return asyncio.run(coro)