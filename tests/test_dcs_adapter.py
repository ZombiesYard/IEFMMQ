import socket
import threading
import time

from adapters.dcs_adapter import DcsAdapter


def start_dummy_udp_server(port_holder, stop_event):
    def run():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(0.1)
        port_holder.append(sock.getsockname()[1])
        while not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            line = data.decode("utf-8").strip()
            if line.startswith("HILITE"):
                reply = f"ACK {line}"
            elif line == "CLEAR":
                reply = "ACK CLEAR"
            else:
                reply = f"ACK {line}"
            sock.sendto(reply.encode("utf-8"), addr)
        sock.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.05)
    return t


def test_dcs_adapter_handshake_and_overlay():
    ports = []
    stop_event = threading.Event()
    server_thread = start_dummy_udp_server(ports, stop_event)
    port = ports[0]

    with DcsAdapter("127.0.0.1", port, capabilities=["overlay"]) as client:
        client.negotiate()  # no-op; dummy server sends replies for testing
        resp = client.send_overlay_intent(
            {"intent": "highlight", "element_id": "pnt_404"}, expect_reply=True
        )
        assert resp and "ACK" in resp["reply"]

        resp2 = client.send_overlay_intent(
            {"intent": "clear", "element_id": "pnt_404"}, expect_reply=True
        )
        assert resp2 and "CLEAR" in resp2["reply"]
    stop_event.set()
    server_thread.join(timeout=0.5)


def test_dcs_adapter_invalid_intent_raises():
    ports = []
    stop_event = threading.Event()
    server_thread = start_dummy_udp_server(ports, stop_event)
    port = ports[0]
    try:
        with DcsAdapter("127.0.0.1", port) as client:
            try:
                client.send_overlay_intent({"intent": "highlight"}, expect_reply=False)
                assert False, "expected ValueError"
            except ValueError as exc:
                assert "Invalid overlay intent" in str(exc)
    finally:
        stop_event.set()
        server_thread.join(timeout=0.5)


def test_dcs_adapter_timeout_returns_none():
    # Use an unused port to force socket timeout on reply
    with DcsAdapter("127.0.0.1", 65530, timeout=0.01) as client:
        resp = client.send_overlay_intent(
            {"intent": "highlight", "element_id": "pnt_404"}, expect_reply=True
        )
        assert resp is None


def test_dcs_adapter_direct_methods():
    ports = []
    stop_event = threading.Event()
    server_thread = start_dummy_udp_server(ports, stop_event)
    port = ports[0]
    try:
        with DcsAdapter("127.0.0.1", port) as client:
            client.highlight("pnt_404")
            client.clear()
    finally:
        stop_event.set()
        server_thread.join(timeout=0.5)
