import json
import socket
import threading
import time

from adapters.dcs_adapter import DcsAdapter


def start_dummy_udp_server(port_holder):
    def run():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        port_holder.append(sock.getsockname()[1])
        while True:
            try:
                data, addr = sock.recvfrom(4096)
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
    server_thread = start_dummy_udp_server(ports)
    port = ports[0]

    client = DcsAdapter("127.0.0.1", port, capabilities=["overlay"])
    client.negotiate()  # no-op

    resp = client.send_overlay_intent({"intent": "highlight", "element_id": "pnt_404"}, expect_reply=True)
    assert resp and "ACK" in resp["reply"]

    resp2 = client.send_overlay_intent({"intent": "clear", "element_id": "pnt_404"}, expect_reply=True)
    assert resp2 and "CLEAR" in resp2["reply"]

    client.close()
