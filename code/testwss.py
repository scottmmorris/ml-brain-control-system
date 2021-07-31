url = "wss://localhost:6868"
ws = websocket.create_connection(url, sslopt={"cert_reqs": ssl.CERT_NONE})