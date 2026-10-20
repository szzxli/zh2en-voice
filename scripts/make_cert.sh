#!/usr/bin/env bash
# 手机浏览器麦克风需要 HTTPS（localhost 除外）。生成带局域网 IP 的自签名证书（iOS 需要 SAN 匹配）
# 用法: bash scripts/make_cert.sh [电脑局域网IP]
IP=${1:-$( (ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}') )}
openssl req -x509 -newkey rsa:2048 -nodes -days 365 -keyout key.pem -out cert.pem \
  -subj "/CN=zh2en-voice" -addext "subjectAltName=IP:${IP:-127.0.0.1},DNS:localhost"
echo "证书已生成（SAN IP=${IP}）：cert.pem / key.pem"
