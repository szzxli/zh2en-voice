#!/usr/bin/env bash
# 手机浏览器麦克风需要 HTTPS（localhost 除外），生成自签名证书
openssl req -x509 -newkey rsa:2048 -nodes -days 365 -keyout key.pem -out cert.pem -subj "/CN=zh2en-voice"
