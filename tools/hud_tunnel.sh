#!/bin/bash
# HUD 를 인터넷 주소로 연다 — 이 PC 의 HUD 포트 하나를 터널로 내보낸다.
#
#   1) 접속 키를 줘서 스택을 띄운다:   ros2 launch isaac_bringup mvp_multi.launch.py ... hud_key:=<키>
#   2) 다른 터미널에서:                 tools/hud_tunnel.sh [포트=8088]
#   3) 화면에 찍힌 https://... 주소 뒤에 /?key=<키> 를 붙여서 연다.
#
# 같은 공유기 안의 기기는 터널이 필요 없다 — http://<이 PC 의 IP>:8088 (HUD 시작 로그에 찍힌다).
#
# 주의
#   · 터널 주소는 누구나 접속을 시도할 수 있다. 그래서 접속 키 없이 떠 있는 HUD 는 열어 주지
#     않는다 (ACTIONS 버튼이 로버를 움직인다). 보기만 시킬 거면 hud_actions:=false 도 같이.
#   · 영상·상태가 터널 사업자의 서버를 거쳐 나간다.
#   · cloudflared 가 깔려 있으면 그것을(trycloudflare.com 임시 주소), 없으면 ssh 만으로 되는
#     localhost.run 을 쓴다. 둘 다 Ctrl+C 로 닫으면 주소가 사라진다.
PORT="${1:-8088}"
code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/api/info")
case "$code" in
  403) ;;                                   # 키가 걸려 있다 — 열어도 된다
  200) echo "✗ 포트 ${PORT} 의 HUD 가 접속 키 없이 떠 있습니다. hud_key:=<키> 를 줘서 다시 띄우세요." >&2; exit 1 ;;
  *)   echo "✗ 포트 ${PORT} 에 HUD 가 없습니다 (응답 ${code}). 스택을 먼저 띄우세요." >&2; exit 1 ;;
esac
echo "HUD(포트 ${PORT}) 를 터널로 엽니다. 나오는 주소 뒤에 /?key=<키> 를 붙이세요. 닫기: Ctrl+C"
if command -v cloudflared >/dev/null 2>&1; then
    exec cloudflared tunnel --url "http://127.0.0.1:${PORT}"
else
    exec ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 \
         -R "80:127.0.0.1:${PORT}" nokey@localhost.run
fi
