#!/bin/bash
# 이미 떠 있는 HUD 를 인터넷 주소로 연다 — 이 PC 의 HUD 포트 하나를 Cloudflare 터널로 내보낸다.
#
# 보통은 이 스크립트가 필요 없다. 스택을 띄울 때 hud_public:=true 를 주면 HUD 가 터널을
# 직접 열고, 주소를 시작 로그와 화면 위쪽에 보여 준다:
#
#     ros2 launch isaac_bringup mvp_multi.launch.py ... hud_public:=true
#
# 이 스크립트는 hud_key:=<키> 로 이미 떠 있는 HUD 를 내리지 않고 주소만 만들 때 쓴다:
#
#     tools/hud_tunnel.sh [포트=8088]      → 찍힌 https://….trycloudflare.com 뒤에 /?key=<키>
#
# 같은 공유기 안의 기기는 터널이 필요 없다 — http://<이 PC 의 IP>:8088 (HUD 시작 로그에 찍힌다).
#
# 주의
#   · 터널 주소는 누구나 접속을 시도할 수 있다. 그래서 접속 키 없이 떠 있는 HUD 는 열어 주지
#     않는다 (ACTIONS 버튼이 로버를 움직인다). 보기만 시킬 거면 hud_actions:=false 도 같이.
#   · 영상·상태가 Cloudflare 의 서버를 거쳐 나간다. Ctrl+C 로 닫으면 주소가 사라진다.
#   · cloudflared 가 필요하다: https://github.com/cloudflare/cloudflared/releases 의
#     cloudflared-linux-amd64 를 ~/.local/bin/cloudflared 로 두고 chmod +x.
PORT="${1:-8088}"
CF="$(command -v cloudflared || true)"
[ -z "$CF" ] && [ -x "$HOME/.local/bin/cloudflared" ] && CF="$HOME/.local/bin/cloudflared"
if [ -z "$CF" ]; then
    echo "✗ cloudflared 가 없습니다 (이 파일 머리말의 설치 방법 참고)." >&2; exit 1
fi
# 밖에서 온 요청처럼(전달 헤더를 붙여) 물어본다 — 이 PC 에서 직접 연 요청은 키 없이도 통과하므로.
code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' -H 'X-Forwarded-For: 192.0.2.1' \
       "http://127.0.0.1:${PORT}/api/info")
case "$code" in
  403) ;;                                   # 키가 걸려 있다 — 열어도 된다
  200) echo "✗ 포트 ${PORT} 의 HUD 가 접속 키 없이 떠 있습니다. hud_key:=<키> 를 줘서 다시 띄우세요." >&2; exit 1 ;;
  *)   echo "✗ 포트 ${PORT} 에 HUD 가 없습니다 (응답 ${code}). 스택을 먼저 띄우세요." >&2; exit 1 ;;
esac
echo "HUD(포트 ${PORT}) 를 터널로 엽니다. 나오는 주소 뒤에 /?key=<키> 를 붙이세요. 닫기: Ctrl+C"
exec "$CF" tunnel --no-autoupdate --url "http://127.0.0.1:${PORT}"
