#!/bin/bash
# HUD 의 인터넷 주소(Cloudflare 터널) 확인 · 닫기.
#
#   tools/hud_tunnel.sh            지금 열려 있는 주소 (없으면 없다고)
#   tools/hud_tunnel.sh stop       터널을 닫는다 — 넘겨준 주소가 더는 열리지 않는다
#
# 터널은 스택을 hud_public:=true 로 띄울 때 HUD 가 연다:
#
#     ros2 launch isaac_bringup mvp_multi.launch.py ... hud_public:=true
#
# 한 번 열린 터널은 스택을 내려도 남아 있다 (그동안 주소는 502). 그래서 스택을 다시 띄워도
# 주소가 같다. 주소가 바뀌는 때는 이 스크립트로 닫았거나 PC 를 재부팅했을 때뿐이다.
# 접속 키는 ~/.ros/hud_key 에 있다 (지우면 다음에 새로 만든다).
#
#   · 영상·상태가 Cloudflare 의 서버를 거쳐 나간다.
#   · 키 없이 띄운 HUD(hud_public·hud_key 둘 다 없음)는 터널로 들어온 요청을 전부 거절한다.
#   · cloudflared 가 필요하다: https://github.com/cloudflare/cloudflared/releases 의
#     cloudflared-linux-amd64 를 ~/.local/bin/cloudflared 로 두고 chmod +x.
STATE="$HOME/.ros/hud_tunnel.json"
field() { python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" "$STATE" "$1" 2>/dev/null; }
PID="$(field pid)"; URL="$(field url)"
alive() { [ -n "$PID" ] && tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null | grep -q cloudflared; }
case "${1:-status}" in
  status)
    if alive; then
        echo "열려 있음: $URL   (pid $PID, $(field started) 부터)"
        [ -f "$HOME/.ros/hud_key" ] && echo "접속 주소: $URL/?key=\$(cat ~/.ros/hud_key)"
    else
        echo "열려 있는 터널 없음"
    fi ;;
  stop)
    if alive; then kill "$PID" && echo "닫음: $URL"; else echo "열려 있는 터널 없음"; fi
    rm -f "$STATE" ;;
  *) echo "사용법: $0 [status|stop]" >&2; exit 2 ;;
esac
