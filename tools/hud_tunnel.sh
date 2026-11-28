#!/bin/bash
# HUD 의 인터넷 주소 확인 · 닫기.
#
#   tools/hud_tunnel.sh            지금 열려 있는 주소 (없으면 없다고)
#   tools/hud_tunnel.sh stop       닫는다 — 넘겨준 주소가 더는 열리지 않는다
#
# 주소는 스택을 hud_public:=true 로 띄울 때 HUD 가 연다:
#
#     ros2 launch isaac_bringup mvp_multi.launch.py ... hud_public:=true
#
# 두 가지 방식이 있고, HUD 가 쓸 수 있는 쪽을 고른다.
#
#   고정 주소 (Tailscale Funnel) — tailscale 이 설치돼 있고 로그인돼 있을 때.
#       https://<PC 이름>.<tailnet 이름>.ts.net  재부팅해도 같다.
#       준비: tailscale 설치 → `tailscale up` 으로 로그인 → 처음 한 번 Funnel 허용
#       (`tailscale funnel 8088` 이 알려 주는 주소에서 승인).
#   임시 주소 (Cloudflare) — 위가 안 될 때. https://….trycloudflare.com
#       계정이 필요 없지만 이름이 무작위다. 한 번 열린 터널은 스택을 내려도 남아 있어
#       (그동안 주소는 502) 스택을 다시 띄워도 같고, 닫았거나 재부팅하면 바뀐다.
#       cloudflared 가 필요하다: https://github.com/cloudflare/cloudflared/releases 의
#       cloudflared-linux-amd64 를 ~/.local/bin/cloudflared 로 두고 chmod +x.
#
# 접속 키는 ~/.ros/hud_key 에 있다 (지우면 다음에 새로 만든다).
#   · 영상·상태가 Tailscale 또는 Cloudflare 의 서버를 거쳐 나간다.
#   · 키 없이 띄운 HUD(hud_public·hud_key 둘 다 없음)는 터널로 들어온 요청을 전부 거절한다.
STATE="$HOME/.ros/hud_tunnel.json"
field() { python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" "$STATE" "$1" 2>/dev/null; }
PID="$(field pid)"; URL="$(field url)"
alive() { [ -n "$PID" ] && [ "$PID" != "0" ] && tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null | grep -q cloudflared; }
ts_url() {   # Tailscale Funnel 이 켜져 있으면 그 주소
    command -v tailscale >/dev/null || return 1
    tailscale funnel status 2>/dev/null | grep -q "Funnel on" || return 1
    tailscale funnel status 2>/dev/null | grep -o 'https://[a-z0-9.-]*\.ts\.net' | head -1
}
case "${1:-status}" in
  status)
    if TS="$(ts_url)" && [ -n "$TS" ]; then
        echo "열려 있음 (고정 주소, Tailscale Funnel): $TS"
        [ -f "$HOME/.ros/hud_key" ] && echo "접속 주소: $TS/?key=\$(cat ~/.ros/hud_key)"
    elif alive; then
        echo "열려 있음 (임시 주소, Cloudflare): $URL   (pid $PID, $(field started) 부터)"
        [ -f "$HOME/.ros/hud_key" ] && echo "접속 주소: $URL/?key=\$(cat ~/.ros/hud_key)"
    else
        echo "열려 있는 주소 없음"
    fi ;;
  stop)
    DONE=0
    if TS="$(ts_url)" && [ -n "$TS" ]; then tailscale funnel reset && echo "닫음: $TS" && DONE=1; fi
    if alive; then kill "$PID" && echo "닫음: $URL" && DONE=1; fi
    [ "$DONE" = 0 ] && echo "열려 있는 주소 없음"
    rm -f "$STATE" ;;
  *) echo "사용법: $0 [status|stop]" >&2; exit 2 ;;
esac
