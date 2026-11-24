/* Mission Control front-end — Socket.IO + minimap canvas + HUD bindings.
   Multi-rover capable: byRover caching + selector + minimap 동시 색 분리.
   단일 rover 모드는 rover_id="" 로 자동 fallback. */

(function () {
  const $ = (id) => document.getElementById(id);

  /* ── Connection pill ────────────────────────────────────────── */
  const sock = io();
  const pill = $("conn-pill");

  sock.on("connect", () => {
    pill.textContent = "ONLINE";
    pill.classList.remove("pill-offline");
    pill.classList.add("pill-online");
  });
  sock.on("disconnect", () => {
    pill.textContent = "OFFLINE";
    pill.classList.remove("pill-online");
    pill.classList.add("pill-offline");
  });

  /* ── Multi-rover state ───────────────────────────────────────── */
  // byRover[rid] = {state, odom, path, target, minimap}
  const byRover = {};
  let activeRover = "";  // 선택된 rover_id (selector 와 연동)
  let knownRovers = [""]; // backend 가 connect 시 emit 한 namespace 리스트

  // rover_id → minimap overlay 색. 추가 rover 는 cycle.
  const ROVER_COLORS = {
    "":         { rover: "#6cd9ff", path: "#6cd9ff" }, // 단일 모드 cyan
    "rover_1":  { rover: "#6cd9ff", path: "#6cd9ff" }, // cyan
    // reveal area 가 #ffd84a (yellow) 라 같은 색이면 path 보임 X.
    // violet 으로 변경해 노란 안개걷힌 영역 위에서도 명확히 식별 (2026-05-27).
    "rover_2":  { rover: "#a98cff", path: "#a98cff" }, // violet
    "rover_3":  { rover: "#ff5ce0", path: "#ff5ce0" }, // magenta
    "rover_4":  { rover: "#6dffb0", path: "#6dffb0" }, // green
  };
  function colorFor(rid) {
    return ROVER_COLORS[rid] || { rover: "#aaaaaa", path: "#aaaaaa" };
  }

  function ensureRover(rid) {
    if (!byRover[rid]) byRover[rid] = {};
    return byRover[rid];
  }

  /* ── Selector UI ────────────────────────────────────────────── */
  function buildSelector(namespaces) {
    const host = $("rover-selector");
    if (!host) return;
    host.innerHTML = "";
    if (!namespaces || namespaces.length === 0 || (namespaces.length === 1 && namespaces[0] === "")) {
      // 단일 모드 — selector 숨김
      host.style.display = "none";
      activeRover = "";
      return;
    }
    host.style.display = "";
    namespaces.forEach((ns) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "rover-chip";
      chip.dataset.rid = ns;
      chip.style.borderColor = colorFor(ns).rover;
      chip.textContent = ns || "rover_0";
      chip.addEventListener("click", () => {
        activeRover = ns;
        updateSelectorActive();
        refreshActiveViews();
      });
      host.appendChild(chip);
    });
    if (!activeRover || !namespaces.includes(activeRover)) {
      activeRover = namespaces[0];
    }
    updateSelectorActive();
  }
  function updateSelectorActive() {
    document.querySelectorAll(".rover-chip").forEach((c) => {
      c.classList.toggle("active", c.dataset.rid === activeRover);
    });
  }

  sock.on("rovers", (msg) => {
    knownRovers = msg.namespaces || [""];
    buildSelector(knownRovers);
    refreshActiveViews();
  });

  /* ── Mineral counters / phase / battery / collected (active rover) ── */
  function paintMinerals(state) {
    $("res-collected").textContent = state.collected_count | 0;
    $("res-goal").textContent = state.collection_goal | 0;
    $("res-blue").textContent   = state.collected_blue   | 0;
    $("res-yellow").textContent = state.collected_yellow | 0;
    $("res-green").textContent  = state.collected_green  | 0;
  }

  const PHASE_CLASSES = [
    "phase-EXPLORE", "phase-APPROACH", "phase-PICK_READY",
    "phase-RETURN_TO_BASE", "phase-MISSION_COMPLETE", "phase-MANUAL",
    "phase-RECOVERY",
  ];

  function paintPhase(rawState) {
    const badge = $("phase-badge");
    const manual = rawState.startsWith("MANUAL/");
    const bare = rawState.split("/").pop();
    PHASE_CLASSES.forEach((c) => badge.classList.remove(c));
    badge.classList.add(`phase-${bare}`);
    if (manual) badge.classList.add("phase-MANUAL");
    badge.textContent = (manual ? "MANUAL · " : "") + bare;
  }

  function paintBattery(pct, low, crit) {
    pct = Math.max(0, Math.min(100, pct));
    const bar = $("batt-bar");
    bar.style.width = pct.toFixed(1) + "%";
    bar.classList.remove("bar-ok", "bar-low", "bar-crit");
    bar.classList.add(crit ? "bar-crit" : low ? "bar-low" : "bar-ok");
    $("batt-pct").textContent = pct.toFixed(0) + "%";
  }

  function paintCollected(collected, goal) {
    goal = Math.max(1, goal | 0);
    const pct = Math.min(100, (collected / goal) * 100);
    $("col-bar").style.width = pct.toFixed(1) + "%";
    $("col-text").textContent = `${collected} / ${goal}`;
  }

  function renderStatusFor(rid) {
    const r = byRover[rid];
    if (!r) return;
    if (r.state) {
      const s = r.state;
      paintPhase(s.state || "EXPLORE");
      paintBattery(s.battery_percent, s.low_battery, s.critical_battery);
      paintCollected(s.collected_count | 0, s.collection_goal | 0);
      paintMinerals(s);
      $("task").textContent = s.active_task || "—";
      $("err").textContent = s.last_error || "—";
    }
    if (r.odom) {
      const o = r.odom;
      $("pos-xy").textContent = `${o.x.toFixed(2)} , ${o.y.toFixed(2)}`;
      $("pos-yaw").textContent = `${o.yaw_deg.toFixed(0)} °`;
      $("spd").textContent = `${o.speed.toFixed(2)} m/s`;
    }
    if (r.cmd_vel) {
      $("cmdv").textContent = `v=${r.cmd_vel.lin.toFixed(2)}  ω=${r.cmd_vel.ang.toFixed(2)}`;
    }
  }

  /* ── 카메라 viewport: active rover 의 스트림으로 연결 + 끊기면 다시 연결 ──
     web_video_server 는 raw topic path 를 받는다 (URL encoding 하면 /stream
     핸들러가 토픽을 인식 못 해 검은 화면이 된다).
     스택(시뮬레이터·web_video_server)이 페이지보다 늦게 뜨거나 재시작되면 <img> 는
     한 번 error 가 난 뒤 스스로 다시 시도하지 않는다. 그래서
       · error 가 나면 3 초 뒤 다시 연결하고,
       · socket 이 (다시) 연결될 때 = 스택이 다시 떴을 때 전부 새로 연결한다. */
  const CAMERAS = [
    // [img id, rover namespace 아래의 topic]
    ["cam-wrist", "perception/image_annotated"],        // 차체 카메라 + YOLO
    ["cam-chase", "camera/chase/image_raw"],            // 로버 뒤에서 따라가는 카메라
    ["cam-arm",   "perception/wrist_image_annotated"],  // M0609 손목 카메라 (공구 z 축)
  ];
  const camRetryTimers = {};

  function connectCamera(id, topicTail) {
    const img = $(id);
    if (!img) return;
    if (!img.dataset.srcTemplate) {
      img.dataset.srcTemplate = img.getAttribute("src") || "";   // 첫 실행 시 캐시
    }
    const tmpl = img.dataset.srcTemplate;
    if (!tmpl) return;
    const topic = activeRover ? `/${activeRover}/${topicTail}` : `/${topicTail}`;
    clearTimeout(camRetryTimers[id]);
    img.classList.remove("cam-down");
    // _t: 같은 URL 이면 브라우저가 다시 요청하지 않으므로 매번 다르게.
    img.src = tmpl.replace(/topic=[^&]+/, `topic=${topic}`) + `&_t=${Date.now()}`;
    img.onerror = () => {
      img.classList.add("cam-down");
      clearTimeout(camRetryTimers[id]);
      camRetryTimers[id] = setTimeout(() => connectCamera(id, topicTail), 3000);
    };
  }

  function refreshCameraSrcs() {
    CAMERAS.forEach(([id, tail]) => connectCamera(id, tail));
  }

  function refreshActiveViews() {
    renderStatusFor(activeRover);
    refreshCameraSrcs();
    paintMap();
  }

  /* ── Socket events: per-rover caching ────────────────────────── */
  sock.on("state", (s) => {
    const rid = s.rover_id || "";
    ensureRover(rid).state = s;
    if (!knownRovers.includes(rid)) knownRovers.push(rid);
    if (rid === activeRover) renderStatusFor(rid);
  });
  sock.on("odom", (o) => {
    const rid = o.rover_id || "";
    ensureRover(rid).odom = o;
    paintMap();
    if (rid === activeRover) renderStatusFor(rid);
  });
  sock.on("cmd_vel", (c) => {
    const rid = c.rover_id || "";
    ensureRover(rid).cmd_vel = c;
    if (rid === activeRover) renderStatusFor(rid);
  });

  /* ── Minimap canvas (OccupancyGrid + rover pos) ──────────────── */
  const cvs = $("minimap");
  const ctx = cvs.getContext("2d");

  // 모든 rover 의 OccupancyGrid 를 셀 단위 머지 — 한 rover 라도 탐색한
  // 셀은 탐색완료로 표시. obstacle 발견 시 즉시 confirm. 그 결과 탭과
  // 무관하게 "두 rover 의 탐색 범위 합집합" 이 minimap 에 보인다.
  function mergeMinimaps() {
    const maps = [];
    for (const rid in byRover) {
      const m = byRover[rid].minimap;
      if (m && m.data) maps.push(m);
    }
    if (maps.length === 0) return null;
    if (maps.length === 1) return maps[0];
    const ref = maps[0];
    const n = ref.w * ref.h;
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      let v = -1;
      for (const m of maps) {
        if (m.w !== ref.w || m.h !== ref.h) continue;
        const mv = m.data[i];
        if (mv >= 50)            { v = mv; break; }      // obstacle 확정
        else if (mv >= 0 && v < 0) v = mv;               // 탐색완료 free
      }
      out[i] = v;
    }
    return { w: ref.w, h: ref.h, res: ref.res, ox: ref.ox, oy: ref.oy, data: out };
  }

  function paintMap() {
    const bgMap = mergeMinimaps();
    if (!bgMap) {
      ctx.fillStyle = "#050a14";
      ctx.fillRect(0, 0, cvs.width, cvs.height);
      ctx.fillStyle = "#5a7aa0";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText("awaiting /mission/minimap...", cvs.width / 2, cvs.height / 2);
      return;
    }
    const { w, h, data } = bgMap;
    const img = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const v = data[i];
      let r, g, b;
      if (v < 0)        { r =  18; g =  22; b =  30; }
      else if (v < 50)  { r = 255; g = 216; b =  74; }
      else              { r = 255; g =  80; b =  80; }
      const p = i * 4;
      img.data[p]     = r;
      img.data[p + 1] = g;
      img.data[p + 2] = b;
      img.data[p + 3] = 255;
    }
    const tmp = document.createElement("canvas");
    tmp.width = w; tmp.height = h;
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = "#050a14";
    ctx.fillRect(0, 0, cvs.width, cvs.height);
    ctx.save();
    ctx.translate(0, cvs.height);
    ctx.scale(cvs.width / w, -cvs.height / h);
    ctx.drawImage(tmp, 0, 0);
    ctx.restore();

    const worldToPx = (wx, wy) => {
      const px = ((wx - bgMap.ox) / bgMap.res) * (cvs.width / bgMap.w);
      const py = cvs.height - ((wy - bgMap.oy) / bgMap.res)
                              * (cvs.height / bgMap.h);
      return [px, py];
    };

    // 모든 rover 의 path 를 색 분리해 그림.
    //
    // 핵심: backend mission_manager 가 보내는 path 는 A* 풀린 시점의
    // grid cell 시작점 ~ 목표 의 전체 waypoint. rover 가 이미 path 중간을
    // 따라간 후엔 path[0] 이 stale 한 옛 위치를 가리킴. 그대로 그리면
    // "현위치와 다른 곳에서 path 가 시작" 처럼 보이거나, path 가 짧아
    // length<2 라 안 그려지는 사례 발생 (2026-05-27 rover_1/rover_2 시연).
    //
    // 해결: rover 의 현재 odom 위치를 path 의 첫 점으로 prepend,
    // 그 다음에 현재 위치에서 가장 가까운 waypoint 부터 끝까지 이어 그림.
    // length>=1 이면 충분 (마지막 waypoint 하나만 있어도 rover→그 점 라인).
    for (const rid in byRover) {
      const r = byRover[rid];
      const col = colorFor(rid);
      if (!r.path || !r.path.pts || r.path.pts.length < 1) continue;
      if (!r.odom) continue;  // rover 현 위치 모르면 stale path 안 그림
      const ox = r.odom.x, oy = r.odom.y;
      // 현재 위치에서 가장 가까운 waypoint 의 index 찾기 (남은 경로 시작).
      let startI = 0;
      let bestD2 = Infinity;
      for (let i = 0; i < r.path.pts.length; i++) {
        const dx = r.path.pts[i][0] - ox;
        const dy = r.path.pts[i][1] - oy;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestD2) { bestD2 = d2; startI = i; }
      }
      // 가장 가까운 waypoint 가 rover 뒤(이미 지나간 지점) 라도 인덱스+1
      // 쪽이 "남은 길" 의미상 자연. 단 끝 도달 직전이면 끝 인덱스 유지.
      const remainStart = Math.min(startI + 1, r.path.pts.length - 1);
      ctx.save();
      ctx.strokeStyle = col.path;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.globalAlpha = 1.0;
      ctx.beginPath();
      const [px0, py0] = worldToPx(ox, oy);
      ctx.moveTo(px0, py0);
      for (let k = remainStart; k < r.path.pts.length; k++) {
        const [px, py] = worldToPx(r.path.pts[k][0], r.path.pts[k][1]);
        ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.restore();
    }

    // 모든 rover 의 target marker (분홍 별).
    for (const rid in byRover) {
      const t = byRover[rid].target;
      if (t && Number.isFinite(t.x) && Number.isFinite(t.y)) {
        const [tx, ty] = worldToPx(t.x, t.y);
        ctx.save();
        ctx.fillStyle = "#ff5ce0";
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 1.0;
        const r1 = 9, r2 = 4;
        ctx.beginPath();
        for (let k = 0; k < 10; k++) {
          const r = (k % 2 === 0) ? r1 : r2;
          const a = -Math.PI / 2 + (k * Math.PI / 5);
          const px = tx + Math.cos(a) * r;
          const py = ty + Math.sin(a) * r;
          if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }
    }

    // 모든 rover 의 위치 마커 (색 분리, active 는 진하게).
    for (const rid in byRover) {
      const o = byRover[rid].odom;
      if (!o) continue;
      const col = colorFor(rid);
      const px = ((o.x - bgMap.ox) / bgMap.res) * (cvs.width / w);
      const py = cvs.height - ((o.y - bgMap.oy) / bgMap.res)
                  * (cvs.height / h);
      if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
      const yaw = (o.yaw_deg || 0) * Math.PI / 180;
      ctx.save();
      ctx.translate(px, py);
      ctx.rotate(-yaw);
      ctx.globalAlpha = 1.0;
      ctx.fillStyle = col.rover;
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(9, 0);
      ctx.lineTo(-6, 6);
      ctx.lineTo(-6, -6);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      // rover_id 라벨 (작게)
      if (rid) {
        ctx.rotate(yaw);
        ctx.globalAlpha = 0.95;
        ctx.fillStyle = "#fff";
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 3;
        ctx.font = "bold 10px monospace";
        ctx.textAlign = "center";
        ctx.strokeText(rid, 0, -10);
        ctx.fillText(rid, 0, -10);
      }
      ctx.restore();
    }

    // Basecamp marker (0,0).
    {
      const bx = ((0 - bgMap.ox) / bgMap.res) * (cvs.width / bgMap.w);
      const by = cvs.height - ((0 - bgMap.oy) / bgMap.res)
                  * (cvs.height / bgMap.h);
      ctx.save();
      ctx.strokeStyle = "#6dffb0";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(bx, by, 9, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }

  sock.on("minimap", (m) => {
    const rid = m.rover_id || "";
    ensureRover(rid).minimap = m;
    paintMap();
  });
  sock.on("path", (p) => {
    const rid = p.rover_id || "";
    ensureRover(rid).path = p;
    paintMap();
  });
  sock.on("target", (t) => {
    const rid = t.rover_id || "";
    const target = (Number.isFinite(t.x) && Number.isFinite(t.y))
                  ? { x: t.x, y: t.y } : null;
    ensureRover(rid).target = target;
    paintMap();
  });

  // initial paint with placeholder text.
  paintMap();

  /* ── Action grid is intentionally inert at this stage ────────── */
  document.querySelectorAll(".action-grid .act").forEach((btn) => {
    btn.addEventListener("click", (e) => e.preventDefault());
  });
})();
