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

  // 광물 아이콘 — 지도 캔버스와 범례(index.html 의 svg)가 같은 path 를 쓴다. 16 px 상자, 중심 (0,0).
  //   blue_mineral   파란 결정 (위아래가 뾰족한 육각 기둥)
  //   yellow_mineral 노란 보석 (위가 평평하고 아래가 뾰족한 컷)
  //   green_gas      초록 가스통 (몸통 + 마개)
  const MINERAL_ICONS = {
    blue_mineral: {
      fill: "#3d9bff", dark: "#0b3d80",
      outline: new Path2D("M0,-9 L6,-3 L6,4 L0,9 L-6,4 L-6,-3 Z"),
      detail: new Path2D("M-6,-3 L0,1 L6,-3 M0,1 L0,9"),
    },
    yellow_mineral: {
      fill: "#ffd21f", dark: "#7a5a00",
      outline: new Path2D("M-5,-7 L5,-7 L9,-2 L0,9 L-9,-2 Z"),
      detail: new Path2D("M-9,-2 L9,-2 M-5,-7 L-3,-2 L0,9 L3,-2 L5,-7"),
    },
    green_gas: {
      fill: "#35d96a", dark: "#0b5a26",
      outline: new Path2D("M-3,-9 L3,-9 L3,-6 L6,-6 L6,8 L-6,8 L-6,-6 L-3,-6 Z"),
      detail: new Path2D("M-6,-1 L6,-1 M-6,3 L6,3"),
    },
  };
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

  // 모든 로버가 기억한 광물을 합친다 (같은 종류가 1 m 안이면 같은 광물).
  // by = 그 광물로 가고 있는 로버 ("" = 아직 아무도).
  function sharedMinerals() {
    const out = [];
    for (const rid in byRover) {
      const m = byRover[rid].minerals;
      if (!m || !Array.isArray(m.items)) continue;
      for (const it of m.items) {
        if (!Number.isFinite(it.x) || !Number.isFinite(it.y)) continue;
        let e = out.find((o) => o.c === it.c && Math.hypot(o.x - it.x, o.y - it.y) < 1.0);
        if (!e) { e = { c: it.c, x: it.x, y: it.y, by: "" }; out.push(e); }
        if (it.t) e.by = rid;
        else if (it.o && !e.by) e.by = it.o;
      }
    }
    return out;
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
    "phase-RECOVERY", "phase-ESTOP",
  ];

  function paintPhase(rawState) {
    const badge = $("phase-badge");
    const manual = rawState.startsWith("MANUAL/");
    const bare = rawState.split("/").pop();
    PHASE_CLASSES.forEach((c) => badge.classList.remove(c));
    badge.classList.add(`phase-${bare}`);
    if (manual) badge.classList.add("phase-MANUAL");
    badge.textContent = (manual ? "MANUAL · " : "") + phaseLabel(bare);
  }
  // 화면에 쓰는 이름. ESTOP 은 눈에 띄게 "E-STOP" 으로.
  function phaseLabel(bare) {
    return bare === "ESTOP" ? "E-STOP" : bare;
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
    if (r.terrain !== r._terrainPainted) { r._terrainPainted = r.terrain; paintTerrain(rid); }
    if (r.state) {
      const s = r.state;
      paintPhase(s.state || "EXPLORE");
      paintBattery(s.battery_percent, s.low_battery, s.critical_battery);
      paintCollected(s.collected_count | 0, s.collection_goal | 0);
      paintMinerals(s);
      $("task").textContent = s.active_task || "—";
      $("task").classList.toggle("bad", (s.state || "").split("/").pop() === "ESTOP");
      $("err").textContent = s.last_error || "—";
      paintPortraitState(rid, s.state || "");
      paintButtons(s.state || "");
    }
    if (r.odom) {
      const o = r.odom;
      $("pos-xy").textContent = `${o.x.toFixed(2)} , ${o.y.toFixed(2)}`;
      $("pos-yaw").textContent = `${o.yaw_deg.toFixed(0)} °`;
      $("spd").textContent = `${o.speed.toFixed(2)} m/s`;
      paintAttitude(o);
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
  let yoloOn = true;       // CAM 버튼: YOLO 표시가 얹힌 영상 ↔ 카메라 원본
  const CAMERAS = [
    // [img id, rover namespace 아래의 topic (YOLO 표시), 원본 topic]
    ["cam-wrist", "perception/image_annotated", "camera/rover/image_raw"],        // 차체 카메라
    ["cam-chase", "camera/chase/image_raw", "camera/chase/image_raw"],            // 뒤따르는 카메라
    ["cam-arm",   "perception/wrist_image_annotated", "camera/wrist/image_raw"],  // M0609 손목 카메라
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
    CAMERAS.forEach(([id, ann, raw]) => connectCamera(id, yoloOn ? ann : raw));
  }

  let renderEventsHook = null;   // EVENT LOG 를 고른 로버 것으로 다시 그린다 (아래에서 채움)
  function refreshActiveViews() {
    renderStatusFor(activeRover);
    refreshCameraSrcs();
    paintMap();
    if (renderEventsHook) renderEventsHook();
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
    const out = new Int8Array(n);
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

  // 배경(세 로버의 지도를 합쳐 색칠한 그림)은 지도가 바뀐 때에만 다시 만든다. 예전에는
  // 위치 한 번 받을 때마다 25 만 칸 × 로버 수를 합치고 다시 칠해서 브라우저가 CPU 한 코어를 썼다.
  let bgDirty = true, bgCache = null;
  function background() {
    if (!bgDirty && bgCache) return bgCache;
    const m = mergeMinimaps();
    if (!m) return null;
    const { w, h, data } = m;
    const tmp = document.createElement("canvas");
    tmp.width = w; tmp.height = h;
    const tctx = tmp.getContext("2d");
    const img = tctx.createImageData(w, h);
    const px = new Uint32Array(img.data.buffer);        // 0xAABBGGRR (little endian)
    for (let i = 0; i < w * h; i++) {
      const v = data[i];
      px[i] = v < 0 ? 0xFF1E1612 : (v < 50 ? 0xFF4AD8FF : 0xFF5050FF);
    }
    tctx.putImageData(img, 0, 0);
    bgCache = { w, h, res: m.res, ox: m.ox, oy: m.oy, canvas: tmp };
    bgDirty = false;
    return bgCache;
  }
  // 그리기 요청은 모아서 0.1 초에 한 번만 처리한다.
  let paintTimer = null;
  function paintMap() {
    if (paintTimer) return;
    paintTimer = setTimeout(() => { paintTimer = null; paintMapNow(); }, 100);
  }
  function paintMapNow() {
    const bgMap = background();
    if (!bgMap) {
      ctx.fillStyle = "#050a14";
      ctx.fillRect(0, 0, cvs.width, cvs.height);
      ctx.fillStyle = "#5a7aa0";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText("awaiting /mission/minimap...", cvs.width / 2, cvs.height / 2);
      return;
    }
    const { w, h } = bgMap;
    const tmp = bgMap.canvas;
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

    // 기억한 (아직 못 모은) 광물 — 종류마다 다른 모양·색의 아이콘 (지도 위 범례와 같은 그림).
    // 로버들이 서로 알려 주는 목록이라 모든 로버의 것을 합쳐 한 번씩만 그린다 (어느 로버
    // 버튼을 눌러도 같은 지도). 고리 = 그 광물로 가고 있는 로버의 색.
    for (const it of sharedMinerals()) {
      const icon = MINERAL_ICONS[it.c] || MINERAL_ICONS.blue_mineral;
      const [mx, my] = worldToPx(it.x, it.y);
      ctx.save();
      ctx.translate(mx, my);
      if (it.by) {
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 4.5;
        ctx.beginPath();
        ctx.arc(0, 0, 13, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.strokeStyle = colorFor(it.by).rover;
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 3;
      ctx.stroke(icon.outline);
      ctx.fillStyle = icon.fill;
      ctx.fill(icon.outline);
      ctx.strokeStyle = icon.dark;
      ctx.lineWidth = 1;
      ctx.stroke(icon.detail);
      ctx.restore();
    }

    // 별은 "집으러 갈 광물이 없을 때" 의 탐색 목표에만 쓴다 (로버 색). 광물로 가는 중이면
    // 목표는 그 광물 아이콘의 고리로 보인다.
    for (const rid in byRover) {
      const t = byRover[rid].target;
      const known = byRover[rid].minerals;
      if (known && Array.isArray(known.items) && known.items.some((it) => it.t)) continue;
      if (t && Number.isFinite(t.x) && Number.isFinite(t.y)) {
        const [tx, ty] = worldToPx(t.x, t.y);
        ctx.save();
        ctx.fillStyle = colorFor(rid).rover;
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

  /* ── 주변 높낮이 (지형 스캐너 → 높이 지도). 위쪽 = 로버의 앞 ──────── */
  const tcv = $("terrain");
  const tctx = tcv ? tcv.getContext("2d") : null;
  function heightColor(h) {          // h: 발밑 기준 높이 (m)
    const t = Math.max(-1, Math.min(1, h / 1.0));
    if (t >= 0) {                    // 회색 → 주황 → 빨강
      const r = Math.round(70 + 185 * t), g = Math.round(78 + 60 * t - 100 * t * t), b = Math.round(86 - 60 * t);
      return `rgb(${r},${Math.max(g, 30)},${Math.max(b, 20)})`;
    }
    const u = -t;                    // 회색 → 파랑
    return `rgb(${Math.round(70 - 50 * u)},${Math.round(78 + 20 * u)},${Math.round(86 + 160 * u)})`;
  }
  function fmtH(v) { return (v === null || v === undefined) ? "—" : (v >= 0 ? "+" : "") + v.toFixed(2); }
  function paintTerrain(rid) {
    if (!tctx) return;
    const t = ensureRover(rid).terrain;
    const W = tcv.width, H = tcv.height;
    tctx.fillStyle = "#0b1118"; tctx.fillRect(0, 0, W, H);
    if (!t || !t.patch) {
      $("terrain-fb").textContent = "스캐너 없음"; $("terrain-lr").textContent = ""; $("terrain-hl").textContent = "";
      return;
    }
    const p = t.patch, n = p.n, cell = p.cell, s = W / (n * cell);      // px / m
    const c = Math.cos(t.yaw), si = Math.sin(t.yaw), px = cell * s * 1.5;
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) {
        const h = p.h[a * n + b];
        if (h === null) continue;
        const dx = p.x0 + (b + 0.5) * cell - t.x, dy = p.y0 + (a + 0.5) * cell - t.y;
        const fwd = dx * c + dy * si, left = -dx * si + dy * c;
        tctx.fillStyle = p.hz[a * n + b] === "1" ? "#ff3df2" : heightColor(h * 0.05);
        tctx.fillRect(W / 2 - left * s - px / 2, H / 2 - fwd * s - px / 2, px, px);
      }
    }
    tctx.strokeStyle = "rgba(255,255,255,0.25)"; tctx.lineWidth = 1;      // 2 m, 4 m 원
    for (const r of [2, 4]) { tctx.beginPath(); tctx.arc(W / 2, H / 2, r * s, 0, 2 * Math.PI); tctx.stroke(); }
    tctx.fillStyle = "#ffffff";                                           // 로버 (앞이 위)
    tctx.beginPath(); tctx.moveTo(W / 2, H / 2 - 7); tctx.lineTo(W / 2 - 5, H / 2 + 5); tctx.lineTo(W / 2 + 5, H / 2 + 5); tctx.closePath(); tctx.fill();
    $("terrain-fb").textContent = `앞 ${fmtH(t.front)} · 뒤 ${fmtH(t.back)}`;
    $("terrain-lr").textContent = `좌 ${fmtH(t.left)} · 우 ${fmtH(t.right)}`;
    $("terrain-hl").textContent = `4 m 안 ${fmtH(t.lo)} … ${fmtH(t.hi)}`;
  }
  sock.on("terrain", (t) => {
    const rid = t.rover_id || "";
    ensureRover(rid).terrain = t;
    if (rid === activeRover) paintTerrain(rid);
  });

  // 지도는 "값,개수,값,개수…" 문자열(rle)로 온다 → 칸 배열로 푼다.
  function unpackMinimap(m) {
    if (!m || m.data || typeof m.rle !== "string") return m;
    const out = new Int8Array(m.w * m.h);
    const parts = m.rle.split(",");
    let k = 0;
    for (let i = 0; i + 1 < parts.length; i += 2) {
      const v = +parts[i], n = +parts[i + 1];
      if (v !== 0) out.fill(v, k, k + n);
      k += n;
    }
    m.data = out;
    m.rle = null;
    return m;
  }
  sock.on("minimap", (m) => {
    const rid = m.rover_id || "";
    ensureRover(rid).minimap = unpackMinimap(m);
    bgDirty = true;
    paintMap();
  });
  sock.on("path", (p) => {
    const rid = p.rover_id || "";
    ensureRover(rid).path = p;
    paintMap();
  });
  sock.on("minerals", (m) => {
    const rid = m.rover_id || "";
    ensureRover(rid).minerals = m;
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

  /* ── 사진 위 표시: 이름·상태, 자세계 ─────────────────────────── */
  const PORTRAIT_STATES = ["MANUAL", "RECOVERY", "ESTOP", "MISSION_COMPLETE"];

  function paintPortraitState(rid, rawState) {
    const frame = $("portrait-frame");
    const manual = rawState.startsWith("MANUAL/");
    const bare = rawState.split("/").pop();
    PORTRAIT_STATES.forEach((c) => frame.classList.remove(`st-${c}`));
    if (bare === "ESTOP" || bare === "RECOVERY" || bare === "MISSION_COMPLETE") {
      frame.classList.add(`st-${bare}`);
    } else if (manual) {
      frame.classList.add("st-MANUAL");
    }
    $("portrait-name").textContent = "DOGE" + (rid ? ` · ${rid}` : "");
    const tag = $("portrait-state");
    PHASE_CLASSES.forEach((c) => tag.classList.remove(c));
    tag.classList.add(`phase-${bare}`);
    if (manual && bare !== "ESTOP") tag.classList.add("phase-MANUAL");
    tag.textContent = (bare === "ESTOP" ? "E-STOP"
                       : manual ? "MANUAL" : bare.replace(/_/g, " ")) || "—";
  }

  function levelClass(el, deg, warn, bad) {
    el.classList.toggle("warn", deg >= warn && deg < bad);
    el.classList.toggle("bad", deg >= bad);
  }

  const att = $("attitude");
  const actx = att.getContext("2d");
  function paintAttitude(o) {
    const roll = o.roll_deg, pitch = o.pitch_deg, tilt = o.tilt_deg;
    if (!Number.isFinite(roll) || !Number.isFinite(pitch)) return;
    const sgn = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(0) + "°";
    $("att-roll").textContent = sgn(roll);
    $("att-pitch").textContent = sgn(pitch);
    levelClass($("att-roll"), Math.abs(roll), 20, 32);
    levelClass($("att-pitch"), Math.abs(pitch), 20, 30);
    [$("tilt"), $("att-tilt")].forEach((el) => {
      el.textContent = `${tilt.toFixed(0)}°`;
      levelClass(el, tilt, 25, 40);
    });

    // 자세계: 운전석에서 본 지평선. 앞이 들리면(pitch +) 지평선이 내려가고,
    // 오른쪽으로 기울면(roll +) 지평선이 왼쪽으로 돈다.
    const W = att.width, c = W / 2, R = c - 3;
    actx.clearRect(0, 0, W, W);
    actx.save();
    actx.beginPath(); actx.arc(c, c, R, 0, Math.PI * 2); actx.clip();
    actx.translate(c, c);
    actx.rotate(-roll * Math.PI / 180);
    const dy = Math.max(-R, Math.min(R, (pitch / 45) * R));   // ±45° 가 가장자리
    actx.fillStyle = "#1d4f8f"; actx.fillRect(-2 * R, -2 * R + dy, 4 * R, 2 * R);   // 하늘
    actx.fillStyle = "#8a4a2a"; actx.fillRect(-2 * R, dy, 4 * R, 2 * R);            // 땅
    actx.strokeStyle = "#fff"; actx.lineWidth = 2;
    actx.beginPath(); actx.moveTo(-2 * R, dy); actx.lineTo(2 * R, dy); actx.stroke();
    actx.restore();
    // 차체 기준 표시 (고정)
    const danger = tilt >= 40 ? "#ff6478" : tilt >= 25 ? "#ffba4a" : "#ffd84a";
    actx.strokeStyle = danger; actx.lineWidth = 3;
    actx.beginPath();
    actx.moveTo(c - R * 0.6, c); actx.lineTo(c - R * 0.2, c);
    actx.moveTo(c + R * 0.2, c); actx.lineTo(c + R * 0.6, c);
    actx.stroke();
    actx.fillStyle = danger; actx.beginPath(); actx.arc(c, c, 3, 0, Math.PI * 2); actx.fill();
    actx.strokeStyle = tilt >= 40 ? "#ff6478" : "#2a4870"; actx.lineWidth = 3;
    actx.beginPath(); actx.arc(c, c, R, 0, Math.PI * 2); actx.stroke();
  }

  /* ── EVENT LOG ───────────────────────────────────────────────
     최신이 위. 칸보다 길어지면 스크롤 막대로 넘긴다. 예전 줄을 읽으려고 내려가 있는
     동안에는 새 줄이 들어와도 보던 자리가 밀리지 않게 한다. */
  const evList = $("event-list");
  const EV_MAX = 400;          // 화면에 두는 줄 수
  const EV_KEEP = 1500;        // 기억하는 줄 수 (모든 로버)
  const evData = [];     // 시간순 (복사용)
  const hh = (n) => String(n).padStart(2, "0");
  function clock(t) {
    const d = new Date((t || 0) * 1000);
    return `${hh(d.getHours())}:${hh(d.getMinutes())}:${hh(d.getSeconds())}`;
  }
  function eventLine(ev) {
    const li = document.createElement("li");
    const t = document.createElement("span");
    t.className = "t";
    t.textContent = clock(ev.t);
    li.appendChild(t);
    if (ev.rover_id) {
      const r = document.createElement("span");
      r.className = "r";
      r.textContent = ev.rover_id;
      li.appendChild(r);
    }
    li.appendChild(document.createTextNode(ev.text || ""));
    if (ev.kind === "action") li.classList.add("ev-action");
    if (/ESTOP|RECOVERY|거부|실패/.test(ev.text || "")) li.classList.add("ev-warn");
    li.title = `${t.textContent} ${ev.rover_id || ""} ${ev.text || ""}` +
               (ev.origin ? ` (from ${ev.origin})` : "");
    return li;
  }
  // 고른 로버의 줄만 보인다 (로버 이름이 없는 줄 = 전체 알림은 늘 보인다).
  const evMine = (ev) => !ev.rover_id || !activeRover || ev.rover_id === activeRover;
  function showEvent(ev) {
    const away = evList.scrollTop > 4;          // 예전 줄을 보고 있다
    const li = eventLine(ev);
    evList.insertBefore(li, evList.firstChild);
    if (away) evList.scrollTop += li.offsetHeight;
    while (evList.children.length > EV_MAX) evList.removeChild(evList.lastChild);
  }
  function renderEvents() {
    evList.innerHTML = "";
    const mine = evData.filter(evMine);
    mine.slice(-EV_MAX).forEach(showEvent);
    $("event-count").textContent = `${mine.length}`;
  }
  renderEventsHook = renderEvents;
  function addEvent(ev) {
    evData.push(ev);
    if (evData.length > EV_KEEP) evData.shift();
    if (!evMine(ev)) return;
    showEvent(ev);
    $("event-count").textContent = `${evData.filter(evMine).length}`;
  }
  sock.on("event", addEvent);
  sock.on("events", (msg) => {
    evData.length = 0;
    (msg.items || []).forEach((ev) => evData.push(ev));
    renderEvents();
  });

  function copyText(text, onDone) {
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(onDone, () => {});
      return;
    }
    // http://<IP> 로 열면 secure context 가 아니라 clipboard API 가 없다.
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { if (document.execCommand("copy")) onDone(); } catch (_) { /* 복사 못 함 */ }
    document.body.removeChild(ta);
  }
  $("event-copy").addEventListener("click", () => {
    const b = $("event-copy");
    const text = evData.filter(evMine).map((ev) =>
      `${clock(ev.t)}\t${ev.rover_id || "-"}\t${ev.kind === "action" ? "[HUD] " : ""}${ev.text || ""}`
    ).join("\n");
    copyText(text, () => {
      b.textContent = `COPIED ${evData.filter(evMine).length}`;
      b.classList.add("copied");
      setTimeout(() => { b.textContent = "COPY"; b.classList.remove("copied"); }, 1200);
    });
  });

  /* ── ACTIONS ─────────────────────────────────────────────────── */
  const actionsOn = document.body.dataset.actions === "1";
  const hint = $("act-hint");
  const SPEEDS = [0.3, 0.6, 0.9];
  let speedIdx = 1;
  let manualOn = false;
  const armed = {};      // 두 번 눌러야 하는 버튼의 대기 타이머

  function btn(name) { return document.querySelector(`.act[data-act="${name}"]`); }
  function setHint(text) { hint.textContent = text || ""; }

  function paintButtons(rawState) {
    const manual = rawState.startsWith("MANUAL/");
    const bare = rawState.split("/").pop();
    manualOn = manual;
    $("lbl-mode").textContent = manual ? "MANUAL" : "AUTO";
    btn("mode").classList.toggle("act-on", manual);
    $("lbl-estop").textContent = bare === "ESTOP" ? "RESUME" : "E-STOP";
    $("phase-badge").classList.toggle("blink", bare === "ESTOP");
    btn("estop").classList.toggle("act-alert", bare === "ESTOP");
    $("lbl-rtb").innerHTML = bare === "RETURN_TO_BASE" ? "CANCEL<br/>RTB" : "RTB";
    btn("rtb").classList.toggle("act-on", bare === "RETURN_TO_BASE");
    if (!actionsOn) setHint("보기 전용");
    else if (bare === "ESTOP") setHint("정지 중 — RESUME 으로 해제");
    else if (manual) setHint("W A S D 로 주행");
    else setHint(activeRover ? `→ ${activeRover}` : "");
  }

  function send(action, value) {
    const b = btn(action);
    sock.emit("action", { rover_id: activeRover, action: action, value: value || "" }, (res) => {
      if (!b) return;
      b.classList.add("act-sent");
      setTimeout(() => b.classList.remove("act-sent"), 400);
      if (res && res.ok === false) setHint(res.text);
    });
  }

  function onAction(b) {
    const name = b.dataset.act;
    if (b.dataset.confirm === "1" && !armed[name]) {
      // 되돌리기 어려운 동작 — 3 초 안에 한 번 더 눌러야 실행한다.
      b.classList.add("act-armed");
      setHint("한 번 더 누르면 실행");
      armed[name] = setTimeout(() => {
        b.classList.remove("act-armed");
        armed[name] = null;
        setHint("");
      }, 3000);
      return;
    }
    if (armed[name]) {
      clearTimeout(armed[name]);
      armed[name] = null;
      b.classList.remove("act-armed");
      setHint("");
    }
    if (name === "cam") {
      yoloOn = !yoloOn;
      $("lbl-cam").innerHTML = yoloOn ? "YOLO<br/>ON" : "YOLO<br/>OFF";
      b.classList.toggle("act-on", !yoloOn);
      refreshCameraSrcs();
      send("cam", yoloOn ? "YOLO on" : "YOLO off");
    } else if (name === "speed") {
      speedIdx = (speedIdx + 1) % SPEEDS.length;
      $("lbl-speed").innerHTML = `SPD<br/>${SPEEDS[speedIdx].toFixed(1)}`;
      send("speed", `${SPEEDS[speedIdx].toFixed(1)} m/s`);
    } else {
      send(name);
    }
  }

  document.querySelectorAll(".action-grid .act").forEach((b) => {
    if (!actionsOn) { b.disabled = true; return; }
    b.addEventListener("click", (e) => { e.preventDefault(); onAction(b); b.blur(); });
  });

  /* ── 수동 주행 (MANUAL 일 때만): W/S 전후, A/D 좌우, 방향키도 같다 ──
     누르고 있는 동안 10 Hz 로 보낸다. 손을 떼면 0 을 한 번 보내고 멈춘다
     (mission_manager 도 0.5 s 동안 명령이 없으면 세운다). */
  const KEYS = {
    KeyW: [1, 0], ArrowUp: [1, 0], KeyS: [-1, 0], ArrowDown: [-1, 0],
    KeyA: [0, 1], ArrowLeft: [0, 1], KeyD: [0, -1], ArrowRight: [0, -1],
  };
  const held = new Set();
  let teleopTimer = null;

  function teleopTick() {
    let lin = 0, ang = 0;
    held.forEach((k) => { lin += KEYS[k][0]; ang += KEYS[k][1]; });
    const v = SPEEDS[speedIdx];
    sock.emit("teleop", {
      rover_id: activeRover,
      lin: Math.sign(lin) * v,
      ang: Math.sign(ang) * (lin < 0 ? -1 : 1) * 0.7,   // 후진 중에는 조향 방향을 뒤집는다
    });
    if (held.size === 0) { clearInterval(teleopTimer); teleopTimer = null; }
  }
  window.addEventListener("keydown", (e) => {
    if (!actionsOn || !manualOn || !(e.code in KEYS)) return;
    e.preventDefault();
    held.add(e.code);
    if (!teleopTimer) { teleopTick(); teleopTimer = setInterval(teleopTick, 100); }
  });
  window.addEventListener("keyup", (e) => { held.delete(e.code); });
  window.addEventListener("blur", () => { held.clear(); });

  /* ── 다른 기기에서 여는 주소 ───────────────────────────────────
     왼쪽 알약 = 같은 네트워크 주소, 오른쪽 알약 = 인터넷 주소(hud_public:=true).
     누르면 접속 키가 붙은 주소가 복사된다. 인터넷 주소는 터널이 열린 뒤에 생기므로
     주기적으로 다시 묻는다. */
  const shareUrl = {};
  function paintUrl(id, label, url) {
    const pill = $(id);
    if (!pill) return;
    pill.style.display = url ? "" : "none";
    if (!url) return;
    pill.textContent = label;
    if (!(id in shareUrl)) {
      pill.addEventListener("click", () => copyText(shareUrl[id], () => {
        pill.classList.add("copied");
        setTimeout(() => pill.classList.remove("copied"), 900);
      }));
    }
    shareUrl[id] = url;
  }
  function refreshUrls() {
    fetch("/api/info").then((r) => r.json()).then((info) => {
      const bare = (u) => u.replace(/^https?:\/\//, "").replace(/\/\?key=.*$/, "");
      const lock = info.key_required ? " 🔒" : "";
      const lan = (info.share_lan || [])[0];
      paintUrl("url-pill", lan ? bare(lan) + lock : "", lan);
      const pub = info.share_public;
      paintUrl("url-pill-pub",
               pub ? "🌐 " + bare(pub) + lock : (info.public_on ? "🌐 여는 중…" : ""),
               pub || (info.public_on ? window.location.origin : ""));
    }).catch(() => {});
  }
  refreshUrls();
  setInterval(refreshUrls, 10000);
})();
