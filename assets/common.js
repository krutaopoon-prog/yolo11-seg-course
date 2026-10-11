/* ฟังก์ชันที่ใช้ร่วมทุกบท
 *  - pyRound, letterbox     : คำนวณแบบเดียวกับ ultralytics (ตรวจกับ tools/verify_letterbox.py)
 *  - drawFruitBasket        : ภาพตัวอย่างประจำรายวิชา "ผลไม้ในตะกร้า"
 *  - renderPipeline         : ผัง 🔗 อยู่ตรงไหนใน YOLO11-seg
 *  - initQuiz               : แบบทดสอบท้ายบท
 */

/* ---------- ตัวเลข ---------- */

// round() ของ Python ปัดครึ่งไปหาเลขคู่ (banker's rounding) ต่างจาก Math.round ของ JS
function pyRound(x) {
  const f = Math.floor(x);
  if (x - f === 0.5) return f % 2 === 0 ? f : f + 1;
  return Math.round(x);
}

// เลียนแบบ LetterBox.__call__ ใน ultralytics/data/augment.py (8.4.2) บรรทัดต่อบรรทัด
// mode: 'square' (auto=False), 'auto' (auto=True, ใช้ตอน model.predict กับไฟล์ .pt), 'stretch' (ยืดตรง ๆ เพื่อเปรียบเทียบ)
function letterbox(w, h, mode = 'square', newShape = 640, stride = 32) {
  const r = Math.min(newShape / h, newShape / w);
  let rx = r, ry = r;
  let newW = pyRound(w * r), newH = pyRound(h * r);
  let dw = newShape - newW, dh = newShape - newH;
  const dwFull = dw, dhFull = dh;
  if (mode === 'auto') {
    dw = dw % stride; dh = dh % stride;
  } else if (mode === 'stretch') {
    dw = 0; dh = 0; newW = newShape; newH = newShape;
    rx = newShape / w; ry = newShape / h;
  }
  dw /= 2; dh /= 2;
  const top = pyRound(dh - 0.1), bottom = pyRound(dh + 0.1);
  const left = pyRound(dw - 0.1), right = pyRound(dw + 0.1);
  return {
    r, rx, ry, newW, newH, dwFull, dhFull, top, bottom, left, right,
    outW: newW + left + right, outH: newH + top + bottom,
  };
}

// ตัวสุ่มแบบกำหนด seed ได้ (mulberry32) ให้ซิมได้ข้อมูลชุดเดิมทุกครั้งที่เปิด
function rng(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/* ---------- ภาพตัวอย่างประจำรายวิชา ---------- */

// ผลไม้แต่ละลูกในพิกัดหน่วย (0–1) ของ "กรอบฉาก" สี่เหลี่ยมจัตุรัส เรียงจากหลังมาหน้า
const FRUITS = [
  { id: 'banana', name: 'กล้วย',      kind: 'banana' },
  { id: 'grapes', name: 'องุ่น',      kind: 'grapes', cx: 0.72, cy: 0.43, r: 0.12 },
  { id: 'orange', name: 'ส้ม',        kind: 'round', cx: 0.33, cy: 0.53, r: 0.135, c0: '#ffc46b', c1: '#f08a1c', c2: '#b85a08' },
  { id: 'apple1', name: 'แอปเปิลแดง', kind: 'apple', cx: 0.55, cy: 0.55, r: 0.13,  c0: '#ff8a80', c1: '#d8262b', c2: '#7d0f12' },
  { id: 'lime',   name: 'แอปเปิลเขียว', kind: 'apple', cx: 0.75, cy: 0.60, r: 0.105, c0: '#d9f59a', c1: '#7cbf2a', c2: '#3f7010' },
  { id: 'apple2', name: 'แอปเปิลแดง', kind: 'apple', cx: 0.43, cy: 0.67, r: 0.11,  c0: '#ff9e8f', c1: '#c92d2a', c2: '#6e1111' },
];

// กรอบฉากสี่เหลี่ยมจัตุรัสภายในภาพ w×h: X(u), Y(v) แปลงพิกัดหน่วยเป็นพิกเซล, L(d) แปลงความยาว
function sceneFrame(w, h) {
  const m = Math.min(w, h), S = m * 0.92;
  const ox = (w - S) / 2, oy = Math.min(h - S - m * 0.02, h * 0.72 - S * 0.55);
  return { X: u => ox + u * S, Y: v => oy + v * S, L: d => d * S };
}

// วาดฉากผลไม้ในตะกร้าลงในพื้นที่ (0,0)–(w,h) ของ ctx
// ฉากปรับตามสัดส่วนภาพ: ผนัง/โต๊ะเต็มกรอบ ตะกร้าอยู่กลางล่าง
function drawFruitBasket(ctx, w, h) {
  ctx.save();
  // ผนัง
  let g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, '#e9e2d3'); g.addColorStop(1, '#d8cdb8');
  ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
  // โต๊ะไม้
  const tableY = h * 0.72;
  g = ctx.createLinearGradient(0, tableY, 0, h);
  g.addColorStop(0, '#a8774a'); g.addColorStop(1, '#7a5230');
  ctx.fillStyle = g; ctx.fillRect(0, tableY, w, h - tableY);
  ctx.strokeStyle = 'rgba(60,35,15,0.25)'; ctx.lineWidth = Math.max(1, h * 0.004);
  for (let i = 1; i < 5; i++) {
    const y = tableY + (h - tableY) * i / 5;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  const { X, Y, L } = sceneFrame(w, h);

  // เงาตะกร้า
  ctx.fillStyle = 'rgba(40,20,5,0.25)';
  ctx.beginPath(); ctx.ellipse(X(0.5), Y(0.93), L(0.36), L(0.04), 0, 0, Math.PI * 2); ctx.fill();

  // ขอบหลังของตะกร้า
  ctx.fillStyle = '#6b4321';
  ctx.beginPath(); ctx.ellipse(X(0.5), Y(0.64), L(0.38), L(0.07), 0, Math.PI, Math.PI * 2); ctx.fill();

  for (const f of FRUITS) drawFruit(ctx, f, X, Y, L);

  // ตัวตะกร้า (ด้านหน้า บังผลไม้ส่วนล่าง)
  ctx.beginPath();
  ctx.moveTo(X(0.12), Y(0.64));
  ctx.lineTo(X(0.88), Y(0.64));
  ctx.lineTo(X(0.78), Y(0.92));
  ctx.lineTo(X(0.22), Y(0.92));
  ctx.closePath();
  g = ctx.createLinearGradient(X(0.12), 0, X(0.88), 0);
  g.addColorStop(0, '#7a4c22'); g.addColorStop(0.45, '#b37a3e'); g.addColorStop(1, '#6e431d');
  ctx.fillStyle = g; ctx.fill();
  ctx.save(); ctx.clip();
  ctx.strokeStyle = 'rgba(50,28,8,0.45)'; ctx.lineWidth = L(0.008);
  for (let i = 0; i < 6; i++) {               // ลายสานแนวนอน
    const v = 0.67 + i * 0.05;
    ctx.beginPath(); ctx.moveTo(X(0.1), Y(v)); ctx.lineTo(X(0.9), Y(v)); ctx.stroke();
  }
  for (let i = 0; i <= 12; i++) {             // ลายสานแนวตั้ง (เอียงตามทรงตะกร้า)
    const t = i / 12;
    ctx.beginPath(); ctx.moveTo(X(0.12 + 0.76 * t), Y(0.64)); ctx.lineTo(X(0.22 + 0.56 * t), Y(0.92)); ctx.stroke();
  }
  ctx.restore();
  // ขอบปากตะกร้า
  ctx.strokeStyle = '#5a3716'; ctx.lineWidth = L(0.025);
  ctx.beginPath(); ctx.moveTo(X(0.12), Y(0.64)); ctx.lineTo(X(0.88), Y(0.64)); ctx.stroke();
  ctx.restore();
}

function drawFruit(ctx, f, X, Y, L) {
  if (f.kind === 'banana') {
    ctx.save();
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#a07a10'; ctx.lineWidth = L(0.11);
    ctx.beginPath(); ctx.moveTo(X(0.16), Y(0.40)); ctx.quadraticCurveTo(X(0.36), Y(0.52), X(0.62), Y(0.30)); ctx.stroke();
    ctx.strokeStyle = '#f3cf3a'; ctx.lineWidth = L(0.095);
    ctx.stroke();
    ctx.strokeStyle = '#fbe78a'; ctx.lineWidth = L(0.03);
    ctx.beginPath(); ctx.moveTo(X(0.19), Y(0.385)); ctx.quadraticCurveTo(X(0.37), Y(0.48), X(0.59), Y(0.295)); ctx.stroke();
    ctx.fillStyle = '#4a3a10';
    ctx.beginPath(); ctx.arc(X(0.62), Y(0.30), L(0.022), 0, Math.PI * 2); ctx.fill();
    ctx.restore();
    return;
  }
  if (f.kind === 'grapes') {
    const pts = [[0, -0.07], [-0.05, -0.045], [0.05, -0.045], [-0.075, 0], [0, -0.01], [0.075, 0],
                 [-0.045, 0.045], [0.045, 0.045], [0, 0.08], [0.0, 0.035]];
    for (const [du, dv] of pts) {
      const cx = X(f.cx + du), cy = Y(f.cy + dv), rr = L(0.042);
      const g = ctx.createRadialGradient(cx - rr * 0.35, cy - rr * 0.35, rr * 0.1, cx, cy, rr);
      g.addColorStop(0, '#b48ad6'); g.addColorStop(0.7, '#6a3d94'); g.addColorStop(1, '#3d1f5c');
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, rr, 0, Math.PI * 2); ctx.fill();
    }
    ctx.strokeStyle = '#5b4a1d'; ctx.lineWidth = L(0.012);
    ctx.beginPath(); ctx.moveTo(X(f.cx), Y(f.cy - 0.1)); ctx.lineTo(X(f.cx + 0.02), Y(f.cy - 0.15)); ctx.stroke();
    return;
  }
  const cx = X(f.cx), cy = Y(f.cy), rr = L(f.r);
  const g = ctx.createRadialGradient(cx - rr * 0.4, cy - rr * 0.45, rr * 0.08, cx, cy, rr * 1.05);
  g.addColorStop(0, f.c0); g.addColorStop(0.55, f.c1); g.addColorStop(1, f.c2);
  ctx.fillStyle = g;
  ctx.beginPath();
  if (f.kind === 'apple') {
    // แอปเปิล: วงกลมที่มีรอยบุ๋มด้านบน
    ctx.moveTo(cx, cy - rr * 0.72);
    ctx.bezierCurveTo(cx + rr * 0.55, cy - rr * 1.15, cx + rr * 1.15, cy - rr * 0.55, cx + rr, cy + rr * 0.1);
    ctx.bezierCurveTo(cx + rr * 0.9, cy + rr * 0.85, cx + rr * 0.35, cy + rr * 1.05, cx, cy + rr * 0.92);
    ctx.bezierCurveTo(cx - rr * 0.35, cy + rr * 1.05, cx - rr * 0.9, cy + rr * 0.85, cx - rr, cy + rr * 0.1);
    ctx.bezierCurveTo(cx - rr * 1.15, cy - rr * 0.55, cx - rr * 0.55, cy - rr * 1.15, cx, cy - rr * 0.72);
  } else {
    ctx.arc(cx, cy, rr, 0, Math.PI * 2);
  }
  ctx.fill();
  if (f.kind === 'apple') {
    ctx.strokeStyle = '#4b2e12'; ctx.lineWidth = rr * 0.09; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(cx, cy - rr * 0.65); ctx.lineTo(cx + rr * 0.08, cy - rr * 1.05); ctx.stroke();
    ctx.fillStyle = '#4f9a2f';
    ctx.beginPath(); ctx.ellipse(cx + rr * 0.3, cy - rr * 0.95, rr * 0.25, rr * 0.11, -0.5, 0, Math.PI * 2); ctx.fill();
  } else {
    ctx.fillStyle = 'rgba(90,50,0,0.5)';
    ctx.beginPath(); ctx.arc(cx + rr * 0.1, cy - rr * 0.82, rr * 0.07, 0, Math.PI * 2); ctx.fill();
  }
}

// สร้าง canvas นอกจอที่วาดภาพตัวอย่างไว้ ขนาด w×h พิกเซลจริง
function makeFruitImage(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  drawFruitBasket(c.getContext('2d'), w, h);
  return c;
}

// พิกเซลจริงจากภาพผลไม้ สำหรับโจทย์แยกสีในบทที่ 2–3 (ลำดับการสุ่มเดียวกับซิม 2.1 จึงได้จุดชุดเดียวกัน)
// คืนค่า { red, green, orange, yellow } แต่ละตัวเป็นอาร์เรย์ของ { r, g, c } โดย r, g อยู่ในช่วง 0–1
function fruitPixels() {
  const N = 320, img = makeFruitImage(N, N).getContext('2d').getImageData(0, 0, N, N).data;
  const { X, Y } = sceneFrame(N, N), rand = rng(7);
  const px = (u, v) => {
    const x = Math.round(X(u)), y = Math.round(Y(v)), i = (y * N + x) * 4;
    const c = [img[i], img[i + 1], img[i + 2]];
    return { r: c[0] / 255, g: c[1] / 255, c };
  };
  const disc = (u, v, r, n) => Array.from({ length: n }, () => {
    const a = rand() * Math.PI * 2, d = Math.sqrt(rand()) * r;
    return px(u + Math.cos(a) * d, v + Math.sin(a) * d);
  });
  const banana = n => Array.from({ length: n }, () => {
    const t = 0.65 + rand() * 0.2, s = 1 - t;
    const u = s * s * 0.16 + 2 * s * t * 0.36 + t * t * 0.62, v = s * s * 0.40 + 2 * s * t * 0.52 + t * t * 0.30;
    return px(u + (rand() - 0.5) * 0.04, v + (rand() - 0.5) * 0.04);
  });
  const red = [...disc(0.58, 0.55, 0.045, 20), ...disc(0.40, 0.60, 0.025, 10)];
  const green = disc(0.79, 0.57, 0.035, 30);
  const orange = disc(0.28, 0.56, 0.045, 30);
  const yellow = banana(20);
  return { red, green, orange, yellow };
}

/* ---------- ตัวช่วยวาดกราฟ ---------- */

const logistic = z => 1 / (1 + Math.exp(-z));

// ทำให้ canvas คมบนจอความละเอียดสูง โดยยังวาดด้วยพิกัดตามขนาดเดิม
function setupCanvas(cv) {
  const r = window.devicePixelRatio || 1, w = cv.width, h = cv.height;
  if (r !== 1 && !cv.dataset.dpi) {
    cv.width = w * r; cv.height = h * r; cv.dataset.dpi = w + 'x' + h;
    cv.getContext('2d').scale(r, r);
  }
  const [W, H] = cv.dataset.dpi ? cv.dataset.dpi.split('x').map(Number) : [w, h];
  return { ctx: cv.getContext('2d'), W, H };
}

// แปลงตำแหน่งเมาส์/นิ้วเป็นพิกัดของ canvas (ขนาด W×H)
function canvasXY(cv, e, W, H) {
  const rect = cv.getBoundingClientRect();
  return [(e.clientX - rect.left) / rect.width * W, (e.clientY - rect.top) / rect.height * H];
}

/* ---------- ผัง 🔗 อยู่ตรงไหนใน YOLO11-seg ---------- */

const PIPELINE_STEPS = [
  ['image', 'ภาพ'], ['letterbox', 'Letterbox 640'], ['backbone', 'Backbone'], ['neck', 'Neck'],
  ['head', 'Head + Proto'], ['decode', 'ถอดรหัส (decode)'], ['conf', 'กรอง conf'], ['nms', 'NMS'],
  ['mask', 'Mask ของวัตถุแต่ละชิ้น'],
];

function renderPipeline(el, activeIds) {
  el.classList.add('pipeline');
  el.innerHTML = PIPELINE_STEPS.map(([id, label], i) =>
    (i ? '<span class="arrow">→</span>' : '') +
    `<span class="step${activeIds.includes(id) ? ' on' : ''}">${label}</span>`
  ).join('');
}

/* ---------- แบบทดสอบ ---------- */
// questions: [{ q, options: [..4], answer: index, explain }]
function initQuiz(el, questions) {
  let answered = 0, correct = 0;
  const score = document.createElement('p');
  score.className = 'quiz-score';
  const update = () => {
    score.textContent = `ตอบถูก ${correct} จาก ${answered} ข้อที่ตอบแล้ว (ทั้งหมด ${questions.length} ข้อ)`;
  };

  questions.forEach((item, qi) => {
    const box = document.createElement('div');
    box.className = 'quiz-q';
    box.innerHTML = `<p class="q">${qi + 1}. ${item.q}</p><ol></ol><div class="explain"></div>`;
    const ol = box.querySelector('ol');
    item.options.forEach((opt, oi) => {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.innerHTML = `${'กขคง'[oi]}. ${opt}`;
      btn.addEventListener('click', () => {
        if (box.classList.contains('done')) return;
        box.classList.add('done');
        answered++;
        const ok = oi === item.answer;
        if (ok) correct++;
        btn.classList.add(ok ? 'right' : 'wrong');
        ol.children[item.answer].firstChild.classList.add('right');
        box.querySelector('.explain').innerHTML = (ok ? '✅ ถูกต้อง — ' : '❌ ยังไม่ถูก — ') + item.explain;
        update();
      });
      li.appendChild(btn);
      ol.appendChild(li);
    });
    el.appendChild(box);
  });

  const again = document.createElement('button');
  again.className = 'reset';
  again.textContent = 'ทำใหม่ทั้งหมด';
  again.addEventListener('click', () => { el.innerHTML = ''; initQuiz(el, questions); });
  update();
  el.appendChild(score);
  el.appendChild(again);
}
