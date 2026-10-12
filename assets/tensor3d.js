/* ฉาก 3 มิติที่ใช้ร่วมกันทุกบท (ต้องโหลด three.js r128 ก่อนไฟล์นี้)
 *  - ลากเพื่อหมุน, ปุ่ม data-view3d="iso|front|top", data-zoom3d="in|out"
 *  - ป้ายข้อความ HTML ที่ตามตำแหน่ง 3 มิติ
 *  - จัดกล้องให้ฉากพอดีจออัตโนมัติจากขอบเขตที่กำหนด (setBounds)
 *  - คลิก (ไม่ลาก) แล้วเรียก onPick(hit) จากการยิงรังสีใส่วัตถุในฉาก
 * ชื่อที่ประกาศ: Scene3D
 */
class Scene3D {
  constructor(host, opt = {}) {
    this.host = host;
    this.T = window.THREE;
    this.ok = !!this.T;
    this.views = Object.assign({ iso: [-0.62, 1.08], front: [0, Math.PI / 2], top: [-0.15, 0.25] }, opt.views || {});
    const v0 = this.views[opt.view || 'iso'];
    this.view = { th: v0[0], ph: v0[1], zoom: 1 };
    this.heights = opt.heights || [320, 400];   // [จอแคบ, จอกว้าง]
    this.fitScale = opt.fitScale || 1.08;
    this.bounds = null; this.labelDefs = []; this.labelEls = [];
    this.onPick = null; this.pickables = null;
    Object.assign(host.style, { position: 'relative', overflow: 'hidden', touchAction: 'none', cursor: 'grab' });
    if (!this.ok) {
      host.innerHTML = '<p style="padding:12px">⚠ โหลด three.js ไม่ได้ (ต้องต่ออินเทอร์เน็ตครั้งแรก) ภาพ 3 มิติจึงไม่แสดง</p>';
      host.style.height = 'auto';
      return;
    }
    const T = this.T;
    this.renderer = new T.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    host.appendChild(this.renderer.domElement);
    this.scene = new T.Scene(); this.scene.background = new T.Color(0xffffff);
    this.cam = new T.PerspectiveCamera(35, 2, 0.05, 5000);
    this.scene.add(new T.AmbientLight(0xffffff, 0.72));
    const dl = new T.DirectionalLight(0xffffff, 0.55); dl.position.set(4, 9, 7); this.scene.add(dl);
    this.root = new T.Group(); this.scene.add(this.root);
    this._m4 = new T.Matrix4(); this._q = new T.Quaternion(); this._s = new T.Vector3(); this._p = new T.Vector3();

    let drag = null, moved = 0;
    host.addEventListener('pointerdown', e => {
      drag = [e.clientX, e.clientY]; moved = 0; host.setPointerCapture(e.pointerId); host.style.cursor = 'grabbing';
    });
    host.addEventListener('pointermove', e => {
      if (!drag) return;
      const dx = e.clientX - drag[0], dy = e.clientY - drag[1];
      moved += Math.abs(dx) + Math.abs(dy);
      this.view.th -= dx * 0.008;
      this.view.ph = Math.max(0.08, Math.min(Math.PI - 0.08, this.view.ph - dy * 0.008));
      drag = [e.clientX, e.clientY]; this.render();
    });
    host.addEventListener('pointerup', e => {
      host.style.cursor = 'grab';
      if (drag && moved < 6 && this.onPick) this.onPick(this.pick(e));
      drag = null;
    });
    if (window.ResizeObserver) new ResizeObserver(() => this.render()).observe(host);
  }

  /* ---------- เนื้อหาในฉาก ---------- */
  clear() {
    if (!this.ok) return;
    this.root.traverse(o => {
      if (o.geometry && !o.geometry.userData.shared) o.geometry.dispose();
      if (o.material) { (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => { if (m.map) m.map.dispose(); m.dispose(); }); }
    });
    this.scene.remove(this.root);
    this.root = new this.T.Group(); this.scene.add(this.root);
    this.pickables = null;
  }
  add(o) { this.root.add(o); return o; }
  // ลูกบาศก์หลายลูกในวัตถุเดียว (InstancedMesh) ขนาด size, สีตั้งทีละลูกด้วย color()
  boxes(n, size = 0.82, mat = {}) {
    const T = this.T;
    if (!this._cube || this._cubeSize !== size) { this._cube = new T.BoxGeometry(size, size, size); this._cube.userData.shared = true; this._cubeSize = size; }
    const m = new T.InstancedMesh(this._cube, new T.MeshLambertMaterial(mat), Math.max(n, 1));
    m.count = n; return this.add(m);
  }
  place(mesh, i, x, y, z, s = 1) {
    this._p.set(x, y, z); this._s.set(s, s, s);
    this._m4.compose(this._p, this._q, this._s); mesh.setMatrixAt(i, this._m4);
  }
  outline(w, h, d, color = 0xff00aa) {
    const T = this.T;
    return this.add(new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(w, h, d)), new T.LineBasicMaterial({ color })));
  }
  line(points, color = 0x888888) {
    const T = this.T, g = new T.BufferGeometry().setFromPoints(points.map(p => new T.Vector3(...p)));
    return this.add(new T.Line(g, new T.LineBasicMaterial({ color })));
  }
  // เปลี่ยนจุดของเส้น (ต้องคำนวณทรงกลมครอบใหม่ ไม่งั้นกล้องอาจคิดว่าเส้นอยู่นอกจอแล้วไม่วาด)
  setLine(l, points) { l.geometry.setFromPoints(points.map(p => new this.T.Vector3(...p))); l.geometry.computeBoundingSphere(); }

  /* ---------- กล้องและป้าย ---------- */
  setBounds(lo, hi) { this.bounds = { lo, hi }; }
  setLabels(defs) {
    this.labelDefs = defs;
    while (this.labelEls.length < defs.length) {
      const d = document.createElement('div');
      d.style.cssText = 'position:absolute;transform:translate(-50%,-100%);font-size:13px;font-weight:700;background:rgba(255,255,255,0.85);padding:0 4px;border-radius:3px;pointer-events:none;white-space:nowrap';
      this.host.appendChild(d); this.labelEls.push(d);
    }
    this.labelEls.forEach((el, i) => { el.style.display = i < defs.length ? '' : 'none'; });
  }
  setView(name) { const v = this.views[name]; if (v) { [this.view.th, this.view.ph] = v; this.render(); } }
  zoomBy(f) { this.view.zoom *= f; this.render(); }
  bindButtons(container) {
    container.querySelectorAll('[data-view3d]').forEach(b => b.addEventListener('click', () => this.setView(b.dataset.view3d)));
    container.querySelectorAll('[data-zoom3d]').forEach(b => b.addEventListener('click', () => this.zoomBy(b.dataset.zoom3d === 'in' ? 1.25 : 0.8)));
  }
  render() {
    if (!this.ok) return;
    const W = this.host.clientWidth || 600, H = W < 520 ? this.heights[0] : this.heights[1];
    this.host.style.height = H + 'px';
    this.renderer.setSize(W, H); this.cam.aspect = W / H; this.cam.updateProjectionMatrix();
    const { th, ph } = this.view;
    let ctr = [0, 0, 0], d = 30;
    if (this.bounds) {
      const { lo, hi } = this.bounds, sz = [0, 1, 2].map(k => hi[k] - lo[k]);
      ctr = [0, 1, 2].map(k => (lo[k] + hi[k]) / 2);
      // ขนาดที่เห็นบนจอเมื่อมองจากมุมปัจจุบัน (รวมความลึกที่เอียงเข้ามา)
      const ct = Math.abs(Math.cos(th)), st = Math.abs(Math.sin(th)), cp = Math.abs(Math.cos(ph)), sp = Math.abs(Math.sin(ph));
      const vf = this.cam.fov * Math.PI / 180, hf = 2 * Math.atan(Math.tan(vf / 2) * this.cam.aspect);
      const w = sz[0] * ct + sz[2] * st, h = sz[1] * sp + (sz[0] * st + sz[2] * ct) * cp, deep = sz[0] * st + sz[2] * ct;
      d = Math.max(w / 2 / Math.tan(hf / 2), h / 2 / Math.tan(vf / 2)) * this.fitScale + deep / 2;
    }
    d /= this.view.zoom;
    this.cam.position.set(ctr[0] + d * Math.sin(ph) * Math.sin(th), ctr[1] + d * Math.cos(ph), ctr[2] + d * Math.sin(ph) * Math.cos(th));
    this.cam.lookAt(ctr[0], ctr[1], ctr[2]);
    this.renderer.render(this.scene, this.cam);
    this.labelDefs.forEach((L, i) => {
      const p = new this.T.Vector3(...L.pos).project(this.cam), el = this.labelEls[i];
      el.textContent = L.text; el.style.color = L.color || '#1f2933';
      el.style.left = ((p.x + 1) / 2 * W) + 'px'; el.style.top = ((1 - p.y) / 2 * H) + 'px';
      el.style.display = p.z < 1 ? '' : 'none';
    });
  }
  // ยิงรังสีจากตำแหน่งเมาส์ ได้ { object, instanceId, point } หรือ null
  pick(e) {
    const T = this.T, r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new T.Vector2((e.clientX - r.left) / r.width * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    const ray = new T.Raycaster(); ray.setFromCamera(ndc, this.cam);
    // เลือกเฉพาะวัตถุทึบ (เส้นถูกรังสี "โดน" ง่ายเกินจริงเพราะ threshold ของเส้นกว้าง 1 หน่วย)
    const hits = ray.intersectObjects(this.pickables || this.root.children, true).filter(h => h.object.isMesh);
    return hits.length ? hits[0] : null;
  }
}
