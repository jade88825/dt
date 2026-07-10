/* ============================================================
   智慧交通云平台 - 前端交互 v2.0
   ============================================================ */

/**
 * 灯箱功能 - 点击图片放大查看
 */
function openLightbox(src) {
    const lightbox = document.getElementById('lightbox');
    const img = document.getElementById('lightbox-img');
    if (!lightbox || !img) return;
    img.src = src;
    lightbox.style.display = 'flex';
}

function closeLightbox() {
    const lightbox = document.getElementById('lightbox');
    if (lightbox) lightbox.style.display = 'none';
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeLightbox();
});

/**
 * 移动端侧边栏开关
 */
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    if (!sidebar || !overlay) return;
    sidebar.classList.toggle('open');
    overlay.classList.toggle('show');
}

/**
 * 实时时钟
 */
function updateClock() {
    const el = document.getElementById('clock-time');
    if (!el) return;
    const now = new Date();
    const h = String(now.getHours()).padStart(2, '0');
    const m = String(now.getMinutes()).padStart(2, '0');
    const s = String(now.getSeconds()).padStart(2, '0');
    el.textContent = `${h}:${m}:${s}`;
}

setInterval(updateClock, 1000);
updateClock();

/**
 * 标签页切换
 */
function switchTab(tabId, group) {
    const tabs = document.querySelectorAll(`[data-tab-group="${group}"] .tab`);
    const contents = document.querySelectorAll(`[data-tab-group="${group}"] .tab-content`);
    tabs.forEach(t => t.classList.remove('active'));
    contents.forEach(c => c.classList.remove('active'));
    const tab = document.querySelector(`[data-tab-group="${group}"] .tab[data-tab="${tabId}"]`);
    const content = document.querySelector(`[data-tab-group="${group}"] .tab-content[data-content="${tabId}"]`);
    if (tab) tab.classList.add('active');
    if (content) content.classList.add('active');
}
