// Tema (persistente)
(function(){
  const root = document.documentElement;
  const btn = document.getElementById('themeToggle');
  const stored = localStorage.getItem('ta-theme');
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

  if (stored) root.setAttribute('data-theme', stored);
  else root.setAttribute('data-theme', prefersDark ? 'dark' : 'light');

  if (btn){
    btn.addEventListener('click', () => {
      const current = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', current);
      localStorage.setItem('ta-theme', current);
      btn.setAttribute('aria-pressed', current === 'dark');
    });
  }
})();

// Tilt 3D suave
(function(){
  const max = 10; // grados
  const els = document.querySelectorAll('.tilt');
  els.forEach(el => {
    el.addEventListener('mousemove', (e) => {
      const r = el.getBoundingClientRect();
      const x = e.clientX - r.left, y = e.clientY - r.top;
      const rx = ((y / r.height) - 0.5) * -2 * max;
      const ry = ((x / r.width) - 0.5) *  2 * max;
      el.style.transform = `rotateX(${rx}deg) rotateY(${ry}deg)`;
    });
    el.addEventListener('mouseleave', () => {
      el.style.transform = 'rotateX(0) rotateY(0)';
    });
  });
})();

// Scroll-reveal (IntersectionObserver)
(function(){
  const observer = new IntersectionObserver((entries)=>{
    entries.forEach((entry)=>{
      if(entry.isIntersecting){
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15 });

  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
})();
