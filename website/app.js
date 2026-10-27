'use strict';
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

// Sequence player: one button per rendered sequence.
(async () => {
  const video = document.querySelector('#seq-video');
  let data;
  try {
    const response = await fetch('assets/figures/sequences.json');
    if (!response.ok) throw new Error(response.statusText);
    data = await response.json();
  } catch {
    video.src = 'assets/videos/office_Hard_P001.mp4'; video.poster = 'assets/videos/office_Hard_P001.jpg';
    return;
  }
  const buttons = [];
  function select(entry, play) {
    for (const b of buttons) b.setAttribute('aria-pressed', String(b.dataset.seq === entry.sequence));
    video.poster = `assets/videos/${entry.sequence}.jpg`;
    video.src = `assets/videos/${entry.sequence}.mp4`;
    video.setAttribute('aria-label', `${entry.label}: raw events and estimated 3D trajectory`);
    if (play && !reduceMotion) video.play().catch(() => {});
  }
  for (const entry of data.sequences) {
    const [, difficulty, name] = entry.sequence.split('_');
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = name; b.dataset.seq = entry.sequence; b.setAttribute('aria-label', entry.label);
    b.addEventListener('click', () => select(entry, true));
    document.querySelector(`#pick-${difficulty}`).append(b); buttons.push(b);
  }
  select(data.sequences.find(s => s.sequence === 'office_Hard_P001') || data.sequences[0], false);

  if (!reduceMotion && 'IntersectionObserver' in window) {
    new IntersectionObserver(([e]) => { if (e.isIntersecting) video.play().catch(() => {}); else video.pause(); }, {threshold: .5}).observe(video);
  }
})();

document.querySelector('#copy-bib').addEventListener('click', async event => {
  try { await navigator.clipboard.writeText(document.querySelector('#bib-text').textContent); event.target.textContent = 'Copied'; }
  catch { event.target.textContent = 'Select and copy'; }
  setTimeout(() => { event.target.textContent = 'Copy'; }, 1600);
});
