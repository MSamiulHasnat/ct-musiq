const contentEl = document.getElementById('content');
const tocList = document.getElementById('tocList');
const docButtons = [...document.querySelectorAll('.doc-btn')];

function slugify(text) {
  return text
    .toLowerCase()
    .trim()
    .replace(/[\s]+/g, '-')
    .replace(/[^a-z0-9\-]/g, '')
    .replace(/\-+/g, '-');
}

marked.use({
  renderer: {
    heading(token) {
      const text = token.text || '';
      const level = token.depth || 1;
      const id = slugify(text);
      return `<h${level} id="${id}">${text}</h${level}>`;
    }
  }
});

function buildToc(container) {
  tocList.innerHTML = '';
  const headings = [...container.querySelectorAll('h1, h2, h3')];

  headings.forEach((h) => {
    if (!h.id) return;
    const levelClass = `lvl-${h.tagName.toLowerCase().replace('h', '')}`;
    const li = document.createElement('li');
    li.className = levelClass;

    const a = document.createElement('a');
    a.href = `#${h.id}`;
    a.textContent = h.textContent;
    li.appendChild(a);
    tocList.appendChild(li);
  });
}

async function loadDoc(fileName) {
  contentEl.innerHTML = '<div class="loading">Loading document...</div>';

  try {
    const resp = await fetch(fileName, { cache: 'no-store' });
    if (!resp.ok) {
      throw new Error(`Failed to load ${fileName}`);
    }

    const markdown = await resp.text();
    const html = marked.parse(markdown);

    const article = document.createElement('article');
    article.innerHTML = html;
    contentEl.innerHTML = '';
    contentEl.appendChild(article);

    buildToc(article);

    const url = new URL(window.location.href);
    url.searchParams.set('doc', fileName);
    window.history.replaceState({}, '', url.toString());
  } catch (err) {
    contentEl.innerHTML = `<div class="loading">Could not load document. ${err.message}</div>`;
    tocList.innerHTML = '';
  }
}

function setActiveButton(fileName) {
  docButtons.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.doc === fileName);
  });
}

function getInitialDoc() {
  const params = new URLSearchParams(window.location.search);
  const doc = params.get('doc');
  if (doc === 'DefensePrep.md') return doc;
  return 'README.md';
}

docButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    const doc = btn.dataset.doc;
    setActiveButton(doc);
    loadDoc(doc);
  });
});

const initialDoc = getInitialDoc();
setActiveButton(initialDoc);
loadDoc(initialDoc);
