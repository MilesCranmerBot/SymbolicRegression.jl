import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";

const baseTemp = {
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

const navTemp = {
  nav: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

const nav = [
  ...navTemp.nav.filter(item => item.text === 'Home' || item.text === 'Examples'),
  {
    text: 'API',
    items: navTemp.nav.filter(item => ['API', 'Losses', 'Types', 'Customization', 'Plugins'].includes(item.text))
  },
  {
    text: 'Julia',
    items: [
      { text: 'Julia', link: '/' },
      { text: 'Python', link: 'https://pysr.ai/', target: '_self' }
    ]
  },
  {
    component: 'VersionPicker'
  }
]

// pysr.ai and julia.pysr.ai are separate origins with separate localStorage, so mirror
// VitePress's light/dark choice into a cookie that both sites read before first paint.
const shareAppearanceAcrossSites = `(() => {
  const key = "vitepress-theme-appearance";
  const domain = "pysr.ai";
  const host = location.hostname;
  if (host !== domain && !host.endsWith("." + domain)) return;
  const shared = () => document.cookie.match(/(?:^|; )vitepress-theme-appearance=([^;]*)/)?.[1];
  const saved = shared();
  if (saved) localStorage.setItem(key, saved);
  addEventListener("pageshow", (event) => {
    const value = shared();
    if (event.persisted && value && value !== localStorage.getItem(key)) {
      localStorage.setItem(key, value);
      location.reload();
    }
  });
  addEventListener("DOMContentLoaded", () => {
    // The class also flips when the OS theme changes under "auto"; only a changed stored value is a user choice.
    let last = localStorage.getItem(key);
    new MutationObserver(() => {
      const value = localStorage.getItem(key);
      if (value === last) return;
      last = value;
      document.cookie = key + "=" + value + "; domain=" + domain + "; path=/; max-age=31536000; SameSite=Lax";
    }).observe(document.documentElement, { attributeFilter: ["class"] });
  });
})();`

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  lastUpdated: true,
  cleanUrls: true,
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // This is required for MarkdownVitepress to work correctly...
  head: [
    ['script', {}, shareAppearanceAcrossSites],
    ['link', { rel: 'icon', type: 'image/x-icon', href: 'https://raw.githubusercontent.com/MilesCranmer/PySR/master/docs/src/public/favicon.ico' }],
    ['link', { rel: 'icon', type: 'image/png', sizes: '16x16', href: 'https://raw.githubusercontent.com/MilesCranmer/PySR/master/docs/src/public/favicon-16x16.png' }],
    ['link', { rel: 'icon', type: 'image/png', sizes: '32x32', href: 'https://raw.githubusercontent.com/MilesCranmer/PySR/master/docs/src/public/favicon-32x32.png' }],
    ['link', { rel: 'apple-touch-icon', sizes: '180x180', href: 'https://raw.githubusercontent.com/MilesCranmer/PySR/master/docs/src/public/apple-touch-icon.png' }],
    ['link', { rel: 'stylesheet', href: `${baseTemp.base}nav-logo.css` }],
    ['script', {src: '/versions.js'}],
    ['script', {src: `${baseTemp.base}siteinfo.js`}]
  ],
  ignoreDeadLinks: true,
  vite: {
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('/'),
    },
    optimizeDeps: {
      exclude: [
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ],
    },
    ssr: {
      noExternal: [
        // If there are other packages that need to be processed by Vite, you can add them here.
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ],
    },
  },

  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin),
      md.use(mathjax3),
      md.use(footnote)
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    },
  },
  themeConfig: {
    outline: 'deep',
    // https://vitepress.dev/reference/default-theme-config
    logo: 'https://raw.githubusercontent.com/MilesCranmer/PySR_Docs/master/images/pysr-logo.png',
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav,
    sidebar: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    editLink: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    socialLinks: [
      { icon: 'github', link: 'https://github.com/MilesCranmer/SymbolicRegression.jl' }
    ],
    footer: {
      message: 'Made with <a href="https://documenter.juliadocs.org/stable/" target="_blank"><strong>Documenter.jl</strong></a>, <a href="https://vitepress.dev" target="_blank"><strong>VitePress</strong></a> and <a href="https://luxdl.github.io/DocumenterVitepress.jl/stable/" target="_blank"><strong>DocumenterVitepress.jl</strong></a> <br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    },
  }
})
