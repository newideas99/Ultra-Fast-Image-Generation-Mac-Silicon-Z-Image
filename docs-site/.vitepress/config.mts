import { defineConfig } from 'vitepress'
import path from 'path'

export default defineConfig({
  title: 'Ultra Fast Image Gen',
  description: 'AI image generation and editing on Mac Silicon and CUDA',
  base: '/ultra-fast-image-gen/',
  ignoreDeadLinks: true,
  vite: {
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './'),
      },
    },
  },
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/quick-start' },
      { text: 'MCP', link: '/guide/mcp' },
      { text: 'GitHub', link: 'https://github.com/newideas99/ultra-fast-image-gen' }
    ],
    sidebar: {
      '/guide/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Quick Start', link: '/guide/quick-start' },
            { text: 'Supported Models', link: '/guide/models' }
          ]
        },
        {
          text: 'Usage',
          items: [
            { text: 'Web UI & CLI', link: '/guide/usage' },
            { text: 'MCP Server', link: '/guide/mcp' },
            { text: 'Benchmarks', link: '/guide/benchmarks' }
          ]
        },
        {
          text: 'More',
          items: [
            { text: 'Credits & License', link: '/guide/credits' }
          ]
        }
      ]
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/newideas99/ultra-fast-image-gen' }
    ],
    footer: {
      message: 'Released under the original model licenses.',
      copyright: 'Copyright © 2024 Ultra Fast Image Gen'
    }
  }
})
