# Astro + Tailwind CSS (Static Cloudflare Site)

A static Astro project styled with Tailwind CSS v4, managed using **`nub`**, and pre-configured for Cloudflare Pages static deployment.

## 🚀 Getting Started

### 1. Install Dependencies
```sh
nub install
```

### 2. Development
Start the local dev server:
```sh
nub run dev
```
Open [http://localhost:4321](http://localhost:4321) in your browser.

### 3. Build for Production
Build the optimized static site into the `dist/` directory:
```sh
nub run build
```

### 4. Preview Locally
Preview the production build locally:
```sh
nub run preview
```

---

## ☁️ Cloudflare Deployment

This project is configured as a static site (`output: 'static'` in [astro.config.mjs](file:///D:/code/P2P/astro.config.mjs)).

### Option 1: Direct Deployment with Wrangler CLI
Deploy directly to Cloudflare Pages:
```sh
nub run deploy
```
*(Wrangler will prompt you to authenticate with Cloudflare on first use and select or create your Pages project)*

### Option 2: Cloudflare Dashboard / Git Integration
Connect your Git repository (GitHub/GitLab) to **Cloudflare Pages**:
- **Framework Preset**: `Astro`
- **Build command**: `nub run build` (or `bun run build`)
- **Build output directory**: `dist`
