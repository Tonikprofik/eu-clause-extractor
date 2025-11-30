# Playwright E2E Test Plan

## Overview

End-to-end tests for **EU Clause Extractor** — covering UI interactions and API contract testing.

```
tests/
├── e2e/
│   ├── ui.spec.ts          # UI component & interaction tests
│   └── api.spec.ts         # API contract & integration tests
├── fixtures/
│   └── sample-regulation.ts # Test data
└── playwright.config.ts
```

---

## UI Tests (`ui.spec.ts`)

### 1. Page Load & Hero Section
```typescript
test('displays hero section with correct title', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('h1')).toContainText('EU Clause Extractor');
  await expect(page.getByText('MSc Thesis Project')).toBeVisible();
});

test('shows technology badges (ChromaDB, LiteLLM, Langfuse)', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('ChromaDB')).toBeVisible();
  await expect(page.getByText('LiteLLM')).toBeVisible();
  await expect(page.getByText('Langfuse')).toBeVisible();
});
```

### 2. Form Interactions
```typescript
test('loads sample text when clicking Load Sample button', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Load Sample' }).click();
  const textarea = page.locator('#documentText');
  await expect(textarea).toContainText('Article 1');
  await expect(textarea).toContainText('Subject matter and scope');
});

test('toggles clause type checkboxes', async ({ page }) => {
  await page.goto('/');
  const checkbox = page.getByLabel('Definitions');
  await expect(checkbox).toBeChecked();
  await checkbox.click();
  await expect(checkbox).not.toBeChecked();
});

test('expands advanced settings', async ({ page }) => {
  await page.goto('/');
  await page.getByText('Advanced Settings').click();
  await expect(page.getByLabel('Embedding Model')).toBeVisible();
  await expect(page.getByLabel('Enable HyDE')).toBeVisible();
});
```

### 3. Extraction Flow (Mocked API)
```typescript
test('shows loading state during extraction', async ({ page }) => {
  // Mock slow API response
  await page.route('**/api/v1/extract-clauses', async route => {
    await new Promise(r => setTimeout(r, 1000));
    await route.fulfill({ json: mockResponse });
  });
  
  await page.goto('/');
  await page.getByRole('button', { name: 'Extract Clauses' }).click();
  await expect(page.getByText('Extracting clauses...')).toBeVisible();
});

test('displays extracted clauses grouped by type', async ({ page }) => {
  await page.route('**/api/v1/extract-clauses', route => route.fulfill({
    json: {
      predicted_annotations: [
        { clause_type: 'Definitions', clause_text: 'personal data means...' },
        { clause_type: 'Penalties', clause_text: 'Member States shall...' },
      ],
      retrieved_chunks: ['chunk1', 'chunk2'],
      trace_id: 'abc123',
    }
  }));
  
  await page.goto('/');
  await page.getByRole('button', { name: 'Extract Clauses' }).click();
  
  await expect(page.getByText('Definitions')).toBeVisible();
  await expect(page.getByText('personal data means...')).toBeVisible();
  await expect(page.getByText('Trace: abc123')).toBeVisible();
});

test('displays error message on API failure', async ({ page }) => {
  await page.route('**/api/v1/extract-clauses', route => 
    route.fulfill({ status: 500, body: 'Internal Server Error' })
  );
  
  await page.goto('/');
  await page.getByRole('button', { name: 'Extract Clauses' }).click();
  await expect(page.locator('.border-destructive')).toBeVisible();
});
```

### 4. Footer & Links
```typescript
test('footer links to GitHub repo', async ({ page }) => {
  await page.goto('/');
  const githubLink = page.getByRole('link', { name: 'GitHub' });
  await expect(githubLink).toHaveAttribute(
    'href', 
    'https://github.com/Tonikprofik/eu-clause-extractor'
  );
});
```

---

## API Tests (`api.spec.ts`)

### 1. Health Endpoint
```typescript
test('GET /api/v1/health returns status', async ({ request }) => {
  const res = await request.get('/api/v1/health');
  expect(res.status()).toBe(200);
  const body = await res.json();
  expect(body).toHaveProperty('status');
  expect(body).toHaveProperty('litellm');
  expect(body.defaults).toHaveProperty('reader_model');
});
```

### 2. Models Endpoint
```typescript
test('GET /api/v1/models returns model lists', async ({ request }) => {
  const res = await request.get('/api/v1/models');
  expect(res.status()).toBe(200);
  const body = await res.json();
  expect(Array.isArray(body.reader_models)).toBe(true);
  expect(Array.isArray(body.embedding_models)).toBe(true);
  expect(body.defaults).toHaveProperty('reader_model');
});
```

### 3. Extract Clauses Endpoint
```typescript
test('POST /api/v1/extract-clauses with valid payload', async ({ request }) => {
  const res = await request.post('/api/v1/extract-clauses', {
    data: {
      document_id: 'test-doc',
      document_text: 'Article 1\nThis regulation defines personal data...',
      user_query: 'Extract definitions',
      clause_types: ['Definitions'],
      language: 'en',
      options: {
        use_hyde: false,
        top_k: 3,
        reader_model: 'claude-3-7',
      }
    }
  });
  expect(res.status()).toBe(200);
  const body = await res.json();
  expect(body).toHaveProperty('predicted_annotations');
  expect(Array.isArray(body.predicted_annotations)).toBe(true);
  expect(body).toHaveProperty('retrieved_chunks');
  expect(body).toHaveProperty('model_info');
});

test('POST /api/v1/extract-clauses rejects empty document', async ({ request }) => {
  const res = await request.post('/api/v1/extract-clauses', {
    data: {
      document_id: '',
      document_text: '',
      user_query: 'test',
    }
  });
  // Expect 422 validation error or 500
  expect([422, 500]).toContain(res.status());
});
```

### 4. Metrics Endpoint
```typescript
test('GET /metrics returns Prometheus format', async ({ request }) => {
  const res = await request.get('/metrics');
  expect(res.status()).toBe(200);
  const text = await res.text();
  expect(text).toContain('http_request_duration_seconds');
  expect(text).toContain('rag_stage_duration_seconds');
});
```

---

## Setup

### Install
```bash
cd ui
npm install -D @playwright/test
npx playwright install
```

### Config (`playwright.config.ts`)
```typescript
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  retries: 2,
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  webServer: [
    {
      command: 'npm run dev',
      url: 'http://localhost:3000',
      reuseExistingServer: !process.env.CI,
    },
    {
      command: 'cd ../api && uvicorn main:app --port 8080',
      url: 'http://localhost:8080/api/v1/health',
      reuseExistingServer: !process.env.CI,
    },
  ],
});
```

### Run
```bash
# All tests
npx playwright test

# UI only
npx playwright test ui.spec.ts

# API only  
npx playwright test api.spec.ts

# With UI (debug)
npx playwright test --ui
```

---

## Test Matrix

| Area | Test | Priority | Mocked? |
|------|------|----------|---------|
| UI - Hero | Title, badges visible | P1 | No |
| UI - Form | Load sample, toggle checkboxes | P1 | No |
| UI - Advanced | Expand settings, HyDE toggle | P2 | No |
| UI - Extraction | Loading state, results display | P1 | Yes (mock API) |
| UI - Error | Error card on 500 | P1 | Yes |
| UI - Footer | GitHub link correct | P2 | No |
| API - Health | Returns status + defaults | P1 | No |
| API - Models | Returns model arrays | P1 | No |
| API - Extract | Valid request → clauses | P1 | No (real API) |
| API - Metrics | Prometheus format | P2 | No |

---

## CI Integration

```yaml
# .github/workflows/e2e.yml
name: E2E Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20 }
      - name: Install UI deps
        run: cd ui && npm ci
      - name: Install Playwright
        run: cd ui && npx playwright install --with-deps
      - name: Start API (mock mode)
        run: |
          pip install fastapi uvicorn pydantic
          cd api && uvicorn main:app --port 8080 &
      - name: Run tests
        run: cd ui && npx playwright test
      - uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: playwright-report
          path: ui/playwright-report/
```
