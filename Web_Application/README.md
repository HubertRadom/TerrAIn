# Web application
The source code of the web application

## How to run frontend

1. CD into folder

```bash
cd frontend
```

2. Install TypeScript

```bash
npm install -g typescript
```

3. Install dependencies

```bash
npm install
```

4. Start it

```bash
npm run dev
```

5. Visit [http://127.0.0.1:8080](http://127.0.0.1:8080)

## How to run backend

1. CD into folder

```bash
cd backend
```

2. Install Python packages

```bash
pip install fastapi, tensorflow, pytorch, uvicorn
```

3. Start it
```bash
python -m uvicorn main:app --reload
```