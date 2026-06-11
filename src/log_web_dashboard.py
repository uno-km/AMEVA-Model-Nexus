import sqlite3
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="AMEVA Lightweight Web Dashboard")
DB_PATH = "ameva_universal_logs.db"

# 세련된 다크모드 + 글래스모피즘(Glassmorphism) + 마이크로 애니메이션이 적용된 초경량 HTML/CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AMEVA Log Nexus</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        
        :root {
            --bg: #0f172a;
            --surface: rgba(30, 41, 59, 0.7);
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --accent: #38bdf8;
            --err: #f43f5e;
            --warn: #fbbf24;
            --bot: #c084fc;
        }

        body {
            margin: 0;
            padding: 30px;
            background-color: var(--bg);
            background-image: radial-gradient(circle at 50% -20%, #1e293b, #0f172a);
            color: var(--text);
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
            min-height: 100vh;
            box-sizing: border-box;
        }

        h1 {
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 25px;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .dashboard {
            background: var(--surface);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 0;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            height: calc(100vh - 120px);
            overflow-y: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            padding: 16px 24px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        th {
            color: var(--text-dim);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 1.5px;
            position: sticky;
            top: 0;
            background: rgba(30, 41, 59, 0.95);
            backdrop-filter: blur(10px);
            z-index: 10;
        }

        tr {
            transition: background 0.2s ease;
        }

        tr:hover {
            background: rgba(255,255,255,0.03);
        }

        .level-INFO { color: #34d399; font-weight: 600; text-shadow: 0 0 10px rgba(52, 211, 153, 0.3); }
        .level-ERROR { color: var(--err); font-weight: 600; text-shadow: 0 0 10px rgba(244, 63, 94, 0.3); }
        .level-WARN { color: var(--warn); font-weight: 600; }
        .level-BOTTLENECK { color: var(--bot); font-weight: 600; text-shadow: 0 0 10px rgba(192, 132, 252, 0.3); }

        .payload {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            color: var(--text-dim);
            font-size: 13px;
            white-space: pre-wrap;
            word-break: break-all;
        }
        
        .pulse {
            display: inline-block;
            width: 12px;
            height: 12px;
            background-color: var(--accent);
            border-radius: 50%;
            box-shadow: 0 0 15px var(--accent);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(56, 189, 248, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
        }
        
        /* 스크롤바 디자인 */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
    </style>
</head>
<body>
    <h1><span class="pulse"></span> AMEVA Log Nexus (Live Web)</h1>
    <div class="dashboard">
        <table>
            <thead>
                <tr>
                    <th style="width: 80px;">ID</th>
                    <th style="width: 200px;">Time</th>
                    <th style="width: 200px;">Source</th>
                    <th style="width: 120px;">Level</th>
                    <th>Payload</th>
                </tr>
            </thead>
            <tbody id="log-body">
            </tbody>
        </table>
    </div>

    <script>
        async function fetchLogs() {
            try {
                // 초경량 API 호출
                const res = await fetch('/api/logs');
                const logs = await res.json();
                
                let html = '';
                for (const log of logs) {
                    html += `
                        <tr>
                            <td style="color: var(--text-dim)">#${log.log_id}</td>
                            <td>${log.timestamp.replace('T', ' ').substring(0, 23)}</td>
                            <td style="color: var(--accent)">${log.source}</td>
                            <td class="level-${log.level}">${log.level}</td>
                            <td class="payload">${log.payload_json}</td>
                        </tr>
                    `;
                }
                document.getElementById('log-body').innerHTML = html;
            } catch (e) {
                console.error(e);
            }
        }

        // 페이지 로드 시 즉시 렌더링 후, 1초마다 자동 갱신
        fetchLogs();
        setInterval(fetchLogs, 1000);
    </script>
</body>
</html>
"""

@app.get("/")
def home():
    """HTML 템플릿 반환 (프론트엔드)"""
    return HTMLResponse(content=HTML_TEMPLATE)

@app.get("/api/logs")
def api_logs(limit: int = 100):
    """DB에서 최근 로그를 조회하여 JSON 반환 (백엔드)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # 가장 최근 로그 limit 개 조회
        cursor.execute("SELECT * FROM universal_logs ORDER BY log_id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []

if __name__ == "__main__":
    # 포트 14000 지정
    uvicorn.run(app, host="0.0.0.0", port=14000)
