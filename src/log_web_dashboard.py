import sqlite3
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="AMEVA Ultra-Light Dashboard")
DB_PATH = "ameva_universal_logs.db"

# 순수 HTML 뼈대와 필수 CSS만 남긴 초경량 템플릿
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AMEVA Log Nexus</title>
    <style>
        body { font-family: Consolas, monospace; background: #000; color: #ddd; margin: 10px; font-size: 13px; }
        table { width: 100%; border-collapse: collapse; table-layout: fixed; }
        th, td { border: 1px solid #333; padding: 4px; text-align: left; word-wrap: break-word; }
        th { background: #222; position: sticky; top: 0; }
        th:nth-child(1) { width: 80px; }
        th:nth-child(2) { width: 160px; }
        th:nth-child(3) { width: 150px; }
        th:nth-child(4) { width: 100px; }
        .INFO { color: #0f0; }
        .ERROR { color: #f00; font-weight: bold; }
        .WARN { color: #ff0; }
        .BOTTLENECK { color: #f0f; }
        h2 { margin: 0 0 10px 0; font-size: 16px; color: #fff; }
        #sys-info { color: #0ff; }
    </style>
</head>
<body>
    <h2>
        [ AMEVA Log Nexus ] | System: <span id="sys-info">Detecting...</span>
    </h2>
    <table>
        <thead>
            <tr><th>ID</th><th>Time</th><th>Source</th><th>Level</th><th>Payload</th></tr>
        </thead>
        <tbody id="log-body"></tbody>
    </table>

    <script>
        async function fetchSystem() {
            try {
                const res = await fetch('/api/system');
                const sys = await res.json();
                const infoText = sys.mode === "GPU" ? `GPU Mode (${sys.device_name})` : "CPU Mode";
                document.getElementById('sys-info').innerText = infoText;
            } catch (e) {
                console.error(e);
            }
        }

        async function fetchLogs() {
            try {
                const res = await fetch('/api/logs');
                const logs = await res.json();
                
                let html = '';
                for (const log of logs) {
                    html += `
                        <tr>
                            <td style="color: #777">${log.log_id}</td>
                            <td>${log.timestamp.replace('T', ' ').substring(0, 19)}</td>
                            <td style="color: #0af">${log.source}</td>
                            <td class="${log.level}">${log.level}</td>
                            <td>${log.payload_json}</td>
                        </tr>
                    `;
                }
                document.getElementById('log-body').innerHTML = html;
            } catch (e) {
                console.error(e);
            }
        }

        // 초기 시스템 정보 로드 (1번만)
        fetchSystem();
        
        // 로그 로드 및 1초 주기 갱신
        fetchLogs();
        setInterval(fetchLogs, 1000);
    </script>
</body>
</html>
"""

@app.get("/")
def home():
    return HTMLResponse(content=HTML_TEMPLATE)

@app.get("/api/system")
def api_system():
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "mode": "GPU",
                "device_name": torch.cuda.get_device_name(0)
            }
        else:
            return {"mode": "CPU", "device_name": "Generic"}
    except ImportError:
        return {"mode": "CPU", "device_name": "Generic"}

@app.get("/api/logs")
def api_logs(limit: int = 100):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM universal_logs ORDER BY log_id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=14000)
