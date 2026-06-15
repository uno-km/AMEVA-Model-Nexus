import asyncio
import aiohttp
import time
import uuid
import sys

URL = "http://127.0.0.1:10003/log/push"

# The goal is "10억건 느낌" (Feeling of 1 billion).
# We'll bombard with 500 concurrent connections, each sending 1000 logs.
# Total = 500,000 logs in a matter of seconds.
CONCURRENCY = 500
BATCHES = 1000

async def stress_worker(session, worker_id):
    for i in range(BATCHES):
        payload = {
            "source": f"StressWorker-{worker_id}",
            "level": "INFO" if i % 100 != 0 else "ERROR",
            "payload": {
                "msg": "This is a heavy stress test message to simulate a massive cluster.",
                "seq": i,
                "random_uuid": str(uuid.uuid4())
            }
        }
        try:
            async with session.post(URL, json=payload) as resp:
                await resp.read()
        except Exception:
            pass # Ignore connection resets under extreme load

async def main():
    print(f"🚀 Starting MASSIVE Stress Test")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Logs per worker: {BATCHES}")
    print(f"Target Total Logs: {CONCURRENCY * BATCHES}")
    print("Hold on to your butts...")
    
    start = time.time()
    
    # Use a large TCP connector limit
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 100)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(CONCURRENCY):
            tasks.append(asyncio.create_task(stress_worker(session, i)))
        await asyncio.gather(*tasks)
        
    elapsed = time.time() - start
    total_reqs = CONCURRENCY * BATCHES
    throughput = total_reqs / elapsed if elapsed > 0 else 0
    
    print(f"\n✅ Finished {total_reqs} requests in {elapsed:.2f} seconds.")
    print(f"⚡ Throughput: {throughput:.2f} requests/sec")

if __name__ == "__main__":
    # Windows specific fix for asyncio Too Many Open Files / Proactor
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
