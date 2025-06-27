import asyncio
from playwright.async_api import async_playwright

async def test_setup():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("Testing backend...")
        try:
            # Test backend root endpoint
            response = await page.goto("http://localhost:8000")
            if response and response.ok:
                data = await response.json()
                print(f"✓ Backend is running: {data}")
            else:
                print("✗ Backend is not accessible")
        except Exception as e:
            print(f"✗ Backend error: {e}")
        
        print("\nTesting frontend...")
        try:
            # Test frontend
            response = await page.goto("http://localhost:3000")
            if response and response.ok:
                title = await page.title()
                print(f"✓ Frontend is running: {title}")
            else:
                print("✗ Frontend is not accessible")
        except Exception as e:
            print(f"✗ Frontend error: {e}")
        
        print("\nTesting Neo4j browser...")
        try:
            # Test Neo4j browser
            response = await page.goto("http://localhost:7475")
            if response and response.ok:
                print("✓ Neo4j browser is accessible")
            else:
                print("✗ Neo4j browser is not accessible")
        except Exception as e:
            print(f"✗ Neo4j error: {e}")
        
        print("\nTesting Neo4j connection from backend...")
        try:
            # Test Neo4j connection through backend
            response = await page.goto("http://localhost:8000/test-neo4j")
            if response and response.ok:
                data = await response.json()
                print(f"✓ Neo4j connection successful: {data}")
            else:
                print("✗ Neo4j connection failed")
        except Exception as e:
            print(f"✗ Neo4j connection error: {e}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_setup())